"""[GPU] 用 LLM 为稀有类生成候选样本（每个 prompt 采样 m 条，便于构造 DPO 偏好对）
用法：
  python gpu/g2_generate.py --model <模型目录或HF名> --name llm_base   # 未微调基线（旧 B 组）
  python gpu/g2_generate.py --model models/llm/cse2018/round0_merged --name llm_sft --round 0
引擎：默认 vLLM，且每张 GPU 各跑一个独立进程分摊 prompt（VLLM_MODE=dp，默认）；\n      VLLM_MODE=tp 为双卡张量并行；未安装 vLLM 时回退到 transformers（多卡模型并行）。
输出：data/<ds>/llm/gen_<name>.jsonl（原始文本+解析结果）
      data/<ds>/synth/<name>.csv（解析成功的样本，已统一后处理，含 raw_valid / prompt_id 列）
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ds_dir, finalize_synth, load_cfg, load_split, rare_classes, synth_dir, validity_mask  # noqa
from model_path import resolve_model  # noqa: E402
from prompts import PRIMARY, anchor_schedule, anchor_shots, build_messages, class_knn, parse, sample_shots  # noqa: E402


def _visible_gpus():
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v:
        return [x.strip() for x in v.split(",") if x.strip()]
    import torch
    return [str(i) for i in range(torch.cuda.device_count())]


def _vllm_engine(model, tp, seed):
    from vllm import LLM
    kw = dict(model=model, dtype="bfloat16", tensor_parallel_size=tp, max_model_len=2048, seed=seed,
              gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", 0.90)))
    if tp > 1:
        kw["disable_custom_all_reduce"] = True  # PCIe A10 上自定义 all-reduce 容易卡死
    return LLM(**kw)


def _vllm_run(llm, prompts, m, gc, max_new, seed, tag=""):
    from vllm import SamplingParams
    # 注意：不要给 SamplingParams 传 seed——vLLM 0.6.x 中逐请求固定种子会让同一 prompt 的 n 条采样完全相同、
    # 且不同 prompt 之间的随机数也相同（实测 75–95% 的 prompt 4 条输出一模一样）。可复现性由 LLM(seed=...) 保证。
    sp = SamplingParams(n=m, temperature=gc["temperature"], top_p=gc["top_p"], max_tokens=max_new)
    res, chunk, t0 = [], 250, time.time()
    for i in range(0, len(prompts), chunk):  # 分块生成，便于在集群输出里看到进度
        outs = llm.generate(prompts[i:i + chunk], sp, use_tqdm=False)
        res += [[c.text for c in o.outputs] for o in outs]
        print(f"  {tag}{min(i + chunk, len(prompts))}/{len(prompts)} prompts  ({time.time() - t0:.0f}s)", flush=True)
    return res


def gen_vllm(model, prompts, m, gc, max_new):
    """默认 dp 模式：每张 GPU 各起一个独立的 vLLM 进程（单卡放得下 7B bf16），各生成一半 prompt。
    卡间不需要任何通信，避免了 PCIe A10 上 NCCL/张量并行初始化卡死的问题，速度也是 tp 的约 2 倍。
    VLLM_MODE=tp 时改用张量并行（单卡显存不够时才需要）。"""
    gpus = _visible_gpus()
    mode = os.environ.get("VLLM_MODE", "dp")
    if mode == "tp" or len(gpus) <= 1:
        tp = 2 if len(gpus) >= 2 else 1
        if tp > 1:
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        print(f"[vllm] mode=tp tensor_parallel_size={tp}", flush=True)
        return _vllm_run(_vllm_engine(model, tp, 0), prompts, m, gc, max_new, 0)

    print(f"[vllm] mode=dp：{len(gpus)} 个独立进程，GPU={gpus}", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="g2_dp_", dir=str(Path(__file__).resolve().parents[1])))
    json.dump({"model": model, "prompts": prompts, "m": m, "gc": gc, "max_new": max_new},
              open(tmp / "job.json", "w", encoding="utf-8"))
    timeout = float(os.environ.get("GEN_TIMEOUT_MIN", 240)) * 60
    procs = []
    for k, g in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        procs.append(subprocess.Popen([sys.executable, __file__, "--worker", str(k), str(len(gpus)), str(tmp)], env=env))
    t0, failed = time.time(), False
    while any(p.poll() is None for p in procs):
        if time.time() - t0 > timeout:
            for p in procs:
                p.kill()
            sys.exit(f"!! 生成超过 GEN_TIMEOUT_MIN={timeout / 60:.0f} 分钟，已终止（请把输出日志发回检查）")
        time.sleep(10)
    for k, p in enumerate(procs):
        if p.returncode != 0:
            print(f"!! worker {k} (GPU {gpus[k]}) 退出码 {p.returncode}", flush=True)
            failed = True
    if failed:
        sys.exit("!! dp 生成失败；可尝试 VLLM_MODE=tp 或 --engine hf")
    res = [None] * len(prompts)
    for k in range(len(gpus)):
        part = json.load(open(tmp / f"out_{k}.json", encoding="utf-8"))
        for idx, txt in zip(range(k, len(prompts), len(gpus)), part):
            res[idx] = txt
    shutil.rmtree(tmp, ignore_errors=True)
    return res


def worker(k, n, tmp):
    job = json.load(open(Path(tmp) / "job.json", encoding="utf-8"))
    prompts = job["prompts"][k::n]
    print(f"[worker {k}] GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} 负责 {len(prompts)} 个 prompt", flush=True)
    llm = _vllm_engine(job["model"], 1, k)
    out = _vllm_run(llm, prompts, job["m"], job["gc"], job["max_new"], k, tag=f"[worker {k}] ")
    json.dump(out, open(Path(tmp) / f"out_{k}.json", "w", encoding="utf-8"), ensure_ascii=False)


def gen_hf(model, tok, prompts, m, gc, max_new):
    import torch
    from transformers import AutoModelForCausalLM
    from gpu_utils import load_kwargs
    mdl = AutoModelForCausalLM.from_pretrained(model, **load_kwargs("infer")).eval()
    tok.padding_side = "left"
    res = []
    bs = max(1, gc["batch_size"] // m)
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False,
                  return_token_type_ids=False).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**enc, do_sample=True, temperature=gc["temperature"], top_p=gc["top_p"],
                               max_new_tokens=max_new, num_return_sequences=m, pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        res += [txt[j * m:(j + 1) * m] for j in range(len(txt) // m)]
        print(f"  {min(i + bs, len(prompts))}/{len(prompts)}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--round", type=int, default=-1)
    ap.add_argument("--m", type=int, default=4, help="每个 prompt 采样条数")
    ap.add_argument("--n-per-class", type=int, default=None)
    ap.add_argument("--engine", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.model = resolve_model(a.model)
    print("[model]", a.model, flush=True)
    cfg = load_cfg()
    lc, gc = cfg["llm"], cfg["llm"]["gen"]
    n_per = a.n_per_class or cfg["datasets"].get(cfg["_ds"], {}).get("gen_n_per_class", gc["n_per_class"])
    train = load_split(cfg, "train")
    rare = rare_classes(cfg, train)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rng = np.random.default_rng(a.seed)
    meta, prompts = [], []
    mode = lc.get("prompt_mode", "anchor")  # anchor（v2 默认）| random（v1：随机抽同类示例）
    for c in rare:
        g = train[train.label == c].reset_index(drop=True)
        n_prompts = int(np.ceil(n_per / a.m))
        if mode == "anchor":
            knn = class_knn(g, 2 * lc["shots"])
            anchors = anchor_schedule(len(g), n_prompts, rng)
        for i in range(n_prompts):
            if mode == "anchor":
                shots = anchor_shots(g, knn, int(anchors[i]), lc["shots"], rng, include_anchor=True)
            else:
                shots = sample_shots(g, lc["shots"], rng)
            msgs = build_messages(c, shots)
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            meta.append({"prompt_id": len(meta), "label": c, "anchor": int(anchors[i]) if mode == "anchor" else -1})
    print(f"{len(prompts)} prompts x {a.m} samples, classes={rare}, prompt_mode={mode}")

    engine = a.engine
    if engine == "auto":
        try:
            import vllm  # noqa: F401
            engine = "vllm"
        except ImportError:
            engine = "hf"
    texts = gen_vllm(a.model, prompts, a.m, gc, lc["max_new_tokens"]) if engine == "vllm" else \
        gen_hf(a.model, tok, prompts, a.m, gc, lc["max_new_tokens"])

    rows = []
    llm_dir = ds_dir(cfg) / "llm"
    llm_dir.mkdir(exist_ok=True)
    with open(llm_dir / f"gen_{a.name}.jsonl", "w", encoding="utf-8") as f:
        for mt, p, outs in zip(meta, prompts, texts):
            for j, t in enumerate(outs):
                d = parse(t)
                rec = {**mt, "sample_id": j, "prompt": p, "completion": t, "parsed": d}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if d is not None:
                    rows.append({**d, "label": mt["label"], "prompt_id": mt["prompt_id"], "sample_id": j})
    df = pd.DataFrame(rows, columns=PRIMARY + ["label", "prompt_id", "sample_id"])
    n_total = len(prompts) * a.m
    if len(df) == 0:
        print("!! 没有任何可解析的输出（模型未按 JSON 格式回答）。原始文本见", llm_dir / f"gen_{a.name}.jsonl")
    df["raw_valid"] = validity_mask(df).astype(int) if len(df) else pd.Series(dtype=int)
    df["round"] = a.round
    out = finalize_synth(df, a.name)
    out.to_csv(synth_dir(cfg) / f"{a.name}.csv", index=False)
    stats = {"name": a.name, "engine": engine, "n_requested": n_total, "parse_rate": len(df) / n_total,
             "valid_rate_of_all": float(df["raw_valid"].sum() / n_total) if len(df) else 0.0,
             "per_class": out.label.value_counts().to_dict(),
             # 多样性自检：去重后占比；同一 prompt 的 m 条采样全部相同的比例（正常应接近 0）
             "unique_rate": float(len(df.drop_duplicates(subset=PRIMARY + ["label"])) / max(1, len(df))),
             "all_same_within_prompt": float(df.groupby("prompt_id").apply(
                 lambda g: len(g) == a.m and len(g.drop_duplicates(subset=PRIMARY)) == 1).mean()) if len(df) else 0.0}
    json.dump(stats, open(llm_dir / f"gen_{a.name}_stats.json", "w"), indent=1)
    print(stats)
    if stats["all_same_within_prompt"] > 0.3:
        print("!! 警告：超过 30% 的 prompt 多次采样结果完全相同，采样随机性可能有问题", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        main()
