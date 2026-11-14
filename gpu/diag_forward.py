"""[GPU 诊断，约 5–10 分钟] 定位“微调初始 loss≈14、随后 NaN、且很慢”的原因
用法（在 v2/ 下）：python gpu/diag_forward.py 2>&1 | tee diag.log      然后把 diag.log 和集群的 err 文件发回

依次测试（每项只做前向/少量反向，不训练、不保存任何东西）：
  0. 环境与 GPU 实测算力（bf16 矩阵乘 TFLOPS；A10 正常约 50–70）
  1. 模型文件与 tokenizer 自检（权重是否完整加载、lm_head 是否存在、特殊 token）
  2. 单卡 cuda:0 + sdpa：答案部分 loss、逐 token 预测、贪心生成一段
  3. 单卡 cuda:0 + eager 注意力：同上
  4. 双卡切分（训练用的切分方式）+ sdpa：同上 + 一次反向的耗时
正常结果：各项答案 loss 约 0.5–2.5，贪心生成是一条合法 JSON
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ds_dir, load_cfg  # noqa: E402
from model_path import resolve_model  # noqa: E402


CPU = os.environ.get("DIAG_CPU") == "1"  # 仅用于无 GPU 时检查脚本本身
DEV0 = "cpu" if CPU else 0


def hr(t):
    print("\n" + "=" * 20, t, "=" * 20, flush=True)


def tflops(dev):
    a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize(dev)
    t = time.time()
    for _ in range(20):
        a @ b
    torch.cuda.synchronize(dev)
    return 20 * 2 * 8192 ** 3 / (time.time() - t) / 1e12


def build_example(tok, cfg):
    from g1_sft_lora import tokenize
    rows = [json.loads(l) for l in open(ds_dir(cfg) / "llm" / "sft.jsonl", encoding="utf-8")][:4]
    exs = [tokenize(r, tok) for r in rows]
    return rows, exs


def answer_loss(model, ex, tok, tag, show=True):
    dev = model.get_input_embeddings().weight.device
    ids = torch.tensor([ex["input_ids"]], device=dev)
    lab = torch.tensor([ex["labels"]], device=dev)
    with torch.no_grad():
        out = model(input_ids=ids, labels=lab)
    lg = out.logits[0, :-1].float()
    tg = lab[0, 1:].to(lg.device)
    m = tg != -100
    ce = torch.nn.functional.cross_entropy(lg[m], tg[m])
    print(f"[{tag}] 答案 token 数={int(m.sum())}  我的 loss={ce.item():.4f}  transformers 自带 loss={out.loss.item():.4f}  "
          f"|logits|max={lg.abs().max().item():.1f}  有限={bool(torch.isfinite(lg).all())}", flush=True)
    if show:
        pred = lg[m].argmax(-1)
        print(f"[{tag}] 目标前 25 个 token：", repr(tok.decode(tg[m][:25])), flush=True)
        print(f"[{tag}] 模型预测前 25 个：  ", repr(tok.decode(pred[:25])), flush=True)
    return ce.item()


def greedy(model, tok, row, tag):
    msgs = row["messages"][:-1]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(p, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False)
    dev = model.get_input_embeddings().weight.device
    enc = {k: v.to(dev) for k, v in enc.items()}
    t = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
    txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[{tag}] 贪心生成（{time.time() - t:.1f}s）：{txt!r}", flush=True)


def load(path, device_map, attn):
    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_warning()  # 显示“权重未初始化”等关键警告
    t = time.time()
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32 if CPU else torch.bfloat16, device_map=device_map,
                                             attn_implementation=attn).eval()
    print(f"  加载 {time.time() - t:.0f}s  dtype={next(m.parameters()).dtype}  "
          f"device_map={dict(__import__('collections').Counter(str(v) for v in getattr(m, 'hf_device_map', {}).values()))}",
          flush=True)
    return m


def free(m):
    del m
    gc.collect()
    torch.cuda.empty_cache()


def batch_loss(model, batch, tag):
    dev = model.get_input_embeddings().weight.device
    with torch.no_grad():
        out = model(input_ids=batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev))
    lg = out.logits[:, :-1].float()
    tg = batch["labels"][:, 1:].to(lg.device)
    for i in range(lg.size(0)):
        m = tg[i] != -100
        print(f"[{tag}] 批内第 {i} 条（长度 {int(batch['attention_mask'][i].sum())}）答案 loss="
              f"{torch.nn.functional.cross_entropy(lg[i][m], tg[i][m]).item():.4f}", flush=True)


def train_step(model, batch, tag, peft_on=True):
    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if peft_on:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    model.train()
    dev = model.get_input_embeddings().weight.device
    if not CPU:
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
    for k in range(3):
        if not CPU:
            torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda" if not CPU else "cpu", dtype=torch.bfloat16, enabled=not CPU):
            out = model(input_ids=batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev))
        lg = out.logits[:, :-1].float()
        tg = batch["labels"][:, 1:].to(lg.device)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1), ignore_index=-100)
        loss.backward()
        gn = torch.sqrt(sum((p.grad.float() ** 2).sum().to(dev) for p in model.parameters() if p.grad is not None))
        if not CPU:
            torch.cuda.synchronize()
        print(f"[{tag}] 第 {k} 次 前向+反向 {time.time() - t:.2f}s  loss={loss.item():.4f}  grad_norm={gn.item():.3f}",
              flush=True)
        model.zero_grad(set_to_none=True)
    if not CPU:
        print(f"[{tag}] 峰值显存：" + ", ".join(f"cuda:{i} {torch.cuda.max_memory_allocated(i) / 2**30:.1f}GiB"
                                           for i in range(torch.cuda.device_count())), flush=True)
    return model


def run_case(cid, path, dmap, attn, kind, tok, rows, exs, batch):
    m = load(path, dmap, attn)
    try:
        if kind == "single":
            for i, e in enumerate(exs[:3]):
                answer_loss(m, e, tok, f"{cid}#{i}", show=(i == 0))
            greedy(m, tok, rows[0], cid)
        elif kind == "batch":
            answer_loss(m, exs[0], tok, f"{cid}-单条", show=False)
            batch_loss(m, batch, f"{cid}-批次")
        else:
            batch_loss(m, batch, f"{cid}-训练前")
            m = train_step(m, batch, cid)
    finally:
        del m  # 确保本项的模型在下一项加载前被释放


def main():
    import transformers
    cfg = load_cfg()
    path = resolve_model(cfg["llm"]["base_model"])

    hr("0. 环境")
    print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda,
          "| transformers", transformers.__version__, flush=True)
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"cuda:{i} {p.name} {p.total_memory / 2**30:.0f}GiB  cc={p.major}.{p.minor}  "
              f"bf16 matmul ≈ {tflops(f'cuda:{i}'):.0f} TFLOPS", flush=True)
    print("bf16 supported:", torch.cuda.is_bf16_supported(), flush=True)

    hr("1. 模型文件")
    c = json.load(open(Path(path) / "config.json"))
    print({k: c.get(k) for k in ("model_type", "architectures", "torch_dtype", "num_hidden_layers", "hidden_size",
                                  "vocab_size", "tie_word_embeddings", "rope_theta", "max_position_embeddings",
                                  "sliding_window", "use_sliding_window", "rope_scaling")}, flush=True)
    files = sorted(os.listdir(path))
    print("目录文件：", files, flush=True)
    idx = Path(path) / "model.safetensors.index.json"
    if idx.exists():
        wm = json.load(open(idx))["weight_map"]
        print(f"index 中权重数={len(wm)}  含 lm_head={'lm_head.weight' in wm}  含 embed={'model.embed_tokens.weight' in wm}",
              flush=True)
        miss = sorted(set(wm.values()) - set(files))
        print("index 引用但缺失的分片：", miss or "无", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"tokenizer: {type(tok).__name__} vocab={len(tok)} eos={tok.eos_token!r} pad={tok.pad_token!r} "
          f"chat_template={'有' if tok.chat_template else '无'}", flush=True)
    rows, exs = build_example(tok, cfg)
    ex = exs[0]
    n_ans = sum(1 for x in ex["labels"] if x != -100)
    print(f"样本 0：总 token={len(ex['input_ids'])}  答案 token={n_ans}", flush=True)
    print("答案原文：", repr(rows[0]["messages"][-1]["content"]), flush=True)

    from g1_sft_lora import collate
    batch = collate(exs[:2], tok.pad_token_id)  # 和训练时一样：2 条、右侧 padding、带 attention_mask
    only = os.environ.get("DIAG_ONLY", "")       # 例如 DIAG_ONLY=5 只跑第 5 项
    cases = [
        ("2", "单卡 + sdpa（单条，不 padding）", {"": DEV0}, "sdpa", "single"),
        ("3", "单卡 + sdpa，2 条 padding 批次（训练时的输入形式）", {"": DEV0}, "sdpa", "batch"),
        ("4", "单卡 + eager，2 条 padding 批次", {"": DEV0}, "eager", "batch"),
        ("5", "单卡 + LoRA + 梯度检查点（训练配置），前向+反向计时与显存", {"": DEV0}, "sdpa", "train"),
        ("6", "双卡切分（当前训练用的方式）+ sdpa，单条与批次", "split", "sdpa", "batch"),
        ("7", "双卡切分 + LoRA + 梯度检查点（复现当前训练）", "split", "sdpa", "train"),
    ]
    for cid, title, dmap, attn, kind in cases:
        if only and cid not in only.split(","):
            continue
        hr(f"{cid}. {title}")
        if dmap == "split":
            if torch.cuda.device_count() < 2:
                print("只有 1 张 GPU 可见，跳过")
                continue
            from gpu_utils import split_device_map
            dmap = split_device_map(path, 2)
        try:
            run_case(cid, path, dmap, attn, kind, tok, rows, exs, batch)
        except torch.OutOfMemoryError as e:
            print(f"[{cid}] OOM：{str(e).splitlines()[0]}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        if not CPU:
            print(f"[{cid}] 释放后各卡已用显存：" + ", ".join(
                f"cuda:{i} {torch.cuda.memory_allocated(i) / 2**30:.1f}GiB" for i in range(torch.cuda.device_count())),
                flush=True)
    hr("完成")


if __name__ == "__main__":
    main()
