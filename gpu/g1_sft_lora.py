"""[GPU] LoRA 监督微调（只对 assistant 回复部分计算损失），结束后合并权重
用法：python gpu/g1_sft_lora.py [--load-4bit]
输入：data/<ds>/llm/sft.jsonl     输出：models/llm/<ds>/sft_adapter/ 与 sft_merged/
显存：Qwen2.5-7B bf16 + LoRA + 梯度检查点，单卡 cuda:0 实测峰值 16.8GiB（A10 22GiB）
速度：实测约 0.95s / micro-batch（2 条），4617 条样本约 40 分钟 / epoch
      直接 `python` 运行，不要用 torchrun；OOM 时加 --load-4bit 走 QLoRA
"""
import os
# 训练只用一张卡：必须在 import torch 之前限制可见 GPU。否则 Trainer 看到 2 张卡会自动用 nn.DataParallel
# 把模型复制到第二张卡，而实验机 GPU 间互传异常 → 训练第一步就卡死。
if os.environ.get("MULTI_GPU_SPLIT") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainerCallback, TrainingArguments)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, ds_dir, load_cfg  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_utils import load_kwargs, report  # noqa: E402
from model_path import resolve_model  # noqa: E402

MAXLEN = 768


def tokenize(ex, tok):
    prompt = tok.apply_chat_template(ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
    ans = ex["messages"][-1]["content"] + tok.eos_token
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(ans, add_special_tokens=False)["input_ids"]
    ids = (p + a)[:MAXLEN]
    labels = ([-100] * len(p) + a)[:MAXLEN]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    def pad(k, v):
        return torch.tensor([b[k] + [v] * (L - len(b[k])) for b in batch])
    return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", -100)}


class Guard(TrainerCallback):
    """出现 NaN/Inf 立即终止（不再白跑十几个小时）；每次日志附带每步耗时和预计剩余时间"""
    def on_train_begin(self, args, state, control, **kw):
        self.t0, self.s0 = time.time(), 0

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs or "loss" not in logs:
            return
        bad = [k for k in ("loss", "grad_norm") if k in logs and not math.isfinite(float(logs[k]))]
        if bad or float(logs["loss"]) > 50:
            print(f"!! 训练数值发散：{logs}，已终止。请把完整输出发回。", flush=True)
            control.should_training_stop = True
            raise SystemExit(3)
        dt = (time.time() - self.t0) / max(1, state.global_step - self.s0)
        left = (state.max_steps - state.global_step) * dt / 3600
        print(f"[time] {dt:.1f}s/step（{args.gradient_accumulation_steps} 个 micro-batch）  "
              f"step {state.global_step}/{state.max_steps}  预计剩余 {left:.1f}h", flush=True)


class MPTrainer(Trainer):
    """多卡模型并行下自己算 loss：把 labels 移到 logits 所在的卡，按 token 求平均。
    （绕开 transformers 4.46 内部用 num_items_in_batch 归一化时可能出现的跨卡张量问题）"""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        out = model(**inputs)
        logits = out.logits[:, :-1, :].float()
        tgt = labels[:, 1:].to(logits.device)
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                               ignore_index=-100, reduction="sum")
        # transformers 4.46：传入 num_items_in_batch（整个梯度累积组的有效 token 数）时，Trainer 不再除以
        # grad_accum，需要这里按它归一化；否则（如评估）按本批 token 数求平均
        denom = num_items_in_batch if num_items_in_batch is not None else (tgt != -100).sum()
        loss = ce / torch.as_tensor(denom, device=ce.device, dtype=ce.dtype).clamp(min=1)
        if self.state.global_step < 3 and model.training:
            print(f"[debug] step {self.state.global_step} ce_sum={ce.item():.3f} denom={float(denom):.0f} "
                  f"loss={loss.item():.4f} tokens={int((tgt != -100).sum())}", flush=True)
        return (loss, out) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1, help="快速测试：只训练这么多步（不合并、不覆盖正式结果）")
    ap.add_argument("--epochs", type=float, default=None)
    a = ap.parse_args()
    cfg = load_cfg()
    lc, sc = cfg["llm"], cfg["llm"]["sft"]
    test = a.max_steps > 0
    lc["base_model"] = resolve_model(lc["base_model"])
    print("[model]", lc["base_model"], flush=True)
    # LLM_TAG（如 llama）：模型目录改为 models/llm/<ds>_<tag>，不覆盖 Qwen 的模型与 sft_log.json
    tag = os.environ.get("LLM_TAG", "")
    outdir = ROOT / "models" / "llm" / (cfg["_ds"] + (f"_{tag}" if tag else ""))
    outdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(lc["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_quant_type="nf4") if a.load_4bit else None
    # 单卡训练（实验机上跨卡切分会导致前向结果错误，见 gpu_utils.load_kwargs 说明）
    attn = os.environ.get("ATTN_IMPL", "sdpa")  # 若怀疑注意力内核导致 NaN，可设 ATTN_IMPL=eager 对比
    model = AutoModelForCausalLM.from_pretrained(lc["base_model"], attn_implementation=attn,
                                                 **load_kwargs("train", q, lc["base_model"]))
    print("[model] attn_implementation =", attn, "dtype =", next(model.parameters()).dtype, flush=True)
    report(model)
    if a.load_4bit:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=sc["lora_r"], lora_alpha=sc["lora_alpha"], lora_dropout=sc["lora_dropout"], task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for n_, p_ in model.named_parameters():  # LoRA 参数用 fp32 训练（bf16 下 Adam 易数值不稳）
        if p_.requires_grad:
            p_.data = p_.data.float()
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(ds_dir(cfg) / "llm" / "sft.jsonl", encoding="utf-8")]
    ds = Dataset.from_list(rows).shuffle(seed=0)
    ds = ds.map(lambda e: tokenize(e, tok), remove_columns=ds.column_names)
    split = ds.train_test_split(test_size=0.03, seed=0)
    lens = sorted(len(x) for x in split["train"]["input_ids"])
    print(f"[data] train={len(split['train'])} eval={len(split['test'])}  tokens/样本 中位数={lens[len(lens) // 2]} 最大={lens[-1]}",
          flush=True)

    # 训练前自检：未训练模型在答案部分的 loss 应在 0.3–3 之间；若已是 NaN/极大，问题在前向（切分/注意力），不在优化
    model.eval()
    with torch.no_grad():
        b = collate([split["train"][i] for i in range(2)], tok.pad_token_id)
        dev = next(model.parameters()).device
        out = model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev))
        lg = out.logits[:, :-1].float()
        tg = b["labels"][:, 1:].to(lg.device)
        l0 = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1), ignore_index=-100)
    print(f"[check] 初始 loss = {l0.item():.4f}  logits 有限={bool(torch.isfinite(lg).all())}  "
          f"|logits|max={lg.abs().max().item():.1f}", flush=True)
    if not math.isfinite(l0.item()) or l0.item() > 20:
        sys.exit("!! 训练前前向就不正常，未开始训练。请把以上输出发回（可试 ATTN_IMPL=eager）。")
    model.train()

    args = TrainingArguments(
        output_dir=str(outdir / "sft_ckpt"), num_train_epochs=a.epochs or sc["epochs"], learning_rate=sc["lr"],
        max_steps=a.max_steps, max_grad_norm=1.0, logging_first_step=True,
        per_device_train_batch_size=sc["batch_size"], gradient_accumulation_steps=sc["grad_accum"],
        per_device_eval_batch_size=sc["batch_size"], lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=1 if test else 10, eval_strategy="no" if test else "epoch", save_strategy="no", report_to="none",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False)
    tr = MPTrainer(model=model, args=args, train_dataset=split["train"], eval_dataset=split["test"],
                 data_collator=lambda b: collate(b, tok.pad_token_id), callbacks=[Guard()])
    print(f"[gpu] Trainer 可见 GPU 数 n_gpu={args.n_gpu}（应为 1）  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    tr.train()
    if test:
        print("[test] 快速测试完成：loss 正常下降、无 NaN 即可正式运行（未保存模型）", flush=True)
        return
    json.dump(tr.state.log_history, open(outdir / "sft_log.json", "w"), indent=1)
    model.save_pretrained(outdir / "sft_adapter")

    # 合并 LoRA -> 完整模型（供生成与下一步 DPO 作为起点/参考模型）
    del model, tr
    torch.cuda.empty_cache()
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(lc["base_model"], torch_dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, outdir / "sft_adapter").merge_and_unload()
    merged.save_pretrained(outdir / "round0_merged", safe_serialization=True)
    tok.save_pretrained(outdir / "round0_merged")
    print("saved", outdir / "round0_merged")


if __name__ == "__main__":
    main()
