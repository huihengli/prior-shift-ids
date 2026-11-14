"""[GPU] DPO 偏好优化（真正的强化学习式策略更新，替代旧稿中名不副实的 “RLHF”）
以上一轮合并模型为起点和参考模型，用 g3 产生的偏好对训练新 LoRA，训练后合并为下一轮模型。
用法：python gpu/g4_dpo.py --model models/llm/cse2018/round0_merged --pairs llm_sft --out round1 [--load-4bit]
输出：models/llm/<ds>/<out>_merged/
多卡：模型自动切分到所有可见 GPU；参考模型 = 同一模型关闭 LoRA，不额外占显存。直接 python 运行，不要 torchrun。
"""
import os
# 训练只用一张卡：必须在 import torch 之前限制可见 GPU。否则 Trainer 看到 2 张卡会自动用 nn.DataParallel
# 把模型复制到第二张卡，而实验机 GPU 间互传异常 → 训练第一步就卡死。
if os.environ.get("MULTI_GPU_SPLIT") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, ds_dir, load_cfg  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_utils import load_kwargs, report  # noqa: E402
from model_path import resolve_model  # noqa: E402
from g1_sft_lora import Guard  # noqa: E402  NaN/发散即终止 + 耗时估计


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True, help="g3 的 --out 名（读取 llm/pairs_<name>.jsonl）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-4bit", action="store_true")
    a = ap.parse_args()
    a.model = resolve_model(a.model)
    print("[model]", a.model, flush=True)
    cfg = load_cfg()
    lc, lp, sc = cfg["llm"], cfg["llm"]["loop"], cfg["llm"]["sft"]
    outdir = ROOT / "models" / "llm" / cfg["_ds"]

    rows = [json.loads(l) for l in open(ds_dir(cfg) / "llm" / f"pairs_{a.pairs}.jsonl", encoding="utf-8")]
    if not rows:
        sys.exit("没有偏好对，检查 g3 输出")
    ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows])
    print("pairs:", len(ds))

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_quant_type="nf4") if a.load_4bit else None
    model = AutoModelForCausalLM.from_pretrained(a.model, **load_kwargs("train", q, a.model))
    report(model)
    args = DPOConfig(output_dir=str(outdir / f"{a.out}_ckpt"), beta=lp["dpo_beta"], learning_rate=lp["dpo_lr"],
                     num_train_epochs=lp["dpo_epochs"], per_device_train_batch_size=1,
                     gradient_accumulation_steps=16, bf16=True, logging_steps=10, save_strategy="no",
                     report_to="none", gradient_checkpointing=True,
                     gradient_checkpointing_kwargs={"use_reentrant": False}, max_length=1024, max_prompt_length=768,
                     remove_unused_columns=False,
                     # RPO：在 DPO 损失上加 chosen 的负对数似然项，防止模型把两个回答的概率一起压低而丢失 JSON 格式
                     #（v2 第 2/3 轮 DPO loss 降到 0.29/0.04 后，生成解析率从 100% 跌到 12%/36%）
                     rpo_alpha=lp.get("dpo_rpo_alpha", None))
    peft_cfg = LoraConfig(r=sc["lora_r"], lora_alpha=sc["lora_alpha"], lora_dropout=sc["lora_dropout"],
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    # ref_model=None + peft：参考模型 = 关闭 LoRA 的起点模型（即上一轮模型），无需额外显存
    tr = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds, processing_class=tok,
                    peft_config=peft_cfg, callbacks=[Guard()])
    tr.train()
    json.dump(tr.state.log_history, open(outdir / f"{a.out}_dpo_log.json", "w"), indent=1)
    tr.model.save_pretrained(outdir / f"{a.out}_adapter")

    del tr, model
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, outdir / f"{a.out}_adapter").merge_and_unload()
    merged.save_pretrained(outdir / f"{a.out}_merged", safe_serialization=True)
    tok.save_pretrained(outdir / f"{a.out}_merged")
    print("saved", outdir / f"{a.out}_merged")


if __name__ == "__main__":
    main()
