"""GPU 加载设置（2 × A10 24GB）
【当前默认】训练与 HF 推理都只用单卡 cuda:0；生成（vLLM）在两张卡上各起一个独立进程，卡间不通信。
以下为早期的跨卡切分说明，仅在 MULTI_GPU_SPLIT=1 时生效：

策略：单进程「模型并行」——用 device_map 把 7B 模型的各层切分到所有可见 GPU 上
  - 训练（SFT / DPO）：每卡只放约 55% 显存的权重（7B bf16 ≈ 15GB → 每卡约 7.6GB），其余留给激活/梯度/优化器
  - 推理（HF 回退生成）：每卡放到 85%
  - vLLM：张量并行 tensor_parallel_size = GPU 数
注意：**不要用 torchrun / accelerate launch 启动**，直接 `python gpu/xxx.py` 即可；
      多进程数据并行（DDP）会让每张卡各放一整份模型，A10 上更容易 OOM。
"""
import json
import os
from pathlib import Path

import torch


def n_gpus() -> int:
    return torch.cuda.device_count()


def describe() -> str:
    if not torch.cuda.is_available():
        return "CUDA 不可用！请检查驱动 / CUDA_VISIBLE_DEVICES"
    return " | ".join(f"cuda:{i} {torch.cuda.get_device_name(i)} "
                      f"{torch.cuda.get_device_properties(i).total_memory / 2**30:.0f}GiB"
                      for i in range(n_gpus()))


def max_memory(mode: str = "train"):
    """返回 from_pretrained(max_memory=...) 用的字典；mode: train | infer
    可用环境变量 GPU_WEIGHT_FRAC 覆盖每卡权重占比（例如 0.5）"""
    n = n_gpus()
    if n == 0:
        raise RuntimeError(describe())
    frac = float(os.environ.get("GPU_WEIGHT_FRAC", 0.55 if mode == "train" else 0.85))
    mm = {}
    for i in range(n):
        tot = torch.cuda.get_device_properties(i).total_memory / 2**30
        mm[i] = f"{max(1, int(tot * frac))}GiB"
    return mm


def split_device_map(model_path: str, n: int):
    """训练用的显式切分：embed_tokens / norm / rotary_emb / lm_head 都放在 cuda:0（和 Trainer 送入的
    input_ids、labels 同一张卡，loss 不会跨卡），解码层按显存均衡切到各卡（cuda:0 少放几层以抵消 embed+lm_head）。
    仅对 Qwen2/LLaMA 类结构（model.layers.N）生效；其他结构返回 None 走 "auto"。"""
    try:
        c = json.load(open(Path(model_path) / "config.json", encoding="utf-8"))
        L = int(c["num_hidden_layers"])
    except Exception:
        return None
    if c.get("model_type") not in ("qwen2", "llama", "mistral") or n < 2:
        return None
    vocab, hid = c.get("vocab_size", 0), c.get("hidden_size", 1)
    inter = c.get("intermediate_size", 4 * hid)
    layer = 4 * hid * hid + 3 * hid * inter                       # 每层参数量（近似）
    extra = vocab * hid * (1 if c.get("tie_word_embeddings") else 2)  # embed + lm_head
    # cuda:0 承担 extra，按“参数量”均衡：sum = L*layer + extra，每卡目标 = sum/n
    tgt = (L * layer + extra) / n
    n0 = max(1, min(L - (n - 1), int(round((tgt - extra) / layer))))
    rest = L - n0
    counts = [n0] + [rest // (n - 1) + (1 if i < rest % (n - 1) else 0) for i in range(n - 1)]
    dm, i = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 0, "lm_head": 0}, 0
    for g, k in enumerate(counts):
        for _ in range(k):
            dm[f"model.layers.{i}"] = g
            i += 1
    return dm


def load_kwargs(mode: str = "train", quant=None, model_path: str = None) -> dict:
    """统一的 from_pretrained 关键字参数（bf16）。
    默认整模型放在单卡 cuda:0：实验机诊断（diag_forward.py）表明，该服务器上模型跨卡切分时
    前向结果错误（答案 loss≈13，单卡为≈1.1）且慢约 20 倍（疑似 GPU 间 P2P 传输异常）；
    单卡 LoRA 训练峰值 16.8GiB，A10 22GiB 放得下。
    MULTI_GPU_SPLIT=1 可恢复跨卡切分（仅在确认多卡互传正常的机器上使用）。"""
    if n_gpus() == 0 and os.environ.get("IDS_CPU_SMOKE") == "1":  # 仅用于无 GPU 的冒烟测试
        return dict(torch_dtype=torch.float32)
    if os.environ.get("MULTI_GPU_SPLIT") != "1":
        kw = dict(torch_dtype=torch.bfloat16, device_map={"": 0})
        print(f"[gpu] {describe()}  -> 单卡 cuda:0（不跨卡切分）", flush=True)
    else:
        dm = split_device_map(model_path, n_gpus()) if (mode == "train" and model_path) else None
        kw = dict(torch_dtype=torch.bfloat16, device_map=dm or "auto")
        if dm is None:
            kw["max_memory"] = max_memory(mode)
        print(f"[gpu] {describe()}  -> 跨卡切分 device_map={'explicit' if dm else 'auto'}", flush=True)
    if quant is not None:
        kw["quantization_config"] = quant
    return kw


def report(model):
    m = getattr(model, "hf_device_map", None)
    if m:
        from collections import Counter
        print("[gpu] layer placement:", dict(Counter(str(v) for v in m.values())), flush=True)
    for i in range(n_gpus()):
        print(f"[gpu] cuda:{i} allocated {torch.cuda.memory_allocated(i) / 2**30:.1f}GiB", flush=True)
