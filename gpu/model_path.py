"""把 config.yaml 里的 base_model 解析成真正含有 config.json 的本地目录

支持以下几种常见放法（都能自动找到）：
  model/Qwen2.5-7B-Instruct/config.json                                  （标准）
  model/Qwen2.5-7B-Instruct/Qwen/Qwen2.5-7B-Instruct/config.json          （modelscope / 多套了一层）
  model/Qwen2.5-7B-Instruct/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<hash>/config.json （HF 缓存格式）
若写的是 HF 仓库名（如 Qwen/Qwen2.5-7B-Instruct）则原样返回，由 transformers 联网下载。
命令行：python gpu/model_path.py <路径或仓库名>   → 打印解析后的路径；找不到时打印诊断并以非 0 退出
"""
import os
import sys
from pathlib import Path

WEIGHT_GLOBS = ("*.safetensors", "*.bin")


def _has_weights(d: Path) -> bool:
    return any(any(d.glob(g)) for g in WEIGHT_GLOBS)


def _diagnose(p: Path) -> str:
    msg = [f"找不到模型目录：{p}"]
    q = p
    while not q.exists() and q != q.parent:
        q = q.parent
    msg.append(f"  最近一级存在的上级目录：{q}")
    try:
        items = sorted(os.listdir(q))[:30]
        msg.append("  其中内容：" + (", ".join(items) if items else "（空）"))
    except OSError as e:
        msg.append(f"  无法列出：{e}")
    msg.append("  请检查：路径拼写/大小写、是否在实验机（而不是本机）上、模型是否下载完整（需要 config.json、"
               "tokenizer 文件和 *.safetensors 权重）。")
    return "\n".join(msg)


def resolve_model(name: str) -> str:
    name = str(name).strip().strip('"').strip("'").rstrip("/")
    p = Path(os.path.expanduser(name))
    looks_local = p.is_absolute() or name.startswith((".", "~")) or p.exists()
    if not looks_local:
        return name  # HF 仓库名
    if not p.exists():
        raise FileNotFoundError(_diagnose(p))
    if (p / "config.json").is_file():
        cand = p
    else:
        found = sorted(c.parent for c in p.glob("**/config.json")
                       if len(c.relative_to(p).parts) <= 6 and _has_weights(c.parent))
        if not found:
            raise FileNotFoundError(_diagnose(p / "config.json") + "\n  （目录存在，但下面没有 config.json + 权重文件）")
        cand = found[0]
        print(f"[model] 在子目录中找到模型：{cand}", file=sys.stderr, flush=True)
    if not _has_weights(cand):
        raise FileNotFoundError(f"{cand} 有 config.json 但没有 *.safetensors / *.bin 权重，模型可能没下载完整")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 本地模型不再联网
    return str(cand.resolve())


if __name__ == "__main__":
    try:
        print(resolve_model(sys.argv[1]))
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
