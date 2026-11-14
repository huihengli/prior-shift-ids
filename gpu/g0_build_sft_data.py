"""[CPU · 轻量，可在本机跑] 构造 LoRA 监督微调数据（只用训练集，杜绝测试集泄漏）
每条样本 = (类别描述 + 目标样本的 k 个同类近邻) -> 目标样本（锚点式；prompt_mode=random 时为随机同类示例）
全部类别都参与（让模型学到一般流量结构）；稀有类每条真实样本只作为目标少数几次，防止背诵。
输出：data/<ds>/llm/sft.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ds_dir, load_cfg, load_split, rare_classes  # noqa: E402
from prompts import anchor_shots, build_messages, class_knn, row_json, sample_shots  # noqa: E402


def main():
    cfg = load_cfg()
    lc = cfg["llm"]
    train = load_split(cfg, "train")
    rare = rare_classes(cfg, train)
    rng = np.random.default_rng(0)
    out = ds_dir(cfg) / "llm"
    out.mkdir(exist_ok=True)
    n_written = {}
    with open(out / "sft.jsonl", "w", encoding="utf-8") as f:
        for c, g in train.groupby("label"):
            g = g.reset_index(drop=True)
            if len(g) < 2:
                continue
            # 稀有类：每条真实样本最多作为目标 rare_target_repeats 次（旧版为 20 次 × 3 epoch，导致模型背下训练样本：
            # 实测 SFT 后 92% 的生成与训练样本几乎相同，去重后只剩 6.7%）
            if c in rare:
                reps = lc["sft"].get("rare_target_repeats", 2)
                targets = np.repeat(np.arange(len(g)), reps)
            else:
                targets = rng.integers(len(g), size=lc["sft"].get("common_examples_per_class", 300))
            rng.shuffle(targets)
            n = len(targets)
            mode = lc.get("prompt_mode", "anchor")
            knn = class_knn(g, 2 * lc["shots"]) if mode == "anchor" else None
            for t in targets:
                t = int(t)
                # 与生成时一致的锚点式示例：示例为目标样本 t 的近邻（不含 t 本身），模型学的是“给定一个局部形态，生成其中的新样本”
                shots = anchor_shots(g, knn, t, lc["shots"], rng, include_anchor=False) if mode == "anchor" \
                    else sample_shots(g, lc["shots"], rng, exclude=[t])
                msgs = build_messages(c, shots) + [{"role": "assistant", "content": row_json(g.iloc[t])}]
                f.write(json.dumps({"label": c, "messages": msgs}, ensure_ascii=False) + "\n")
            n_written[c] = n
    print("SFT examples per class:", n_written)


if __name__ == "__main__":
    main()
