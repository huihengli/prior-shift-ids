"""[GPU] GReaT 基线（Borisov et al., ICLR 2023：用小型语言模型做表格数据生成）
审稿人会问“LLM 生成和已有的 LM 表格生成方法比如何”，这个基线用来回答。
用法：python gpu/g5_great.py [--epochs 100]
输出：data/<ds>/synth/great.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PRIMARY, exp_param, finalize_synth, load_cfg, load_split, rare_classes, stratified_cap, synth_dir  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--llm", default="distilgpt2")
    a = ap.parse_args()
    from be_great import GReaT
    cfg = load_cfg()
    train = load_split(cfg, "train")
    rare = rare_classes(cfg, train)
    sub = stratified_cap(train, 2000, 0)[PRIMARY + ["label"]]
    m = GReaT(llm=a.llm, epochs=a.epochs, batch_size=32, save_steps=100000,
              experiment_dir=str(Path(__file__).resolve().parents[1] / "models" / "great"))
    m.fit(sub)
    n = max(exp_param(cfg, "aug_target_sweep"))
    outs = []
    for c in rare:
        s = m.sample(n_samples=n, start_col="label", start_col_dist={c: 1.0}, device="cuda", max_length=200)
        s["label"] = c
        outs.append(s)
    out = finalize_synth(pd.concat(outs, ignore_index=True), "great")
    out.to_csv(synth_dir(cfg) / "great.csv", index=False)
    print(out.label.value_counts())


if __name__ == "__main__":
    main()
