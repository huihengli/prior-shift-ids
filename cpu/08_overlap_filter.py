"""[CPU] 类间重叠过滤（离线后处理，不需要 GPU）
动机（v2 评估发现）：合成的稀有类样本若落在其他类（Benign / Infiltration 等）的真实样本区域，
检测器会把这些区域判为稀有类 → 大量误报、精确率崩溃。原奖励的保真项只和同类真实样本比，
困难项还会鼓励这种越界样本。
做法：在 log1p + 标准化空间中，对每条已接受的合成样本取全部真实训练样本中最近的 k 个，
同类占比 < tau 的样本判为越界，置 accept=False。
用法：python cpu/08_overlap_filter.py --sources v2_llm_sft_filt [--k 10 --tau 0.5]
输出：data/<ds>/synth/<source>_pure.csv（与原文件同格式，额外列 purity），统计打印到屏幕
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEATURES, load_cfg, load_split, rare_classes, synth_dir  # noqa: E402


def space(df, mu=None, sd=None):
    z = np.log1p(df[FEATURES].clip(lower=0).to_numpy(float))
    if mu is None:
        mu, sd = z.mean(0), z.std(0) + 1e-9
    return (z - mu) / sd, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.5)
    a = ap.parse_args()
    cfg = load_cfg()
    tr = load_split(cfg, "train")
    rare = rare_classes(cfg, tr)
    Z, mu, sd = space(tr)
    lab = tr.label.to_numpy()
    nn = NearestNeighbors(n_neighbors=a.k + 1).fit(Z)

    # 参考：真实稀有样本自身的纯度（留一，去掉自身）
    for c in rare:
        idx = np.where(lab == c)[0]
        _, nb = nn.kneighbors(Z[idx])
        pur = (lab[nb[:, 1:]] == c).mean(1)
        print(f"[real] {c:15s} n={len(idx):4d} 纯度均值={pur.mean():.2f} 低于 tau 的比例={(pur < a.tau).mean():.2f}")

    for src in a.sources:
        p = pd.read_csv(synth_dir(cfg) / f"{src}.csv")
        acc = p["accept"].astype(bool) if "accept" in p else pd.Series(True, index=p.index)
        p["purity"] = np.nan
        sel = acc & p.label.isin(rare)
        Zs, _, _ = space(p.loc[sel], mu, sd)
        _, nb = nn.kneighbors(Zs, n_neighbors=a.k)
        pur = (lab[nb] == p.loc[sel, "label"].to_numpy()[:, None]).mean(1)
        p.loc[sel, "purity"] = pur
        keep = acc & ~(sel & (p["purity"] < a.tau))
        print(f"\n[{src}] 接受 {int(acc.sum())} → 过滤后 {int(keep.sum())}")
        for c in rare:
            m = sel & (p.label == c)
            nbl = pd.Series(lab[nb[(p.loc[sel, 'label'] == c).to_numpy()]].ravel()).value_counts(normalize=True)
            print(f"  {c:15s} {int(m.sum()):5d} → {int((keep & (p.label == c)).sum()):5d}  "
                  f"近邻类别构成: {nbl.round(2).head(4).to_dict()}")
        p["accept"] = keep
        out = synth_dir(cfg) / f"{src}_pure.csv"
        p.to_csv(out, index=False)
        print("  saved", out)


if __name__ == "__main__":
    main()
