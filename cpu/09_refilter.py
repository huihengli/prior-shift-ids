"""[CPU] 离线重新筛选 g3 输出（不需要 GPU）：改变类间重叠约束的阈值，重算 accept
动机（NF v2.1 评估）：NF 真实稀有类自身就大面积重叠（Backdoor 真实样本同类近邻占比均值 0.13），
统一 tau=0.5 只保留比真实数据还“纯”的合成样本，导致 Backdoor 仅 187 条、测试样本所在的重叠区没有覆盖。
模式：
  rel   每类阈值 tau_c = min(cap, 该类真实样本（留一）纯度的分位数 q)，默认 q=0.5、cap=0.5：
        “至少和典型真实样本一样纯”，但不超过多数规则 0.5（可分的类仍用 0.5，重叠的类放宽）
  none  不做重叠约束（注意：g3 已对越界样本 R_hard=0，这里无法还原，结果偏保守）
其余条件与 g3 相同：有效、未记忆、去重、reward >= 阈值，最后按锚点均衡（每个真实锚点最多 cap 条）。
用法：python cpu/09_refilter.py --source llm_sft_filt --mode rel [--q 0.5]
输出：data/<ds>/synth/<source 去掉 _filt>_<mode>ov_filt.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEATURES, PRIMARY, exp_param, load_cfg, load_split, rare_classes, synth_dir  # noqa: E402


def space(df, mu=None, sd=None):
    z = np.log1p(df[FEATURES].clip(lower=0).to_numpy(float))
    if mu is None:
        mu, sd = z.mean(0), z.std(0) + 1e-9
    return (z - mu) / sd, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--mode", choices=["rel", "none"], default="rel")
    ap.add_argument("--q", type=float, default=0.5)
    ap.add_argument("--cap", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    cfg = load_cfg()
    rc = cfg["llm"]["reward"]
    tr = load_split(cfg, "train")
    rare = rare_classes(cfg, tr)
    tmax = int(exp_param(cfg, "aug_target_per_rare_class"))
    Z, mu, sd = space(tr)
    lab = tr.label.to_numpy()
    nn = NearestNeighbors(n_neighbors=a.k + 1).fit(Z)

    p = pd.read_csv(synth_dir(cfg) / f"{a.source}.csv")
    if "ov_purity" not in p:  # v2.1 之前的 g3 输出没有该列：在同一空间现算
        p["ov_purity"] = np.nan
        ok = p.label.isin(rare) & (p["raw_valid"] == 1)
        _, nb = nn.kneighbors(space(p.loc[ok], mu, sd)[0], n_neighbors=a.k)
        p.loc[ok, "ov_purity"] = (lab[nb] == p.loc[ok, "label"].to_numpy()[:, None]).mean(1)
    dup = p.duplicated(subset=PRIMARY + ["label"]).to_numpy()
    base = (p["raw_valid"] == 1) & (p["r_mem"] <= 0) & ~dup & (p["reward"] >= rc["accept_threshold"])
    acc = pd.Series(False, index=p.index)
    for c in rare:
        idx = np.where(lab == c)[0]
        _, nb = nn.kneighbors(Z[idx])
        real_pur = (lab[nb[:, 1:]] == c).mean(1)
        tau = min(a.cap, float(np.quantile(real_pur, a.q))) if a.mode == "rel" else -1.0
        m = base & (p.label == c) & (p["ov_purity"] >= tau)
        cap = int(np.ceil(max(1, tmax - len(idx)) / len(idx)))
        cand = p[m].sort_values("reward", ascending=False)
        keep = cand.groupby("anchor_nn").head(cap).index
        acc.loc[keep] = True
        print(f"{c:12s} real_purity_median={np.median(real_pur):.2f} tau={tau:.2f} 通过={int(m.sum()):5d} "
              f"均衡后={len(keep):5d}（原 g3 接受 {int(p.loc[p.label == c, 'accept'].sum())}）需要={max(0, tmax - len(idx))}")
    p["accept"] = acc.astype(int)
    out = synth_dir(cfg) / f"{a.source.replace('_filt', '')}_{a.mode}ov_filt.csv"
    p.to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
