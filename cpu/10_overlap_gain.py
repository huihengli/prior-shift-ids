"""[CPU · 轻量] 类间重叠度 vs 增强增益（论文核心分析，两个数据集合并）
对每个数据集的每个稀有类：
  - 重叠度：真实训练样本在 10 维 log1p 标准化空间中的 kNN 同类纯度（留一，k=10），均值与 < 0.5 的比例
  - 增益：各增强方法相对不增强的该类 F1 变化（10 种子均值，按种子配对 Wilcoxon 未校正 p）
  - LLM 合成池：实际用到的条数、合成样本的平均纯度
然后在全部稀有类（CSE 4 + NF 4 + CIC17 5 = 13 个点）上算 Spearman 相关（纯度 vs 增益），并画散点图。
方法名映射见 METHODS（不同数据集的 LLM 来源文件名不同）。
用法：python cpu/10_overlap_gain.py
输出：results/overlap_gain/{per_class.csv, correlation.csv, summary.md, overlap_gain.png}
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from common import FEATURES, exp_param, load_cfg, load_split, rare_classes, synth_dir  # noqa: E402

K = 10
DATASETS = {"cse2018": "CSE-CIC-IDS2018", "nf_unsw_v3": "NF-UNSW-NB15-v3", "cic2017": "CIC-IDS2017"}
# 统一方法名 → 各数据集的来源文件名
METHODS = {
    "Class weight": {"cse2018": "cw", "nf_unsw_v3": "cw", "cic2017": "cw"},
    "ROS": {"cse2018": "ros", "nf_unsw_v3": "ros", "cic2017": "ros"},
    "SMOTE": {"cse2018": "smote", "nf_unsw_v3": "smote", "cic2017": "smote"},
    "B-SMOTE": {"cse2018": "bsmote", "nf_unsw_v3": "bsmote", "cic2017": "bsmote"},
    "ADASYN": {"cse2018": "adasyn", "nf_unsw_v3": "adasyn", "cic2017": "adasyn"},
    "LLM-SFT (no overlap)": {"cse2018": "v2_llm_sft_filt", "nf_unsw_v3": "llm_sft_noneov_filt", "cic2017": "llm_sft_noneov_filt"},
    "LLM-SFT (tau=0.5)": {"cse2018": "v2_llm_sft_filt_pure", "nf_unsw_v3": "llm_sft_filt", "cic2017": "llm_sft_filt"},
    "LLM-SFT (relative tau)": {"cse2018": "v2_llm_sft_relov_filt", "nf_unsw_v3": "llm_sft_relov_filt", "cic2017": "llm_sft_relov_filt"},
}
DETS = ["SAORF", "LightGBM", "XGBoost", "RF-grid"]


def space(df, mu=None, sd=None):
    z = np.log1p(df[FEATURES].clip(lower=0).to_numpy(float))
    if mu is None:
        mu, sd = z.mean(0), z.std(0) + 1e-9
    return (z - mu) / sd, mu, sd


def load_runs(cfg, src, det, tgt):
    d = common.RESULTS / cfg["_ds"] / "augment"
    out = {}
    for f in d.glob(f"{src}__{det}__t{tgt}__s*.json"):
        j = json.load(open(f, encoding="utf-8"))
        out[int(j["seed"])] = {c: v["f1"] for c, v in j["per_class"].items()}
    return out


def main():
    rows = []
    for ds, ds_name in DATASETS.items():
        os.environ["IDS_DATASET"] = ds
        cfg = load_cfg()
        tr = load_split(cfg, "train")
        te = load_split(cfg, "test")
        rare = rare_classes(cfg, tr)
        tgt = int(exp_param(cfg, "aug_target_per_rare_class"))
        Z, mu, sd = space(tr)
        lab = tr.label.to_numpy()
        nn = NearestNeighbors(n_neighbors=K + 1).fit(Z)
        base = {det: load_runs(cfg, "none", det, tgt) for det in DETS}
        for c in rare:
            idx = np.where(lab == c)[0]
            _, nb = nn.kneighbors(Z[idx])
            pur = (lab[nb[:, 1:]] == c).mean(1)
            r = {"dataset": ds_name, "class": c, "n_train": len(idx), "n_test": int((te.label == c).sum()),
                 "real_purity": pur.mean(), "real_frac_below_0.5": (pur < 0.5).mean()}
            for det in DETS:
                b = base[det]
                seeds = sorted(b)
                r[f"F1 none [{det}]"] = np.mean([b[s][c] for s in seeds])
                for m, srcs in METHODS.items():
                    runs = load_runs(cfg, srcs[ds], det, tgt)
                    ss = [s for s in seeds if s in runs]
                    if not ss:
                        continue
                    x = np.array([runs[s][c] for s in ss])
                    y = np.array([b[s][c] for s in ss])
                    r[f"dF1 {m} [{det}]"] = (x - y).mean()
                    r[f"p {m} [{det}]"] = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
            for m, srcs in METHODS.items():
                if not m.startswith("LLM"):
                    continue
                f = synth_dir(cfg) / f"{srcs[ds]}.csv"
                if not f.exists():
                    continue
                p = pd.read_csv(f)
                p = p[p["accept"].astype(bool) & (p.label == c)]
                r[f"pool {m}"] = len(p)
                if len(p):
                    _, nbs = nn.kneighbors(space(p, mu, sd)[0], n_neighbors=K)
                    r[f"synth_purity {m}"] = (lab[nbs] == c).mean()
            rows.append(r)
            print(f"[{ds}] {c:15s} purity={r['real_purity']:.2f}")
    t = pd.DataFrame(rows)
    out = common.RESULTS / "overlap_gain"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "per_class.csv", index=False)

    cor = []
    for m in METHODS:
        for det in DETS:
            col = f"dF1 {m} [{det}]"
            if col not in t or t[col].isna().any():
                continue
            rho, p = spearmanr(t["real_purity"], t[col])
            cor.append({"method": m, "detector": det, "spearman_rho": rho, "p": p, "n_classes": len(t),
                        "mean_dF1": t[col].mean(),
                        "mean_dF1_purity>=0.5": t.loc[t.real_purity >= 0.5, col].mean(),
                        "mean_dF1_purity<0.5": t.loc[t.real_purity < 0.5, col].mean()})
    cor = pd.DataFrame(cor)
    cor.to_csv(out / "correlation.csv", index=False)

    # LLM 相对 ROS 的差距与纯度的关系（同一检测器）
    for det in DETS:
        a, b = f"dF1 LLM-SFT (relative tau) [{det}]", f"dF1 ROS [{det}]"
        if a in t and b in t:
            t[f"LLM_rel - ROS [{det}]"] = t[a] - t[b]

    show = ["dataset", "class", "n_train", "n_test", "real_purity", "real_frac_below_0.5", "F1 none [SAORF]"] + \
           [f"dF1 {m} [SAORF]" for m in METHODS if f"dF1 {m} [SAORF]" in t] + \
           [c for c in t.columns if c.startswith("pool ") or c.startswith("synth_purity")]
    md = ["## 每个稀有类：重叠度与增益（SAORF，ΔF1 = 方法 − 不增强，10 种子均值）",
          t[show].round(3).to_markdown(index=False), "",
          *sum([[f"## {d}", t[["dataset", "class", "real_purity"] + [c for c in t.columns if f"[{d}]" in c and c.startswith("dF1")]]
                 .round(3).to_markdown(index=False), ""] for d in DETS[1:]], []),
          "## 纯度 vs 增益（全部稀有类，Spearman）", cor.round(3).to_markdown(index=False), ""]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.2), sharey=True)
    axes = axes.ravel()
    style = {"Class weight": ("#111827", "D"), "ROS": ("#6b7280", "s"), "SMOTE": ("#9ca3af", "^"),
             "LLM-SFT (tau=0.5)": ("#93c5fd", "o"), "LLM-SFT (relative tau)": ("#2563eb", "o")}
    # 两个数据集的类别可能纯度相同（SQLi 与 Analysis 都是 0.32）：按数据集左右错开
    off = {"CSE-CIC-IDS2018": -0.012, "NF-UNSW-NB15-v3": 0.0, "CIC-IDS2017": 0.012}
    x = t["real_purity"] + t["dataset"].map(off)
    short = t["dataset"].map({"CSE-CIC-IDS2018": "CSE", "NF-UNSW-NB15-v3": "NF", "CIC-IDS2017": "C17"}) + ":" + \
        t["class"].str.replace("Web-", "").str.replace("BruteForce", "BF")
    for ax, det in zip(axes, DETS):
        for m, (col, mk) in style.items():
            k = f"dF1 {m} [{det}]"
            if k not in t:
                continue
            ax.scatter(x, t[k], c=col, marker=mk, label=m, s=36, zorder=3, edgecolors="white", linewidths=0.5)
        top = max(t[[f"dF1 {m} [{det}]" for m in style if f"dF1 {m} [{det}]" in t]].max().max(), 0.1)
        for xi, name in zip(x, short):
            ax.text(xi, top + 0.015, name, fontsize=7, rotation=90, ha="center", va="bottom", color="#4b5563")
        ax.axhline(0, color="#111827", lw=0.8)
        ax.axvline(0.5, color="#d1d5db", lw=0.8, ls="--")
        ax.set_title(det, pad=48)
        ax.set_xlabel("real-sample kNN class purity (k=10)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ΔF1 vs. no augmentation")
    axes[2].set_ylabel("ΔF1 vs. no augmentation")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "overlap_gain.png", dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
