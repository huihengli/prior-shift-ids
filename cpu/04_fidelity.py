"""[CPU · 轻量] 合成样本质量评估（逐方法、逐稀有类）
指标：
  valid_raw      生成器原始输出的物理约束通过率（仅 LLM 方法有该列）
  W1             7 个原始特征（log1p 后标准化）的 Wasserstein 距离均值      ↓
  corr_diff      特征相关矩阵差异（Frobenius）                               ↓
  c2st_auc       分类器双样本检验：区分真/假的 AUC，越接近 0.5 越好          →0.5
  exact_copy     与训练集完全相同的比例（记忆/抄袭）                         ↓
  mem_rate       到最近训练样本距离 < 真实样本间最近距离 1% 分位 的比例       ↓
  diversity      合成样本两两距离均值 / 真实样本两两距离均值                 →1
  tstr_f1        TSTR：该稀有类在训练中只用合成样本代替，测试真实数据的 F1    ↑
用法：python cpu/04_fidelity.py [--sources ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (FEATURES, exp_param, PRIMARY, load_cfg, load_split, make_rf, rare_classes, res_dir, synth_dir, xy)

N_EVAL = 300  # 每类最多取多少条合成样本做评估（与真实数相近更公平）


def zlog(a, mu, sd):
    return (np.log1p(a.astype(float)) - mu) / sd


def nn_dist(A, B, same=False):
    from sklearn.neighbors import NearestNeighbors
    k = 2 if same else 1
    nn = NearestNeighbors(n_neighbors=min(k, len(B))).fit(B)
    d, _ = nn.kneighbors(A)
    return d[:, -1]


def c2st(R, S, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    n = min(len(R), len(S))
    if n < 6:
        return np.nan
    rng = np.random.default_rng(seed)
    X = np.vstack([R[rng.choice(len(R), n, False)], S[rng.choice(len(S), n, False)]])
    y = np.r_[np.zeros(n), np.ones(n)]
    cv = StratifiedKFold(min(5, n), shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(RandomForestClassifier(100, random_state=seed, n_jobs=-1),
                                         X, y, cv=cv, scoring="roc_auc")))


def mean_pdist(A, rng, m=300):
    if len(A) < 2:
        return np.nan
    A = A[rng.choice(len(A), min(m, len(A)), False)]
    d = np.sqrt(((A[:, None] - A[None]) ** 2).sum(-1))
    return d[np.triu_indices(len(A), 1)].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--out-sub", default="fidelity", help="结果子目录（默认 fidelity；第二个 LLM 用 fidelity_llama，避免覆盖已有汇总）")
    a = ap.parse_args()
    cfg = load_cfg()
    train, test = load_split(cfg, "train"), load_split(cfg, "test")
    rare = rare_classes(cfg, train)
    tgt = exp_param(cfg, "aug_target_per_rare_class")
    L = np.log1p(train[PRIMARY].to_numpy(float))
    mu, sd = L.mean(0), L.std(0) + 1e-9
    Xte, yte = xy(test)
    rng = np.random.default_rng(0)
    sources = a.sources or sorted(p.stem for p in synth_dir(cfg).glob("*.csv"))
    rows = []
    for src in sources:
        pool = pd.read_csv(synth_dir(cfg) / f"{src}.csv")
        if "accept" in pool:
            pool = pool[pool["accept"].astype(bool)]
        for c in rare:
            real = train[train.label == c]
            syn = pool[pool.label == c]
            if len(syn) == 0:
                rows.append({"source": src, "class": c, "n_synth": 0})
                continue
            syn_e = syn.sample(min(N_EVAL, len(syn)), random_state=0)
            R, S = zlog(real[PRIMARY].to_numpy(), mu, sd), zlog(syn_e[PRIMARY].to_numpy(), mu, sd)
            w1 = np.mean([wasserstein_distance(R[:, j], S[:, j]) for j in range(R.shape[1])])
            cr, cs = np.corrcoef(R.T), np.corrcoef(S.T)
            corr_diff = float(np.linalg.norm(np.nan_to_num(cr) - np.nan_to_num(cs)))
            Ztr = zlog(train[PRIMARY].to_numpy(), mu, sd)
            d_syn = nn_dist(S, Ztr)
            d_rr = nn_dist(R, R, same=True) if len(R) > 1 else np.array([0.0])
            thr = np.quantile(d_rr, 0.01)
            # TSTR：该类真实样本全部替换为 tgt 条合成样本
            tr = pd.concat([train[train.label != c],
                            syn.sample(min(tgt, len(syn)), random_state=0)[FEATURES + ["label"]]])
            m = make_rf(None, 0).fit(*xy(tr))
            from sklearn.metrics import f1_score
            pred = m.predict(Xte)
            rows.append({
                "source": src, "class": c, "n_real": len(real), "n_synth": len(syn),
                "valid_raw": float(syn["raw_valid"].mean()) if "raw_valid" in syn else np.nan,
                "W1": float(w1), "corr_diff": corr_diff, "c2st_auc": c2st(R, S),
                "exact_copy": float(np.mean(d_syn < 1e-9)), "mem_rate": float(np.mean(d_syn <= thr)),
                "diversity": float(mean_pdist(S, rng) / max(mean_pdist(R, rng), 1e-9)),
                "unique_rate": float(syn_e[PRIMARY].drop_duplicates().shape[0] / len(syn_e)),
                "tstr_f1": float(f1_score(yte == c, pred == c)),
            })
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    df = pd.DataFrame(rows)
    out = res_dir(cfg, a.out_sub)
    df.to_csv(out / "fidelity_per_class.csv", index=False)
    num = df.drop(columns=["class"]).groupby("source").mean(numeric_only=True)
    num.to_csv(out / "fidelity_summary.csv")
    print(num.round(3).to_string())


if __name__ == "__main__":
    main()
