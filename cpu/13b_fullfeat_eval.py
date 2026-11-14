"""[CPU] 完整特征上的重叠度与基线（输入：13_fullfeat_prep.py 生成的 data/<ds>/fullfeat/full.csv.gz）
同一批样本、同一划分，在两个特征空间上比较：
  d10   10 维紧凑特征（与主实验相同）
  full  10 维 + 原始文件中其余全部数值流特征（x_*，去掉常数列）
--part purity（轻量，本机即可）：各稀有类真实训练样本的 kNN 同类纯度（留一，k=10，符号 log1p + 标准化空间）
--part eval  （中等，建议实验机）：不增强 / 类别权重 / ROS / SMOTE × 检测器 × 种子，记录各稀有类 F1、PR-AUC
              SMOTE 在对应特征空间的符号 log1p 空间内插值（full 空间不重算派生特征，因为 CICFlowMeter 特征间的约束无法逐一还原）
LLM 合成样本只有 10 维，不参与 full 空间的比较（论文中作为局限性说明）。
用法：python cpu/13b_fullfeat_eval.py --part purity
      python cpu/13b_fullfeat_eval.py --part eval [--detectors SAORF LightGBM XGBoost] [--seeds 5] [--n-jobs 8] [--spaces d10 full]
输出：results/<ds>/fullfeat/purity.csv、<space>__<src>__<det>__s<seed>.json
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (FEATURES, Timer, ds_dir, evaluate, exp_param, load_cfg, make_detector, make_rf, rare_threshold,  # noqa: E402
                    res_dir, save_json)

_spec = importlib.util.spec_from_file_location("eval03", Path(__file__).with_name("03_eval_augmentation.py"))
eval03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval03)
K = 10


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def load(cfg):
    d = pd.read_csv(ds_dir(cfg) / "fullfeat" / "full.csv.gz", low_memory=False)
    xc = [c for c in d.columns if c.startswith("x_")]
    d[xc] = d[xc].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    tr, te = d[d.split == "train"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
    xc = [c for c in xc if tr[c].nunique() > 1]
    return tr, te, {"d10": FEATURES, "full": FEATURES + xc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["purity", "eval"], required=True)
    ap.add_argument("--detectors", nargs="+", default=["SAORF", "LightGBM", "XGBoost"])
    ap.add_argument("--sources", nargs="+", default=["none", "cw", "ros", "smote"])
    ap.add_argument("--spaces", nargs="+", default=["d10", "full"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=None)
    a = ap.parse_args()
    cfg = load_cfg()
    tr, te, spaces = load(cfg)
    vc = tr.label.value_counts()
    rare = sorted(vc[vc < rare_threshold(cfg)].index.tolist())
    labels = sorted(tr.label.unique())
    out = res_dir(cfg, "fullfeat")
    lab = tr.label.to_numpy()

    if a.part == "purity":
        from sklearn.neighbors import NearestNeighbors
        rows = []
        for sp, cols in spaces.items():
            Z = slog(tr[cols].to_numpy(float))
            Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
            nn = NearestNeighbors(n_neighbors=K + 1).fit(Z)
            for c in rare:
                idx = np.where(lab == c)[0]
                _, nb = nn.kneighbors(Z[idx])
                pur = (lab[nb[:, 1:]] == c).mean(1)
                top = pd.Series(lab[nb[:, 1:]].ravel()).value_counts(normalize=True)
                top = top[top.index != c].head(2)
                rows.append({"dataset": cfg["_ds"], "space": sp, "n_features": len(cols), "class": c, "n_train": len(idx),
                             "purity_mean": pur.mean(), "purity_median": np.median(pur), "frac_below_0.5": (pur < 0.5).mean(),
                             "top_other_neighbours": "; ".join(f"{k} {v:.2f}" for k, v in top.items())})
        t = pd.DataFrame(rows)
        t.to_csv(out / "purity.csv", index=False)
        print(t.round(3).to_string(index=False))
        return

    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import average_precision_score
    tgt = int(exp_param(cfg, "aug_target_per_rare_class"))
    nj = a.n_jobs if a.n_jobs is not None else cfg["experiment"]["n_jobs"]
    p_sa = eval03.sa_params(cfg)
    yte = te.label.to_numpy()
    w = {c: (max(tgt, vc[c]) / vc[c] if c in rare else 1.0) for c in labels}
    for sp in a.spaces:
        cols = spaces[sp]
        Xtr0, Xte = tr[cols].to_numpy(float), te[cols].to_numpy(float)
        for src in a.sources:
            for det in a.detectors:
                for seed in range(a.seeds):
                    f = out / f"{sp}__{src}__{det}__s{seed}.json"
                    if f.exists():
                        continue
                    Xtr, ytr, kw = Xtr0, lab, {}
                    if src == "cw":
                        kw["sample_weight"] = np.array([w[c] for c in lab])
                    elif src == "ros":
                        rng = np.random.default_rng(seed)
                        add = np.concatenate([rng.choice(np.where(lab == c)[0], max(0, tgt - vc[c])) for c in rare]).astype(int)
                        Xtr, ytr = np.vstack([Xtr0, Xtr0[add]]), np.concatenate([lab, lab[add]])
                    elif src == "smote":
                        strat = {c: int(max(tgt, vc[c])) for c in rare}
                        k = max(1, min(5, int(min(vc[c] for c in rare)) - 1))
                        Zr, ytr = SMOTE(sampling_strategy=strat, k_neighbors=k, random_state=seed).fit_resample(slog(Xtr0), lab)
                        Xtr = np.sign(Zr) * np.expm1(np.abs(Zr))
                    m = make_rf(p_sa, seed, nj) if det == "SAORF" else make_detector({"LightGBM": "lgbm", "XGBoost": "xgb"}[det], seed, n_jobs=nj)
                    with Timer() as t:
                        m.fit(Xtr, ytr, **kw)
                    P = np.asarray(m.predict_proba(Xte))
                    cls = [str(c) for c in m.classes_]
                    r = evaluate(yte, np.asarray(cls)[P.argmax(1)], labels, rare)
                    r["ap"] = {c: float(average_precision_score(yte == c, P[:, cls.index(c)])) for c in rare}
                    r["rare_macro_ap"] = float(np.mean(list(r["ap"].values())))
                    r.update(space=sp, n_features=len(cols), source=src, detector=det, target=tgt, seed=seed, train_time_s=t.s)
                    save_json(r, f)
                    print(f"{sp:5s} {src:6s} {det:8s} s={seed} rareF1={r['rare_macro_f1']:.4f} rareAP={r['rare_macro_ap']:.4f} ({t.s:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
