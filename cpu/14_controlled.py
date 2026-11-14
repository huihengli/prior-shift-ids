"""[CPU · 中等，建议实验机] 受控实验：把样本充足的攻击类人为下采样成“稀有类”，观察增强增益随样本数与重叠度的变化
动机：真实稀有类只有 13 个，重叠度与增益的相关不显著，无法区分“样本太少”和“类间重叠”各自的作用。
做法：对每个数据集中训练样本 ≥ --min-n 的非良性、非稀有类 c，依次只保留 n ∈ {10, 30, 100, 300} 条（按种子抽样，
其余类不动，测试集不动——c 的测试样本很多，F1 估计稳定），然后比较：
  none   不增强
  cw     类别权重（c 的权重 = T / n）
  ros    随机过采样到 T 条
  smote  SMOTE 到 T 条（与 02 相同：7 个原始特征的 log1p 空间内插值，派生特征重算）
  ps     先验平移（不重新训练：none 模型中 c 的概率乘以 T / n）
同时记录下采样后 c 的 kNN 同类纯度（k=10，留一）和不下采样时的 F1（上限参照）。T 默认 1000。
两种评估（2026-11-11 修订）：
  unmatched  测试集保持原样。c 在训练集被下采样、在测试集仍很常见 → 训练/测试先验不一致，
             任何把预测推向 c 的方法都会占便宜（对应“部署时攻击比例高于训练时”的场景）。
  matched    先验匹配：测试集中的 c 也按同样比例 n / N_train(c) 抽样（其余类不动），重复 --draws 次取平均。
             与真实稀有类的设定（分层划分，训练/测试先验一致）可比，作为主结果。
每个结果文件对应一个 (类, n, 检测器, 种子)，可断点续跑。
用法：python cpu/14_controlled.py [--classes "DoS Hulk" ...] [--detectors SAORF LightGBM] [--seeds 5] [--n-jobs 8]
输出：results/<ds>/controlled2/<class>__n<n>__<det>__s<seed>.json（旧版只有 unmatched，在 controlled/）
"""
import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (FEATURES, finalize_synth, load_cfg, load_split, make_detector, make_rf, rare_classes,  # noqa: E402
                    res_dir, save_json, xy)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


eval03, aug02 = _load("eval03", "03_eval_augmentation.py"), _load("aug02", "02_classic_augment.py")
K = 10


def purity(tr, c):
    from sklearn.neighbors import NearestNeighbors
    Z = np.log1p(tr[FEATURES].clip(lower=0).to_numpy(float))
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    lab = tr.label.to_numpy()
    idx = np.where(lab == c)[0]
    _, nb = NearestNeighbors(n_neighbors=K + 1).fit(Z).kneighbors(Z[idx])
    return float((lab[nb[:, 1:]] == c).mean())


def score(yte, P, cls, c, w=1.0, ratio=None, draws=200, rng_seed=0):
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support
    j = cls.index(c)
    Q = P.copy()
    Q[:, j] *= w
    pred = np.asarray(cls)[Q.argmax(1)]
    pos = yte == c
    p, r, f, _ = precision_recall_fscore_support(pos, pred == c, average="binary", zero_division=0)
    ben = yte == "Benign"
    out = {"f1": float(f), "precision": float(p), "recall": float(r),
           "ap": float(average_precision_score(pos, P[:, j])),
           "fp_benign_per_10k": float((pred[ben] == c).mean() * 1e4)}
    if ratio is not None:  # 先验匹配：测试集中 c 按 ratio 抽样，其余类全部保留
        rng = np.random.default_rng(rng_seed)
        ipos = np.where(pos)[0]
        k = max(1, int(round(len(ipos) * ratio)))
        fp = int(((pred == c) & ~pos).sum())
        hit = (pred[ipos] == c)
        tp = np.array([hit[rng.choice(len(ipos), k, replace=False)].sum() for _ in range(draws)])
        f1 = 2 * tp / np.maximum(2 * tp + fp + (k - tp), 1)
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        neg = np.where(~pos)[0]
        aps = []
        for _ in range(min(draws, 20)):
            sel = np.concatenate([rng.choice(ipos, k, replace=False), neg])
            aps.append(average_precision_score(pos[sel], P[sel, j]))
        out["matched"] = {"k_test": k, "f1": float(f1.mean()), "f1_sd": float(f1.std()), "precision": float(prec.mean()),
                          "recall": float((tp / k).mean()), "ap": float(np.mean(aps)), "fp_other": fp}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--ns", type=int, nargs="+", default=[10, 30, 100, 300])
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--min-n", type=int, default=1000)
    ap.add_argument("--detectors", nargs="+", default=["SAORF", "LightGBM"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--draws", type=int, default=200, help="先验匹配评估的抽样次数")
    ap.add_argument("--list", action="store_true", help="只列出可用的类")
    a = ap.parse_args()
    cfg = load_cfg()
    train, test = load_split(cfg, "train"), load_split(cfg, "test")
    rare = rare_classes(cfg, train)
    vc = train.label.value_counts()
    classes = a.classes or [c for c in vc.index if c != "Benign" and c not in rare and vc[c] >= a.min_n]
    if a.list:
        print("\n".join(classes))
        return
    Xte, yte = xy(test)
    nj = a.n_jobs if a.n_jobs is not None else cfg["experiment"]["n_jobs"]
    p_sa = eval03.sa_params(cfg)
    out = res_dir(cfg, "controlled2")
    T = a.target

    def fit(det, seed, tr, sw=None):
        m = make_rf(p_sa, seed, nj) if det == "SAORF" else make_detector({"LightGBM": "lgbm", "XGBoost": "xgb"}[det], seed, n_jobs=nj)
        X, y = xy(tr)
        m.fit(X, y, **({"sample_weight": sw} if sw is not None else {}))
        return np.asarray(m.predict_proba(Xte)), [str(k) for k in m.classes_]

    for c in classes:
        tagc = re.sub(r"[^A-Za-z0-9]+", "-", c)
        rest, own = train[train.label != c], train[train.label == c]
        for det in a.detectors:
            for seed in range(a.seeds):
                ref_f = out / f"{tagc}__full__{det}__s{seed}.json"
                if not ref_f.exists():  # 不下采样的参照
                    P, cls = fit(det, seed, train)
                    save_json({"class": c, "n": int(vc[c]), "detector": det, "seed": seed, "none": score(yte, P, cls, c, ratio=1.0, draws=1)}, ref_f)
                for n in a.ns:
                    f = out / f"{tagc}__n{n}__{det}__s{seed}.json"
                    if f.exists():
                        continue
                    tr = pd.concat([rest, own.sample(n, random_state=seed)], ignore_index=True)
                    r = {"class": c, "n": n, "target": T, "detector": det, "seed": seed, "purity": purity(tr, c)}
                    kw = dict(ratio=n / vc[c], draws=a.draws, rng_seed=seed)
                    P, cls = fit(det, seed, tr)
                    r["none"] = score(yte, P, cls, c, **kw)
                    r["ps"] = score(yte, P, cls, c, w=T / n, **kw)
                    P, cls = fit(det, seed, tr, sw=np.where(tr.label.to_numpy() == c, T / n, 1.0))
                    r["cw"] = score(yte, P, cls, c, **kw)
                    for meth in ("ros", "smote"):
                        syn = finalize_synth(aug02.smote_family(tr, [c], T - n, meth, seed), meth)
                        P, cls = fit(det, seed, pd.concat([tr, syn[FEATURES + ["label"]]], ignore_index=True))
                        r[meth] = score(yte, P, cls, c, **kw)
                    save_json(r, f)
                    print(f"{c:18s} n={n:<4d} {det:8s} s={seed} purity={r['purity']:.2f} F1 none={r['none']['f1']:.3f} "
                          f"ps={r['ps']['f1']:.3f} cw={r['cw']['f1']:.3f} ros={r['ros']['f1']:.3f} smote={r['smote']['f1']:.3f} ｜ 先验匹配 F1 "
                          + " ".join(f"{m}={r[m]['matched']['f1']:.3f}" for m in ("none", "ps", "cw", "ros", "smote")), flush=True)


if __name__ == "__main__":
    main()
