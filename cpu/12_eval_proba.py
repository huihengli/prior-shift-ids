"""[CPU · 中等] 增强评估（概率版）：在 03 的协议上补充阈值无关指标、运营指标和“只改决策规则”的基线
与 03 完全相同的训练协议（同一 build_aug、同一检测器与超参数），每次训练后用 predict_proba 额外计算：
  ap                 各稀有类的 PR-AUC（一对多 average precision，与决策阈值无关）及其均值 rare_macro_ap
  fp_benign_per_10k  每 1 万条良性测试流中被判为任一稀有类的条数（运营视角的误报）
对“不增强”（none）额外给出两个不重新训练、只改决策规则的基线，用来检验“增强的增益是否等同于抬高稀有类先验”：
  ps    先验平移：稀有类 c 的预测概率乘以 w_c = target / n_c（与 ROS / 类别权重补足到 target 的先验比相同），再取最大
  cvps  同上，但 w_c = (target / n_c)^alpha，alpha 在训练集 3 折袋外概率上按稀有类 macro-F1 选取（不看测试集）
输出：results/<ds>/augment_proba/<src>__<det>__t<tgt>__s<seed>.json（可断点续跑）
用法：python cpu/12_eval_proba.py [--sources none cw ros ...] [--detectors SAORF LightGBM XGBoost RF-grid]
                                  [--seeds 5] [--n-jobs 8] [--targets 500]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (Timer, evaluate, exp_param, load_cfg, load_split, make_detector, make_rf, rare_classes,  # noqa: E402
                    res_dir, save_json, synth_dir, xy)

_spec = importlib.util.spec_from_file_location("eval03", Path(__file__).with_name("03_eval_augmentation.py"))
eval03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval03)

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def new_model(det, seed, nj, p_sa, p_rf):
    if det == "SAORF":
        return make_rf(p_sa, seed, nj)
    if det in p_rf:
        return make_rf(p_rf[det], seed, nj)
    return make_detector({"LightGBM": "lgbm", "XGBoost": "xgb", "RF": "rf"}[det], seed, n_jobs=nj)


def proba(m, X, labels):
    """按 labels 的顺序返回概率矩阵（模型没见过的类填 0）"""
    P = np.asarray(m.predict_proba(X))
    out = np.zeros((len(X), len(labels)))
    pos = {c: i for i, c in enumerate(labels)}
    for j, c in enumerate(m.classes_):
        out[:, pos[str(c)]] = P[:, j]
    return out


def metrics(yte, P, labels, rare, w=None):
    Q = P if w is None else P * w[None, :]
    pred = np.asarray(labels)[Q.argmax(1)]
    r = evaluate(yte, pred, labels, rare)
    ben = yte == "Benign"
    r["fp_benign_per_10k"] = float(np.isin(pred[ben], rare).mean() * 1e4) if ben.any() else float("nan")
    r["rare_macro_precision"] = float(np.mean([r["per_class"][c]["precision"] for c in rare]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--detectors", nargs="+", default=["SAORF", "LightGBM"])
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--targets", type=int, nargs="+", default=None)
    ap.add_argument("--no-cv", action="store_true", help="不做 cvps（省 3 次训练）")
    a = ap.parse_args()
    from sklearn.metrics import average_precision_score, f1_score
    from sklearn.model_selection import StratifiedKFold

    cfg = load_cfg()
    train, test = load_split(cfg, "train"), load_split(cfg, "test")
    rare = rare_classes(cfg, train)
    Xte, yte = xy(test)
    labels = sorted(train.label.unique())
    ridx = [labels.index(c) for c in rare]
    seeds = cfg["experiment"]["seeds"][: a.seeds] if a.seeds else cfg["experiment"]["seeds"]
    sources = a.sources or ["none", "cw"] + sorted(p.stem for p in synth_dir(cfg).glob("*.csv"))
    targets = a.targets or [exp_param(cfg, "aug_target_per_rare_class")]
    nj = a.n_jobs if a.n_jobs is not None else cfg["experiment"]["n_jobs"]
    p_sa = eval03.sa_params(cfg)
    p_rf = {d: eval03.sa_params(cfg, d.split("-", 1)[1]) for d in a.detectors if d.startswith("RF-")}
    out = res_dir(cfg, "augment_proba")
    vc0 = train.label.value_counts()

    for src in sources:
        pool = None if src in ("none", "cw") else pd.read_csv(synth_dir(cfg) / f"{src}.csv")
        for tgt in targets:
            w_ps = np.array([max(tgt, vc0[c]) / vc0[c] if c in rare else 1.0 for c in labels])
            for det in a.detectors:
                for seed in seeds:
                    f = out / f"{src}__{det}__t{tgt}__s{seed}.json"
                    if f.exists():
                        continue
                    tr = eval03.build_aug(train, pool, rare, tgt, seed)
                    Xtr, ytr = xy(tr)
                    fit_kw = {}
                    if src == "cw":
                        fit_kw["sample_weight"] = tr.label.map(dict(zip(labels, w_ps))).to_numpy(float)
                    m = new_model(det, seed, nj, p_sa, p_rf)
                    with Timer() as t:
                        m.fit(Xtr, ytr, **fit_kw)
                    P = proba(m, Xte, labels)
                    r = metrics(yte, P, labels, rare)
                    r["ap"] = {c: float(average_precision_score(yte == c, P[:, i])) for c, i in zip(rare, ridx)}
                    r["rare_macro_ap"] = float(np.mean(list(r["ap"].values())))
                    r.update(source=src, detector=det, target=tgt, seed=seed, train_rows=len(tr),
                             n_synth=len(tr) - len(train), train_time_s=t.s)
                    if src == "none":
                        r["ps"] = metrics(yte, P, labels, rare, w_ps)
                        if not a.no_cv:  # alpha 只用训练集袋外概率选择
                            oof = np.zeros((len(Xtr), len(labels)))
                            for k, (i_tr, i_va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr)):
                                mk = new_model(det, seed, nj, p_sa, p_rf).fit(Xtr[i_tr], ytr[i_tr])
                                oof[i_va] = proba(mk, Xtr[i_va], labels)
                            sc = []
                            for al in ALPHAS:
                                pr = np.asarray(labels)[(oof * (w_ps ** al)[None, :]).argmax(1)]
                                sc.append(float(np.mean(f1_score(ytr, pr, labels=rare, average=None, zero_division=0))))
                            al = ALPHAS[int(np.argmax(sc))]
                            r["cvps"] = metrics(yte, P, labels, rare, w_ps ** al)
                            r["cvps"]["alpha"] = al
                            r["cvps"]["oof_rare_f1_by_alpha"] = dict(zip(map(str, ALPHAS), sc))
                    save_json(r, f)
                    extra = (f" ps={r['ps']['rare_macro_f1']:.4f}" + (f" cvps={r['cvps']['rare_macro_f1']:.4f}(a={r['cvps']['alpha']})" if "cvps" in r else "")) if src == "none" else ""
                    print(f"{src:22s} {det:8s} t={tgt:<5d} s={seed} rareF1={r['rare_macro_f1']:.4f} rareAP={r['rare_macro_ap']:.4f} "
                          f"FP/10k={r['fp_benign_per_10k']:.1f}{extra} ({t.s:.0f}s)", flush=True)
    if not (out / "meta.json").exists():
        save_json({"labels": labels, "rare": rare}, out / "meta.json")


if __name__ == "__main__":
    main()
