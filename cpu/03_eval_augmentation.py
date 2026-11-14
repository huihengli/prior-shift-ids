"""[CPU · 中等] 数据增强效果评估（论文核心实验）
协议（所有增强方法完全一致）：
  训练集 = 真实训练集 + 每个稀有类从该方法的合成池中抽 (target - 真实数) 条（按种子抽样）
  测试集 = 固定的真实测试集（不含任何合成样本）
  检测器 = SAORF（01 中 SA 得到的最优参数）+ LightGBM；补充检测器 XGBoost、RF-grid / RF-optuna（01 中对应调参的最优参数）
  种子   = 10 个
来源：'none'（不增强）、'cw'（代价敏感：不加样本，稀有类样本权重 = target / 真实数，与 ROS 补足到 target 的有效权重相同）
      + data/<ds>/synth/*.csv 中的全部方法（传统方法与 GPU 端 LLM 方法输出）
可断点续跑。
用法：python cpu/03_eval_augmentation.py [--sweep] [--sources none smote llm_loop_r3 ...] [--detectors SAORF LightGBM XGBoost RF-grid] [--n-jobs 8]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (FEATURES, exp_param, Timer, evaluate, load_cfg, load_split, make_detector, make_rf, rare_classes,
                    res_dir, save_json, synth_dir, xy)


def sa_params(cfg, method="sa"):
    fs = sorted(res_dir(cfg, "detectors/tune").glob(f"{method}_s*.json"))
    if not fs:
        print(f"!! 未找到 {method} 调参结果，退化为默认 RF。请先运行 01_detectors.py --part tune")
        return None
    rs = [json.load(open(f, encoding="utf-8")) for f in fs]
    return max(rs, key=lambda r: r["best_cv"])["best_params"]


def build_aug(train, pool, rare, target, seed, accept_col="accept"):
    if pool is None:
        return train
    if accept_col in pool:  # LLM 方法：只用通过奖励筛选的样本（若有该列）
        pool = pool[pool[accept_col].astype(bool)]
    parts = [train]
    vc = train.label.value_counts()
    for c in rare:
        need = max(0, target - int(vc.get(c, 0)))
        cand = pool[pool.label == c]
        if need and len(cand):
            parts.append(cand.sample(min(need, len(cand)), random_state=seed)[FEATURES + ["label"]])
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--detectors", nargs="+", default=["SAORF", "LightGBM"])
    ap.add_argument("--sweep", action="store_true", help="增强规模消融（仅 SAORF）")
    ap.add_argument("--seeds", type=int, default=None, help="只跑前 n 个种子")
    ap.add_argument("--n-jobs", type=int, default=None, help="每个模型的线程数（默认 config experiment.n_jobs）")
    ap.add_argument("--targets", type=int, nargs="+", default=None, help="指定补足目标数（覆盖配置/sweep）")
    a = ap.parse_args()

    cfg = load_cfg()
    train, test = load_split(cfg, "train"), load_split(cfg, "test")
    rare = rare_classes(cfg, train)
    Xte, yte = xy(test)
    labels = sorted(train.label.unique())
    seeds = cfg["experiment"]["seeds"][: a.seeds] if a.seeds else cfg["experiment"]["seeds"]
    sources = a.sources or ["none"] + sorted(p.stem for p in synth_dir(cfg).glob("*.csv"))
    targets = exp_param(cfg, "aug_target_sweep") if a.sweep else [exp_param(cfg, "aug_target_per_rare_class")]
    if a.targets:
        targets = a.targets
    dets = ["SAORF"] if a.sweep else a.detectors
    p_sa = sa_params(cfg)
    nj = a.n_jobs if a.n_jobs is not None else cfg["experiment"]["n_jobs"]
    p_rf = {d: sa_params(cfg, d.split("-", 1)[1]) for d in dets if d.startswith("RF-")}
    out = res_dir(cfg, "augment_sweep" if a.sweep else "augment")
    pred_dir = out / "preds"
    pred_dir.mkdir(exist_ok=True)
    lab_idx = {l: i for i, l in enumerate(labels)}

    for src in sources:
        pool = None if src in ("none", "cw") else pd.read_csv(synth_dir(cfg) / f"{src}.csv")
        for tgt in targets:
            for det in dets:
                for seed in seeds:
                    tag = f"{src}__{det}__t{tgt}__s{seed}"
                    f = out / f"{tag}.json"
                    if f.exists():
                        continue
                    tr = build_aug(train, pool, rare, tgt, seed)
                    Xtr, ytr = xy(tr)
                    if det == "SAORF":
                        m = make_rf(p_sa, seed, nj)
                    elif det in p_rf:
                        m = make_rf(p_rf[det], seed, nj)
                    else:
                        m = make_detector({"LightGBM": "lgbm", "XGBoost": "xgb", "RF": "rf"}[det], seed, n_jobs=nj)
                    fit_kw = {}
                    if src == "cw":  # 代价敏感基线：只改权重，检验“增益是否只是抬高稀有类先验”
                        vc = tr.label.value_counts()
                        w = {c: (max(tgt, vc[c]) / vc[c] if c in rare else 1.0) for c in vc.index}
                        fit_kw["sample_weight"] = tr.label.map(w).to_numpy(float)
                    with Timer() as t:
                        m.fit(Xtr, ytr, **fit_kw)
                    pred = m.predict(Xte)
                    r = evaluate(yte, pred, labels, rare)
                    r.update(source=src, detector=det, target=tgt, seed=seed, train_rows=len(tr),
                             n_synth=len(tr) - len(train), train_time_s=t.s)
                    save_json(r, f)
                    np.save(pred_dir / f"{tag}.npy", np.array([lab_idx[p] for p in pred], dtype=np.int8))
                    print(f"{src:18s} {det:8s} t={tgt:<5d} s={seed} macroF1={r['macro_f1']:.4f} "
                          f"rareF1={r['rare_macro_f1']:.4f} ({t.s:.0f}s)", flush=True)
    if not (out / "y_test.npy").exists():
        np.save(out / "y_test.npy", np.array([lab_idx[y] for y in yte], dtype=np.int8))
    if not (out / "meta.json").exists():
        save_json({"labels": labels, "rare": rare}, out / "meta.json")


if __name__ == "__main__":
    main()
