"""[CPU · 中等] 检测器对比实验（论文第一部分：轻量检测器）
  A. 调参方法对比：SA(=SAORF) / Random / Grid / Optuna-TPE / GA，**相同评估预算**，多种子
  B. 最终检测器对比：RF-default、RF+各调参方法、XGBoost、LightGBM、MLP，10 个种子
     记录 macro-F1、稀有类 F1、训练时间、推理延迟、模型体积
可断点续跑：每个 (方法, 种子) 结果单独保存，已存在则跳过。
用法：python cpu/01_detectors.py [--part tune|final|all] [--quick]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (Timer, evaluate, load_cfg, load_split, make_detector, make_rf, rare_classes, res_dir,
                    save_json, stratified_cap, tune_rf, xy)

TUNERS = ["sa", "random", "grid", "optuna", "ga"]


def part_tune(cfg, quick):
    out = res_dir(cfg, "detectors/tune")
    train = load_split(cfg, "train")
    sa = cfg["detector"]["sa"]
    seeds = cfg["experiment"]["tune_seeds"][:1] if quick else cfg["experiment"]["tune_seeds"]
    budget = 10 if quick else sa["maxiter"]
    for seed in seeds:
        # 调参子样本：每类最多 n 条（稀有类全保留），保证 CV 中稀有类可见
        n_cls = train.label.nunique()
        sub = stratified_cap(train, max(sa["tune_subsample"] // n_cls, 50), seed)
        X, y = xy(sub)
        for m in TUNERS:
            f = out / f"{m}_s{seed}.json"
            if f.exists():
                continue
            print(f"[tune] {m} seed={seed} n={len(sub)} budget={budget}", flush=True)
            with Timer() as t:
                p, s, h = tune_rf(m, X, y, cfg, seed, budget)
            save_json({"method": m, "seed": seed, "best_params": p, "best_cv": s,
                       "history": h, "time_s": t.s, "budget": budget}, f)
            print(f"   best={s:.4f} {p} ({t.s:.0f}s)", flush=True)


def best_params(cfg, method):
    """各调参方法：取多个种子中 CV 最好的那组参数，作为该方法的最终配置"""
    fs = sorted(res_dir(cfg, "detectors/tune").glob(f"{method}_s*.json"))
    if not fs:
        return None
    import json
    rs = [json.load(open(f, encoding="utf-8")) for f in fs]
    return max(rs, key=lambda r: r["best_cv"])["best_params"]


def latency(model, X, reps=3):
    one = X[:1]
    t = time.perf_counter()
    for _ in range(50):
        model.predict(one)
    single_ms = (time.perf_counter() - t) / 50 * 1000
    batch = X[:10000]
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        model.predict(batch)
        ts.append(time.perf_counter() - t)
    return single_ms, float(np.median(ts)) / len(batch) * 1e6  # 单条 ms，批量 µs/flow


def part_final(cfg, quick):
    out = res_dir(cfg, "detectors/final")
    train, test = load_split(cfg, "train"), load_split(cfg, "test")
    Xtr, ytr = xy(train)
    Xte, yte = xy(test)
    labels = sorted(set(ytr))
    rare = rare_classes(cfg, train)
    seeds = cfg["experiment"]["seeds"][:2] if quick else cfg["experiment"]["seeds"]
    configs = {"RF-default": ("rf", None)}
    for m in TUNERS:
        p = best_params(cfg, m)
        if p:
            configs[f"RF-{m}" if m != "sa" else "SAORF"] = ("rf", p)
    configs.update({"XGBoost": ("xgb", None), "LightGBM": ("lgbm", None), "MLP": ("mlp", None)})
    for name, (kind, params) in configs.items():
        for seed in seeds:
            f = out / f"{name}_s{seed}.json"
            if f.exists():
                continue
            model = make_rf(params, seed, cfg["experiment"]["n_jobs"]) if kind == "rf" else make_detector(kind, seed)
            with Timer() as t:
                model.fit(Xtr, ytr)
            with Timer() as ti:
                pred = model.predict(Xte)
            r = evaluate(yte, pred, labels, rare)
            r.update(model=name, seed=seed, params=params, train_time_s=t.s, infer_time_s=ti.s)
            r["latency_single_ms"], r["latency_batch_us_per_flow"] = latency(model, Xte)
            r["model_size_mb"] = len(pickle.dumps(model)) / 2**20
            save_json(r, f)
            print(f"[final] {name:12s} s={seed} macroF1={r['macro_f1']:.4f} rareF1={r['rare_macro_f1']:.4f} "
                  f"train={t.s:.1f}s size={r['model_size_mb']:.1f}MB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["tune", "final", "all"])
    ap.add_argument("--quick", action="store_true", help="快速冒烟测试（少种子、小预算）")
    a = ap.parse_args()
    cfg = load_cfg()
    if a.quick:
        cfg["_tag"] = "_quick"  # 冒烟测试结果写到 results/<ds>_quick/
    if a.part in ("tune", "all"):
        part_tune(cfg, a.quick)
    if a.part in ("final", "all"):
        part_final(cfg, a.quick)
