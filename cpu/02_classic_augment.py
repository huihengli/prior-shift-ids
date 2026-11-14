"""[CPU · 轻量 / CTGAN-TVAE 为中等] 传统数据增强基线：为每个稀有类生成合成样本池
  ROS(随机过采样) / SMOTE / Borderline-SMOTE / ADASYN       —— 轻量，本机即可
  CTGAN / TVAE（需 `pip install sdv`，CPU 可跑但较慢，可放实验机）
输出：data/<ds>/synth/<method>.csv（每个稀有类 pool_per_class 条，已统一后处理）
所有生成器只生成 7 个原始特征，派生特征统一重算（与 LLM 方法一致）。
用法：python cpu/02_classic_augment.py [--methods ros smote bsmote adasyn ctgan tvae]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PRIMARY, exp_param, finalize_synth, load_cfg, load_split, rare_classes, stratified_cap, synth_dir


def pool_size(cfg):
    return max(exp_param(cfg, "aug_target_sweep"))


def smote_family(train, rare, n_pool, method, seed):
    from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
    X = np.log1p(train[PRIMARY].to_numpy(float))
    y = train["label"].to_numpy()
    vc = train["label"].value_counts()
    strat = {c: int(vc[c] + n_pool) for c in rare}
    kmin = int(min(vc[c] for c in rare)) - 1
    k = max(1, min(5, kmin))
    sampler = {
        "ros": RandomOverSampler(sampling_strategy=strat, random_state=seed),
        "smote": SMOTE(sampling_strategy=strat, k_neighbors=k, random_state=seed),
        "bsmote": BorderlineSMOTE(sampling_strategy=strat, k_neighbors=k, random_state=seed),
        "adasyn": ADASYN(sampling_strategy=strat, n_neighbors=k, random_state=seed),
    }[method]
    if method == "adasyn":
        # ADASYN 按类单独做：某类近邻中没有其他类（完全可分，如 CIC-IDS2017 的 Heartbleed）时 ADASYN 无定义，
        # 该类退化为 SMOTE（imblearn 的建议），并打印说明
        parts_X, parts_y = [], []
        for c in rare:
            try:
                Xr, yr = ADASYN(sampling_strategy={c: strat[c]}, n_neighbors=k, random_state=seed).fit_resample(X, y)
            except (RuntimeError, ValueError) as e:
                print(f"  [adasyn] {c}: {str(e)[:60]}… → 该类改用 SMOTE", flush=True)
                Xr, yr = SMOTE(sampling_strategy={c: strat[c]}, k_neighbors=k, random_state=seed).fit_resample(X, y)
            parts_X.append(Xr[len(X):]); parts_y.append(yr[len(X):])
        new_X, new_y = np.vstack(parts_X), np.concatenate(parts_y)
    else:
        Xr, yr = sampler.fit_resample(X, y)
        new_X, new_y = Xr[len(X):], yr[len(X):]
    out = pd.DataFrame(np.expm1(new_X), columns=PRIMARY)
    out["label"] = new_y
    return out


def sdv_model(train, rare, n_pool, method, seed, epochs=300):
    from sdv.sampling import Condition
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    sub = stratified_cap(train, 3000, seed)[PRIMARY + ["label"]].copy()
    # 稀有类在训练表中占比过低（如 8 / 3 万）时，按标签条件采样（SDV 用拒绝采样）几乎取不到样本；
    # 先把每个稀有类随机复制到至少 1000 行（只影响标签边缘分布，不引入新的特征值）
    parts = [sub]
    for c in rare:
        g = sub[sub.label == c]
        if len(g) < 1000:
            parts.append(g.sample(1000 - len(g), replace=True, random_state=seed))
    sub = pd.concat(parts, ignore_index=True)
    sub[PRIMARY] = np.log1p(sub[PRIMARY].astype(float))
    # 显式给出列类型，不依赖各版本 SDV 的自动检测接口（2.0 起删除了 SingleTableMetadata，
    # 检测函数也改成了 Metadata.detect_from_dataframes）
    cols = {c: {"sdtype": "numerical"} for c in PRIMARY}
    cols["label"] = {"sdtype": "categorical"}
    try:  # SDV ≥ 1.15 / 2.x
        from sdv.metadata import Metadata
        md = Metadata.load_from_dict({"tables": {"flows": {"columns": cols}}})
    except ImportError:  # 旧版 SDV
        from sdv.metadata import SingleTableMetadata
        md = SingleTableMetadata.load_from_dict({"columns": cols})
    Syn = CTGANSynthesizer if method == "ctgan" else TVAESynthesizer
    unwrap = lambda r: next(iter(r.values())) if isinstance(r, dict) else r  # SDV 2.x 采样返回 {表名: DataFrame}

    def fit(model, d):  # 新版 SDV 的 fit 需要 {表名: DataFrame}，旧版需要 DataFrame；两种都试
        try:
            model.fit({"flows": d})
        except Exception as e1:
            try:
                model.fit(d)
            except Exception:
                raise e1
    m = Syn(md, epochs=epochs, verbose=True)
    fit(m, sub)
    outs, mode = [], {}
    for c in rare:
        try:
            outs.append(unwrap(m.sample_from_conditions([Condition({"label": c}, num_rows=n_pool)], max_tries_per_batch=500)))
            mode[c] = "conditional"
        except Exception as e:
            # 回退：只用该类数据单独训练一个模型，无条件采样（不再依赖拒绝采样）
            print(f"  [{method}] {c}: 条件采样失败（{str(e)[:80]}），改为单独训练该类的模型", flush=True)
            sc = sub[sub.label == c].reset_index(drop=True)
            mc = Syn(md, epochs=epochs, verbose=False)
            fit(mc, sc)
            r = unwrap(mc.sample(num_rows=n_pool))
            r["label"] = c
            outs.append(r)
            mode[c] = "per_class"
    print(f"  [{method}] 采样方式：{mode}", flush=True)
    if not outs:
        raise RuntimeError(f"{method}: 所有稀有类的条件采样都失败")
    out = pd.concat(outs, ignore_index=True)
    out[PRIMARY] = np.expm1(out[PRIMARY])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["ros", "smote", "bsmote", "adasyn"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300, help="CTGAN / TVAE 训练轮数")
    a = ap.parse_args()
    cfg = load_cfg()
    train = load_split(cfg, "train")
    rare = rare_classes(cfg, train)
    n_pool = pool_size(cfg)
    print("rare classes:", rare, "pool/class:", n_pool)
    for m in a.methods:
        f = synth_dir(cfg) / f"{m}.csv"
        if f.exists():
            print("skip", f)
            continue
        if m in ("ctgan", "tvae"):
            raw = sdv_model(train, rare, n_pool, m, a.seed, a.epochs)
        else:
            raw = smote_family(train, rare, n_pool, m, a.seed)
        out = finalize_synth(raw, m)
        out.to_csv(f, index=False)
        print(m, out.label.value_counts().to_dict())


if __name__ == "__main__":
    main()
