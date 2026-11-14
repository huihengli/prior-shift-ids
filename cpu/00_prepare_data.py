"""[CPU · 轻量] 数据清洗与固定划分
修复旧实验的问题：
  1) 标签拼写统一（旧版 'DDOSattack-HOIC' 与 'DDOS attack-HOIC' 被当成两类）
  2) 先去重、删除“同特征不同标签”的冲突样本，再划分 → 杜绝训练/测试重复行泄漏
  3) 所有方法共用同一套 10 维特征；派生特征统一重算
  4) 一次性固定 train/test 划分；测试集只含真实数据
用法：  python cpu/00_prepare_data.py            （数据集由 config.yaml 的 dataset 决定）
        IDS_DATASET=cic2017 python cpu/00_prepare_data.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (CIC17_MAP, CSE_MAP, FEATURES, NF_MAP, PRIMARY, ROOT, add_derived, canon_label,
                    ds_dir, load_cfg, rare_threshold, save_json, stratified_cap)


def read_raw(cfg) -> pd.DataFrame:
    name = cfg["_ds"]
    dcfg = cfg["datasets"][name]
    if name == "cse2018":
        files, cmap = [ROOT / f for f in dcfg["raw_files"]], CSE_MAP
    elif name == "cic2017":
        files, cmap = sorted(glob.glob(str(ROOT / dcfg["raw_dir"] / "*.csv"))), CIC17_MAP
    else:  # NetFlow v2 / v3 版本（字段名相同）
        files, cmap = sorted(glob.glob(str(ROOT / dcfg["raw_dir"] / dcfg.get("pattern", "NF-*.csv")))), NF_MAP
    if not files:
        sys.exit(f"找不到原始文件：{dcfg}")
    parts = []
    for f in files:
        print("reading", f, flush=True)
        head = pd.read_csv(f, nrows=0, encoding="latin1")
        strip = {c: c.strip() for c in head.columns}
        use = [c for c in head.columns if c.strip() in cmap]
        for ch in pd.read_csv(f, usecols=use, chunksize=400_000, encoding="latin1", low_memory=False):
            ch = ch.rename(columns=strip).rename(columns=cmap)
            for c in PRIMARY:
                ch[c] = pd.to_numeric(ch[c], errors="coerce")
            parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    if name not in ("cse2018", "cic2017"):
        df["flow_duration"] = df["flow_duration"] * 1000  # 毫秒 -> 微秒
    return df


def main():
    cfg = load_cfg()
    sp = cfg["split"]
    out = ds_dir(cfg)
    stats = {}

    df = read_raw(cfg)
    stats["raw_rows"] = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=PRIMARY + ["label"])
    df = df[(df[PRIMARY] >= 0).all(axis=1)]
    df = df[(df["fwd_pkts"] + df["bwd_pkts"]) > 0]
    df[PRIMARY] = df[PRIMARY].astype(np.int64)
    df["label"] = df["label"].map(canon_label)
    stats["after_clean"] = len(df)
    stats["raw_label_counts"] = df["label"].value_counts().to_dict()

    # 去重（派生特征由 PRIMARY 唯一决定，所以在 PRIMARY+label 上去重即可）
    df = df.drop_duplicates(subset=PRIMARY + ["label"])
    stats["after_dedup"] = len(df)
    # 冲突：同一特征向量对应多个标签 -> 全部删除（不可判定样本）
    nlab = df.groupby(PRIMARY)["label"].transform("nunique")
    stats["conflict_rows_removed"] = int((nlab > 1).sum())
    df = df[nlab == 1]
    df = add_derived(df)[FEATURES + ["label"]].reset_index(drop=True)
    stats["final_unique_rows"] = len(df)
    stats["dedup_label_counts"] = df["label"].value_counts().to_dict()

    # 分层固定划分（不依赖 sklearn）
    rng = np.random.default_rng(sp["seed"])
    test_mask = np.zeros(len(df), bool)
    for lab, idx in df.groupby("label").indices.items():
        idx = rng.permutation(idx)
        n_te = int(round(len(idx) * sp["test_size"]))
        if len(idx) >= 2:
            n_te = min(max(n_te, 1), len(idx) - 1)
        test_mask[idx[:n_te]] = True
    pool, test_full = df[~test_mask], df[test_mask]

    train = stratified_cap(pool, sp["cap_train_per_class"], sp["seed"], caps={"Benign": sp["cap_train_benign"]})
    test = stratified_cap(test_full, sp["cap_test_per_class"], sp["seed"], caps={"Benign": sp["cap_test_benign"]})
    # 去掉在训练中样本极少（<5）的类：无法稳定评估
    vc = train["label"].value_counts()
    drop = vc[vc < 5].index.tolist()
    if drop:
        print("drop too-rare classes:", drop)
        train, test = train[~train.label.isin(drop)], test[~test.label.isin(drop)]
        pool = pool[~pool.label.isin(drop)]
    stats["dropped_classes"] = drop

    pool.to_csv(out / "train_pool.csv.gz", index=False)
    train.to_csv(out / "train.csv.gz", index=False)
    test.to_csv(out / "test.csv.gz", index=False)

    tab = pd.DataFrame({"raw": pd.Series(stats["raw_label_counts"]),
                        "unique": pd.Series(stats["dedup_label_counts"]),
                        "train": train.label.value_counts(), "test": test.label.value_counts()}).fillna(0).astype(int)
    tab["rare"] = tab["train"] < rare_threshold(cfg)
    tab.sort_values("train", ascending=False).to_csv(out / "class_table.csv")
    stats["train_rows"], stats["test_rows"] = len(train), len(test)
    # 交叉检查：训练/测试零重复
    stats["train_test_overlap"] = int(len(train.merge(test, on=FEATURES)))
    save_json(stats, out / "prep_stats.json")
    print(tab.sort_values("train", ascending=False))
    print({k: v for k, v in stats.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
