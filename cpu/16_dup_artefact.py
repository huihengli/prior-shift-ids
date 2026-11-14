"""RQ1 重复样本假象：同一划分程序下，“未去重”（常见做法）、“只去重”与本文协议（去重 + 删冲突行）的对比。
输入：results/<ds>/dup_artefact/raw_counts.npz（cpu/16a_dup_extract.py 生成）
两种口径都用 00_prepare_data.py 的分层划分 + 每类上限，划分种子 2026+r（r = 0..4）；
稀有类固定为本文协议原划分上的 13 个类（data/<ds>/train.csv.gz）。
输出：results/<ds>/dup_artefact/{summary.json, runs.csv}
用法：IDS_DATASET=<ds> python cpu/16_dup_artefact.py [--splits 5] [--detectors lgbm rf]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (FEATURES, PRIMARY, add_derived, evaluate, load_cfg, load_split, make_detector,  # noqa: E402
                    rare_classes, res_dir, stratified_cap, xy)


def split(df, sp, seed):
    """与 00_prepare_data.py 相同的分层划分 + 每类上限"""
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(df), bool)
    for _, idx in df.groupby("label").indices.items():
        idx = rng.permutation(idx)
        n_te = int(round(len(idx) * sp["test_size"]))
        if len(idx) >= 2:
            n_te = min(max(n_te, 1), len(idx) - 1)
        test_mask[idx[:n_te]] = True
    tr = stratified_cap(df[~test_mask], sp["cap_train_per_class"], seed, caps={"Benign": sp["cap_train_benign"]})
    te = stratified_cap(df[test_mask], sp["cap_test_per_class"], seed, caps={"Benign": sp["cap_test_benign"]})
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def exact_match_rate(tr, te, cls):
    """测试集中该类的流，有多少条的 7 个原始特征 + 标签在训练集中出现过"""
    t = te[te.label == cls]
    if not len(t):
        return float("nan")
    keys = set(map(tuple, tr.loc[tr.label == cls, PRIMARY].to_numpy()))
    return float(np.mean([tuple(r) in keys for r in t[PRIMARY].to_numpy()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--detectors", nargs="+", default=["lgbm", "rf"])
    a = ap.parse_args()
    cfg = load_cfg()
    sp = cfg["split"]
    out = res_dir(cfg, "dup_artefact")
    z = np.load(out / "raw_counts.npz", allow_pickle=True)
    base = pd.DataFrame(z["X"], columns=PRIMARY)
    base["label"] = z["labels"][z["y"]]
    cnt = z["count"]
    rare = rare_classes(cfg, load_split(cfg, "train"))

    # 本文协议：去重（npz 本身就是不重复行）+ 删冲突行
    nlab = base.groupby(PRIMARY)["label"].transform("nunique")
    clean = add_derived(base[nlab == 1].copy())[FEATURES + ["label"]].reset_index(drop=True)
    # 常见做法：不去重、保留冲突行（按出现次数还原）
    naive = add_derived(base.loc[base.index.repeat(cnt)].copy())[FEATURES + ["label"]].reset_index(drop=True)
    # 只去重、保留冲突行：把“重复”与“冲突标签”两种假象分开
    dedup = add_derived(base.copy())[FEATURES + ["label"]].reset_index(drop=True)
    summary = {"dataset": cfg["_ds"], "rare": rare, "n_naive": len(naive), "n_clean": len(clean),
               "class_naive": naive.label.value_counts().reindex(rare).fillna(0).astype(int).to_dict(),
               "class_clean": clean.label.value_counts().reindex(rare).fillna(0).astype(int).to_dict(),
               "conflict_rows": int((nlab > 1).sum())}
    del base
    rows = []
    for r in range(a.splits):
        seed = sp["seed"] + r
        for proto, df in (("naive", naive), ("dedup_only", dedup), ("protocol", clean)):
            tr, te = split(df, sp, seed)
            labels = sorted(tr.label.unique())
            em = {c: exact_match_rate(tr, te, c) for c in rare}
            any_overlap = len(te.merge(tr[FEATURES].drop_duplicates(), on=FEATURES))
            Xtr, ytr = xy(tr)
            Xte, yte = xy(te)
            for det in a.detectors:
                m = make_detector(det, seed=0, n_jobs=-1).fit(Xtr, ytr)
                pred = np.asarray(m.predict(Xte)).astype(str)
                ev = evaluate(yte, pred, labels, rare)
                ben = yte == "Benign"
                row = {"split": r, "protocol": proto, "detector": det,
                       "rare_macro_f1": float(np.mean([ev["per_class"][c]["f1"] for c in rare])),
                       "fp_benign_per_10k": float(np.isin(pred[ben], rare).mean() * 1e4),
                       "test_rows_seen_in_train": any_overlap / len(te),
                       "rare_test_rows_seen_in_train": float(np.nanmean(
                           [em[c] * (te.label == c).sum() for c in rare]) * len(rare) / te.label.isin(rare).sum())}
                for c in rare:
                    row[f"f1:{c}"] = ev["per_class"][c]["f1"]
                    row[f"ntest:{c}"] = int((te.label == c).sum())
                    row[f"seen:{c}"] = em[c]
                rows.append(row)
                print(cfg["_ds"], r, proto, det, round(row["rare_macro_f1"], 4),
                      round(row["rare_test_rows_seen_in_train"], 3), flush=True)
            pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
