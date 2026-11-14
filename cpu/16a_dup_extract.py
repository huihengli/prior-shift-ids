"""RQ1 重复样本假象：抽取“清洗后、未去重”的数据，压缩为 (7 个原始特征 + 标签) 的不重复行 + 出现次数。
输出 results/<ds>/dup_artefact/raw_counts.npz，供 16_dup_artefact.py 还原“未去重”的朴素划分。
用法：IDS_DATASET=<ds> python cpu/16a_dup_extract.py"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PRIMARY, canon_label, load_cfg, res_dir  # noqa: E402

spec = importlib.util.spec_from_file_location("prep", Path(__file__).with_name("00_prepare_data.py"))
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def main():
    cfg = load_cfg()
    df = prep.read_raw(cfg)
    n_raw = len(df)
    # 与 00_prepare_data.py 完全相同的清洗，只是不去重、不删冲突行
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=PRIMARY + ["label"])
    df = df[(df[PRIMARY] >= 0).all(axis=1)]
    df = df[(df["fwd_pkts"] + df["bwd_pkts"]) > 0]
    df[PRIMARY] = df[PRIMARY].astype(np.int64)
    df["label"] = df["label"].map(canon_label)
    g = df.groupby(PRIMARY + ["label"], sort=False).size().reset_index(name="count")
    labels = np.array(sorted(g["label"].unique()))
    out = res_dir(cfg, "dup_artefact")
    np.savez(out / "raw_counts.npz", X=g[PRIMARY].to_numpy(np.int64), y=np.searchsorted(labels, g["label"].to_numpy()),
             count=g["count"].to_numpy(np.int64), labels=labels, n_raw=n_raw, n_clean=len(df))
    print(cfg["_ds"], "raw", n_raw, "clean", len(df), "distinct", len(g))


if __name__ == "__main__":
    main()
