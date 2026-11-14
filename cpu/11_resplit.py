"""[CPU · 轻量] 重复分层划分：为每个数据集生成 R 份新的 train/test（数据集名 <ds>_r1 … <ds>_rR）
动机：原划分只有一份，稀有类测试样本极少（如 Web-SQLi 4–7 条），10 个种子只反映训练随机性，
不反映“哪几条样本被分到测试集”的不确定性。
做法：把原 train ∪ test（均为去重后的真实样本，二者零重叠）按类别重新随机划分，
每类的训练/测试条数与原划分完全相同，只改变成员。种子 = split.seed + r。
新数据集沿用原数据集的全部配置（common.load_cfg 中的别名）与调参结果（复制 detectors/tune）。
注意：超参数是在原划分的训练集上调的，对重复划分存在轻微的信息泄漏；所有来源共用同一组超参数，不影响来源间比较。
用法：python cpu/11_resplit.py [--reps 5] [--datasets cse2018 nf_unsw_v3 cic2017]
之后所有脚本用 IDS_DATASET=<ds>_r<k> 即可（02 传统增强、03/12 评估、gpu/run_gpu.sh）。
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, FEATURES, RESULTS, load_cfg, rare_threshold, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--datasets", nargs="+", default=["cse2018", "nf_unsw_v3", "cic2017"])
    a = ap.parse_args()
    for ds in a.datasets:
        os.environ["IDS_DATASET"] = ds
        cfg = load_cfg()
        tr = pd.read_csv(DATA / ds / "train.csv.gz")
        te = pd.read_csv(DATA / ds / "test.csv.gz")
        allv = pd.concat([tr, te], ignore_index=True)
        assert not allv.duplicated(subset=FEATURES).any(), "train/test 存在重复行"
        n_te = te.label.value_counts()
        for r in range(1, a.reps + 1):
            name = f"{ds}_r{r}"
            out = DATA / name
            if (out / "test.csv.gz").exists():
                print("skip", name)
                continue
            out.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(cfg["split"]["seed"] + r)
            mask = np.zeros(len(allv), bool)
            for lab, idx in allv.groupby("label").indices.items():
                mask[rng.permutation(idx)[: int(n_te.get(lab, 0))]] = True
            ntr, nte = allv[~mask].reset_index(drop=True), allv[mask].reset_index(drop=True)
            ntr.to_csv(out / "train.csv.gz", index=False)
            nte.to_csv(out / "test.csv.gz", index=False)
            tab = pd.DataFrame({"train": ntr.label.value_counts(), "test": nte.label.value_counts()}).fillna(0).astype(int)
            tab["rare"] = tab["train"] < rare_threshold(cfg)
            tab.sort_values("train", ascending=False).to_csv(out / "class_table.csv")
            moved = int(len(nte.merge(te, on=FEATURES + ["label"])))
            save_json({"base": ds, "rep": r, "seed": cfg["split"]["seed"] + r, "train_rows": len(ntr),
                       "test_rows": len(nte), "train_test_overlap": int(len(ntr.merge(nte, on=FEATURES))),
                       "test_rows_shared_with_original_test": moved}, out / "prep_stats.json")
            src, dst = RESULTS / ds / "detectors" / "tune", RESULTS / name / "detectors" / "tune"
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
            print(name, "train", len(ntr), "test", len(nte), "与原测试集相同的行", moved, flush=True)


if __name__ == "__main__":
    main()
