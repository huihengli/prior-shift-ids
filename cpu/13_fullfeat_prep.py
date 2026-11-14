"""[CPU · 轻量，只需 numpy / pandas / pyyaml] 完整特征抽取：为现有 train/test 的每条样本取回原始文件中的全部流特征
动机：全部实验只用 10 维紧凑特征，需要回答“类间重叠是否只是删掉特征造成的”。
做法：逐块读原始 CSV，按与 00 相同的规则清洗；用 7 个原始特征 + 规范化标签作为键，
只保留键出现在现有 train/test 中的行（每个键取第一次出现的那一行）。
这样完整特征数据与 10 维实验是同一批样本、同一划分，可逐条对应比较。
标识类列（IP、源端口、时间戳、Flow ID、DNS 查询 ID 等）不作为特征；其余数值列以 x_ 前缀保存。
可分次运行：--max-seconds 到时即保存进度退出（输出 CONTINUE），再次运行会接着读；全部读完后输出 DONE。
用法：IDS_DATASET=cic2017 python cpu/13_fullfeat_prep.py [--max-seconds 150] [--raw-dir 路径]
输出：data/<ds>/fullfeat/full.csv.gz（列：split, label, 10 维统一特征, x_*）
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CIC17_MAP, CSE_MAP, FEATURES, NF_MAP, PRIMARY, ROOT, add_derived, canon_label, ds_dir, load_cfg  # noqa: E402

FULL_RAW = {"cic2017": "data/raw/cic2017"}  # 需要原始完整 CSV（全部特征列）
ID_COLS = {"Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp", "Src IP", "Src Port", "Dst IP",
           "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT",
           "DNS_QUERY_ID", "Label", "Attack"}


def raw_files(cfg, raw_dir):
    name = cfg["_ds"]
    dcfg = cfg["datasets"][name]
    if name == "cse2018":
        return [str(ROOT / f) for f in dcfg["raw_files"]], CSE_MAP
    if name == "cic2017":
        return sorted(glob.glob(str(ROOT / (raw_dir or FULL_RAW[name]) / "*.csv"))), CIC17_MAP
    return sorted(glob.glob(str(ROOT / (raw_dir or dcfg["raw_dir"]) / dcfg.get("pattern", "NF-*.csv")))), NF_MAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=1e9)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--raw-dir", default=None)
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_cfg()
    name = cfg["_ds"]
    out = ds_dir(cfg) / "fullfeat"
    out.mkdir(exist_ok=True)
    if (out / "full.csv.gz").exists():
        print("DONE（已存在）")
        return
    files, cmap = raw_files(cfg, a.raw_dir)
    if not files:
        sys.exit("找不到原始文件")
    keys = pd.concat([pd.read_csv(ds_dir(cfg) / f"{s}.csv.gz", usecols=PRIMARY + ["label"]).assign(split=s)
                      for s in ("train", "test")], ignore_index=True)
    st_f, part = out / "state.json", out / "part.csv"
    st = json.load(open(st_f)) if st_f.exists() and st_f.stat().st_size > 0 else {"file": 0, "rows": 0, "cols": None}
    if not (st_f.exists() and st_f.stat().st_size > 0) and part.exists():
        open(part, "w").close()  # 清空旧的中间文件（某些环境不允许删除文件）

    while st["file"] < len(files):
        f = files[st["file"]]
        head = pd.read_csv(f, nrows=0, encoding="latin1")
        rd = pd.read_csv(f, chunksize=a.chunk, encoding="latin1", low_memory=False,
                         skiprows=range(1, st["rows"] + 1) if st["rows"] else None)
        finished = True
        for ch in rd:
            n_read = len(ch)
            ch.columns = [c.strip() for c in head.columns]
            lab_col = [k for k, v in cmap.items() if v == "label"][0]
            feat_cols = [c for c in ch.columns if c not in cmap and c not in ID_COLS]
            x = ch[feat_cols].apply(pd.to_numeric, errors="coerce")
            x.columns = ["x_" + c for c in feat_cols]
            d = ch[[k for k in cmap if k != lab_col]].rename(columns=cmap).apply(pd.to_numeric, errors="coerce")
            d["label"] = ch[lab_col]
            d = pd.concat([d, x], axis=1).replace([np.inf, -np.inf], np.nan).dropna(subset=PRIMARY + ["label"])
            d = d[(d[PRIMARY] >= 0).all(axis=1)]
            d = d[(d["fwd_pkts"] + d["bwd_pkts"]) > 0]
            if name not in ("cse2018", "cic2017"):
                d["flow_duration"] = d["flow_duration"] * 1000
            d[PRIMARY] = d[PRIMARY].astype(np.int64)
            d["label"] = d["label"].map(canon_label)
            d = d.drop_duplicates(subset=PRIMARY + ["label"]).merge(keys, on=PRIMARY + ["label"], how="inner")
            if st["cols"] is None:
                st["cols"] = list(d.columns)
            d.reindex(columns=st["cols"]).to_csv(part, mode="a", header=not (part.exists() and part.stat().st_size > 0), index=False)
            st["rows"] += n_read
            json.dump(st, open(st_f, "w"))
            print(f"{Path(f).name}: 已读 {st['rows']} 行，本块匹配 {len(d)}，用时 {time.time() - t0:.0f}s", flush=True)
            if time.time() - t0 > a.max_seconds:
                finished = False
                break
        if not finished:
            print("CONTINUE")
            return
        st["file"] += 1
        st["rows"] = 0
        json.dump(st, open(st_f, "w"))

    d = pd.read_csv(part, low_memory=False).drop_duplicates(subset=PRIMARY + ["label"])
    d = add_derived(d)
    xcols = [c for c in d.columns if c.startswith("x_")]
    d = d[["split", "label"] + FEATURES + xcols]
    d.to_csv(out / "full.csv.gz", index=False)
    cov = {s: f"{int((d.split == s).sum())}/{int((keys.split == s).sum())}" for s in ("train", "test")}
    json.dump({"rows": len(d), "matched_of_total": cov, "n_extra_features": len(xcols), "extra_features": xcols},
              open(out / "fullfeat_stats.json", "w"), indent=1)
    for p in (part, st_f):  # 清理中间文件；不允许删除时清空即可
        try:
            p.unlink()
        except OSError:
            open(p, "w").close()
    print("DONE", cov, "额外特征数", len(xcols))


if __name__ == "__main__":
    main()
