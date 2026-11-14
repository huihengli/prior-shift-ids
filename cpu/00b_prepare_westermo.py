"""[CPU · 轻量] Westermo 工业网络流量数据集：清洗与固定划分（附录外部验证用，只跑 CPU 实验）
与 00_prepare_data.py 的协议相同（统一 10 维特征、在 7 个原始特征 + 标签上去重、删除冲突行、分层固定划分、
训练/测试零重复），针对该数据集有以下差异，论文附录需如实说明：
  1) 标签：使用 NST_M_Label（只把进出攻击者的流量标为异常；IT 策略把攻击期间的全部流量都标为异常，噪声过大）。
  2) 剔除 BAD-MISCONF / BAD-MISCONF-DUPLICATION：属于设备配置错误而非攻击，NST 把配置错误设备的全部流量标为异常，
     流特征与正常流量无差别（kNN 同类纯度 0.06–0.09，RF F1≈0.07）。
  3) GOOD-SSH（合法 SSH 登录）并入 Benign。
  4) 端口：原始流文件中没有端口字段，dst_port 置 0（常数列，等价于 9 维特征）。
     协议：IPV4-TCP→6、IPV4-UDP→17、IPV4-ICMP→1、IPV4-2(IGMP)→2；非 IP 帧用 EtherType：ARP→2054、LLDP→35020。
  5) 三个采集点（left / right / bottom）会记录到同一条流。划分时按“流键”（端点 + 协议 + 起始时间 10 ms 粒度）分组，
     同一流键的所有副本只进训练或只进测试，避免跨采集点泄漏。
  6) datasets.<ds>.merge_portscan = true 时把 BAD-PORTSCAN1（扫描单台设备）与 BAD-PORTSCAN2（扫描多台设备）合并为 PortScan：
     二者是同一 nmap 扫描，区别只在扫描范围，在单条流上不可区分（PORTSCAN1 的近邻 72% 为 PORTSCAN2）。
     数据集 westermo 为合并口径（主口径），westermo_ps 为不合并口径（“标签定义造成重叠”的压力测试）。
用法：IDS_DATASET=westermo python cpu/00b_prepare_westermo.py
      IDS_DATASET=westermo_ps python cpu/00b_prepare_westermo.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEATURES, PRIMARY, ROOT, add_derived, ds_dir, load_cfg, rare_threshold, save_json, stratified_cap

PROTO = {"IPV4-TCP": 6, "IPV4-UDP": 17, "IPV4-ICMP": 1, "IPV4-2": 2, "ARP": 2054, "LLDP": 35020}
VANTAGES = ["left", "right", "bottom"]


def read_raw(dcfg) -> pd.DataFrame:
    parts = []
    for v in VANTAGES:
        f = ROOT / dcfg["raw_dir"] / f"output_{v}.csv"
        print("reading", f, flush=True)
        d = pd.read_csv(f, low_memory=False)
        parts.append(pd.DataFrame({
            "dst_port": 0,
            "protocol": d["protocol"].map(PROTO),
            "flow_duration": (d["duration"].astype(float) * 1e6).round(),  # 秒 -> 微秒
            "fwd_pkts": d["sPackets"], "bwd_pkts": d["rPackets"],
            "fwd_bytes": d["sBytesSum"], "bwd_bytes": d["rBytesSum"],
            "raw_label": d[dcfg.get("label_col", "NST_M_Label")].astype(str),
            "gkey": d["sAddress"].astype(str) + "|" + d["rAddress"].astype(str) + "|" + d["protocol"].astype(str)
                    + "|" + (d["start"].astype(float) * 100).round().astype("int64").astype(str),
            "vantage": v,
        }))
    return pd.concat(parts, ignore_index=True)


def map_label(x: str, merge_ps: bool) -> str | None:
    if x in ("Normal", "GOOD-SSH"):
        return "Benign"
    if x.startswith("BAD-MISCONF"):
        return None
    if x == "BAD-SSH":
        return "SSH-BruteForce"
    if x == "BAD-MITM":
        return "MITM"
    if x.startswith("BAD-PORTSCAN"):
        return "PortScan" if merge_ps else ("PortScan-Single" if x.endswith("1") else "PortScan-Multi")
    return x


def main():
    cfg = load_cfg()
    dcfg = cfg["datasets"][cfg["_ds"]]
    sp = cfg["split"]
    out = ds_dir(cfg)
    stats = {"label_strategy": dcfg.get("label_col", "NST_M_Label"), "merge_portscan": bool(dcfg.get("merge_portscan"))}

    df = read_raw(dcfg)
    stats["raw_rows"] = len(df)
    stats["raw_label_counts"] = df["raw_label"].value_counts().to_dict()
    df["label"] = df["raw_label"].map(lambda x: map_label(x, bool(dcfg.get("merge_portscan"))))
    stats["removed_misconf_rows"] = int(df["label"].isna().sum())
    df = df.dropna(subset=["label"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=PRIMARY)
    df = df[(df[PRIMARY] >= 0).all(axis=1)]
    df = df[(df["fwd_pkts"] + df["bwd_pkts"]) > 0]
    df[PRIMARY] = df[PRIMARY].astype(np.int64)
    stats["after_clean"] = len(df)

    # 去重与冲突删除（与 00 相同）；保留第一次出现的流键用于分组
    df = df.drop_duplicates(subset=PRIMARY + ["label"])
    stats["after_dedup"] = len(df)
    nlab = df.groupby(PRIMARY)["label"].transform("nunique")
    stats["conflict_rows_removed"] = int((nlab > 1).sum())
    df = df[nlab == 1]
    df = add_derived(df).reset_index(drop=True)
    stats["final_unique_rows"] = len(df)
    stats["dedup_label_counts"] = df["label"].value_counts().to_dict()
    gsize = df.groupby("gkey")["gkey"].transform("size")
    stats["rows_in_multi_row_flow_keys"] = int((gsize > 1).sum())

    # 分组 + 按类分层的固定划分：每个流键整体归入训练或测试；流键的类别取其多数标签
    rng = np.random.default_rng(sp["seed"])
    glab = df.groupby("gkey")["label"].agg(lambda s: s.value_counts().index[0])
    test_keys = set()
    for lab, keys in glab.groupby(glab).groups.items():
        keys = rng.permutation(np.asarray(keys))
        n_te = int(round(len(keys) * sp["test_size"]))
        if len(keys) >= 2:
            n_te = min(max(n_te, 1), len(keys) - 1)
        test_keys.update(keys[:n_te])
    is_te = df["gkey"].isin(test_keys)
    pool, test_full = df[~is_te], df[is_te]
    stats["flow_key_overlap_train_test"] = int(len(set(pool.gkey) & set(test_full.gkey)))

    pool, test_full = pool[FEATURES + ["label"]], test_full[FEATURES + ["label"]]
    train = stratified_cap(pool, sp["cap_train_per_class"], sp["seed"], caps={"Benign": sp["cap_train_benign"]})
    test = stratified_cap(test_full, sp["cap_test_per_class"], sp["seed"], caps={"Benign": sp["cap_test_benign"]})
    pool.to_csv(out / "train_pool.csv.gz", index=False)
    train.to_csv(out / "train.csv.gz", index=False)
    test.to_csv(out / "test.csv.gz", index=False)

    raw_mapped = pd.Series(stats["raw_label_counts"])
    tab = pd.DataFrame({"unique": pd.Series(stats["dedup_label_counts"]),
                        "train": train.label.value_counts(), "test": test.label.value_counts()}).fillna(0).astype(int)
    tab["rare"] = tab["train"] < rare_threshold(cfg)
    tab.sort_values("train", ascending=False).to_csv(out / "class_table.csv")
    stats["raw_label_counts"] = raw_mapped.to_dict()
    stats["train_rows"], stats["test_rows"] = len(train), len(test)
    stats["train_test_overlap"] = int(len(train.merge(test, on=FEATURES)))
    save_json(stats, out / "prep_stats.json")
    print(tab.sort_values("train", ascending=False))
    print({k: v for k, v in stats.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
