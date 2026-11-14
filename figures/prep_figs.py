"""论文图的数据准备（Fig. 1、Fig. 3）：从 实验/v2/results/ 读取逐次评估 JSON，输出 pgfplots 用的 .dat 表
用法：python prep_figs.py <v2 目录> <输出目录 data/>
  Fig. 1  fig1_delta.dat     重复划分 r1–r3 上各来源相对“不增强”的稀有类 macro-F1 差（每份划分先对 5 个种子取均值，
                             再对 3 份划分取均值；误差棒 = 配对差值在划分间的标准差）
  Fig. 3a fig3a_variance.dat 种子间 sd（原划分 10 种子）vs 划分间 sd（r1–r5，每份划分先对 5 种子取均值），
                             对 none / cw / ros / smote 四个来源取平均
  Fig. 3b fig3b_cic_lgbm.dat CIC-IDS2017 · LightGBM：原划分与 r1–r3 上各来源的稀有类 macro-F1（均值 ± 种子 sd）
  Fig. 5  fig5_<ds>.dat      原划分（10 种子）：只调阈值 cvps / 类别权重 / ROS 相对不增强的 ΔF1 与 ΔPR-AUC（配对差的均值 ± 种子间 sd）
  Fig. 4  fig4_<来源>.dat     原划分（10 种子）：13 个稀有类的 ΔF1（相对不增强，4 个检测器的均值、最小值、最大值）对真实样本 10-NN 纯度
  Fig. 7  fig7_<指标>.dat     10 维 vs 完整特征（results/v3_summary/fullfeat_eval.csv，5 种子）：不增强与三种增强的稀有类 F1 / PR-AUC
  Fig. 6  fig6_<口径>.dat     受控实验（results/v3_summary/controlled2_{matched,unmatched}.csv）：按下采样后纯度分组的
                             ΔF1 / ΔPR-AUC 均值与 95% 置信区间（单位 = 类 × n × 检测器）
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V2, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
DS = {"cse2018": "CSE-CIC-IDS2018", "nf_unsw_v3": "NF-UNSW-NB15-v3", "cic2017": "CIC-IDS2017"}
DET = ["SAORF", "RF-grid", "LightGBM", "XGBoost"]


def qwen(ds, name):  # CSE 上 Qwen 来源名带 v2_ 前缀（重复划分上没有）
    return ("v2_" if ds == "cse2018" else "") + name


def load(ds_dir):
    rows = []
    for f in (V2 / "results" / ds_dir / "augment_proba").glob("*__s*.json"):
        j = json.load(open(f, encoding="utf-8"))
        rows.append(dict(source=j["source"], detector=j["detector"], seed=j["seed"], f1=j["rare_macro_f1"],
                         ap=j.get("rare_macro_ap", np.nan)))
        if "cvps" in j:  # 只改决策规则：分数（因而 PR-AUC）与不增强相同
            rows.append(dict(source="none+cvps", detector=j["detector"], seed=j["seed"],
                             f1=j["cvps"]["rare_macro_f1"], ap=j.get("rare_macro_ap", np.nan)))
    return pd.DataFrame(rows)


# ---------------- Fig. 1：重复划分 r1–r3，ΔF1 vs none
SRC1 = [("cw", "cw", "Class weight"), ("ros", "ros", "ROS"), ("smote", "smote", "SMOTE"),
        ("qwen", "llm_sft_relov_filt", "LLM (Qwen2.5-7B)"), ("llama", "llm_sft_llama_relov_filt", "LLM (Llama-3.1-8B)")]
rows = []
for ds in DS:
    rep = pd.concat([load(f"{ds}_r{r}").assign(rep=r) for r in (1, 2, 3)])
    m = rep.groupby(["detector", "rep", "source"]).f1.mean().unstack("source")
    for det in DET:
        x = m.loc[det]
        for key, src, _ in SRC1:
            d = (x[src] - x["none"]).dropna()
            assert len(d) == 3, (ds, det, src, len(d))
            rows.append(dict(dataset=ds, detector=det, source=key, delta=d.mean(), sd=d.std(ddof=0),
                             wins=int((d > 0).sum())))
t1 = pd.DataFrame(rows)
# pgfplots 宽表：每个数据集一个文件，行 = 检测器，列 = 各来源 delta / sd
for ds in DS:
    w = t1[t1.dataset == ds].pivot(index="detector", columns="source", values=["delta", "sd"]).reindex(DET)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w.insert(0, "x", range(len(DET)))
    w.insert(1, "det", DET)
    w.to_csv(OUT / f"fig1_{ds}.dat", sep=" ", index=False, float_format="%.5f")
t1.to_csv(OUT / "fig1_long.csv", index=False)

# ---------------- Fig. 3a：种子间 sd vs 划分间 sd
SRC3 = ["none", "cw", "ros", "smote"]
rows = []
for ds in DS:
    o = load(ds)
    reps = pd.concat([load(f"{ds}_r{r}").assign(rep=r) for r in range(1, 6)])
    for det in DET:
        seed_sd = np.mean([o[(o.detector == det) & (o.source == s)].f1.std(ddof=0) for s in SRC3])
        sm = reps[reps.detector == det].groupby(["source", "rep"]).f1.mean()
        split_sd = np.mean([sm.loc[s].std(ddof=0) for s in SRC3])
        n_rep = min(len(sm.loc[s]) for s in SRC3)
        assert n_rep == 5, (ds, det, n_rep)
        rows.append(dict(dataset=ds, detector=det, label=f"{DS[ds].split('-')[0]} {det}", seed_sd=seed_sd, split_sd=split_sd))
t3 = pd.DataFrame(rows)
t3.insert(0, "y", range(len(t3))[::-1])
t3["label"] = [f"{ {'cse2018': 'CSE', 'nf_unsw_v3': 'NF', 'cic2017': 'CIC'}[d] } · {k}" for d, k in zip(t3.dataset, t3.detector)]
t3.to_csv(OUT / "fig3a_variance.dat", sep=" ", index=False, float_format="%.5f",
          columns=["y", "seed_sd", "split_sd"])
t3.to_csv(OUT / "fig3a_variance_labels.csv", index=False)

# ---------------- Fig. 3b：CIC · LightGBM，原划分 vs r1–r3
SRC3B = [("none", "none"), ("cw", "cw"), ("ros", "ros"),
         ("qwen", "llm_sft_relov_filt"), ("llama", "llm_sft_llama_filt")]
rows = []
for k, name in enumerate(["cic2017", "cic2017_r1", "cic2017_r2", "cic2017_r3"]):
    d = load(name)
    d = d[d.detector == "LightGBM"]
    r = {"x": k}
    for key, src in SRC3B:
        v = d[d.source == src].f1
        r[f"{key}"], r[f"{key}_sd"] = v.mean(), v.std(ddof=0)
    rows.append(r)
t3b = pd.DataFrame(rows)
t3b.to_csv(OUT / "fig3b_cic_lgbm.dat", sep=" ", index=False, float_format="%.5f")

# ---------------- Fig. 5：原划分，阈值移动 vs 重新加权，F1 与 PR-AUC
rows = []
for ds in DS:
    o = load(ds)
    for det in DET:
        x = o[o.detector == det]
        f1 = x.pivot_table(index="seed", columns="source", values="f1")
        ap = x.pivot_table(index="seed", columns="source", values="ap")
        assert len(f1) == 10, (ds, det, len(f1))
        r = dict(dataset=ds, detector=det)
        for key, src in [("cvps", "none+cvps"), ("cw", "cw"), ("ros", "ros")]:
            d = f1[src] - f1["none"]
            r[f"df1_{key}"], r[f"df1sd_{key}"] = d.mean(), d.std(ddof=0)
            a = ap[src] - ap["none"]
            r[f"dap_{key}"], r[f"dapsd_{key}"] = a.mean(), a.std(ddof=0)
        rows.append(r)
t5 = pd.DataFrame(rows)
for ds in DS:
    w = t5[t5.dataset == ds].drop(columns="dataset").set_index("detector").reindex(DET).reset_index()
    w.insert(0, "x", range(len(DET)))
    w.rename(columns={"detector": "det"}).to_csv(OUT / f"fig5_{ds}.dat", sep=" ", index=False, float_format="%.5f")
t5.to_csv(OUT / "fig5_long.csv", index=False)

# ---------------- Fig. 6：受控实验，按纯度分组
BINS = [(0, "overlap", "Overlapping (<0.5)", lambda p: p < 0.5),
        (1, "mid", "Medium (0.5--0.8)", lambda p: (p >= 0.5) & (p <= 0.8)),
        (2, "sep", "Separable (>0.8)", lambda p: p > 0.8)]
rows6 = []
for prior in ("matched", "unmatched"):
    c = pd.read_csv(V2 / "results" / "v3_summary" / f"controlled2_{prior}.csv")
    rows = []
    for x, key, lab, f in BINS:
        g = c[f(c.purity)]
        r = dict(x=x, n=len(g))
        for m in ("ps", "cw", "ros", "smote"):
            for met, col in (("df1", f"d_{m}"), ("dap", f"dap_{m}")):
                v = g[col]
                r[f"{met}_{m}"] = v.mean()
                r[f"{met}ci_{m}"] = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
        rows.append(r)
        rows6.append(dict(prior=prior, bin=key, **r))
    pd.DataFrame(rows).to_csv(OUT / f"fig6_{prior}.dat", sep=" ", index=False, float_format="%.5f")
t6 = pd.DataFrame(rows6)
t6.to_csv(OUT / "fig6_long.csv", index=False)

# ---------------- Fig. 4：按类 ΔF1 对纯度
pur = pd.read_csv(V2 / "results" / "overlap_gain" / "per_class.csv")[["dataset", "class", "real_purity", "n_train", "n_test"]]
DSNAME = {"cse2018": "CSE-CIC-IDS2018", "nf_unsw_v3": "NF-UNSW-NB15-v3", "cic2017": "CIC-IDS2017"}
SRC4 = [("cw", "cw"), ("ros", "ros"), ("smote", "smote"), ("qwen", "llm_sft_relov_filt"), ("llama", "llm_sft_llama_relov_filt")]
rows = []
for k, ds in enumerate(DS):
    for f in (V2 / "results" / ds / "augment_proba").glob("*__s*.json"):
        j = json.load(open(f, encoding="utf-8"))
        for c in j["ap"]:
            rows.append(dict(ds=ds, dsid=k, source=j["source"], detector=j["detector"], seed=j["seed"], cls=c,
                             f1=j["per_class"][c]["f1"]))
pc = pd.DataFrame(rows)
m = pc.groupby(["ds", "dsid", "cls", "detector", "source"]).f1.mean().unstack("source")
out4 = []
for key, src in SRC4:
    s_ = src if not (key == "qwen") else None
    d = []
    for (ds, dsid, cls, det), r in m.iterrows():
        sname = ("v2_" + src) if (key == "qwen" and ds == "cse2018") else src
        d.append(dict(ds=ds, dsid=dsid, cls=cls, detector=det, delta=r[sname] - r["none"]))
    d = pd.DataFrame(d).groupby(["ds", "dsid", "cls"]).delta.agg(["mean", "min", "max"]).reset_index()
    d = d.merge(pur.assign(ds=pur.dataset.map({v: k for k, v in DSNAME.items()})), left_on=["ds", "cls"], right_on=["ds", "class"])
    d["eplus"], d["eminus"] = d["max"] - d["mean"], d["mean"] - d["min"]
    d = d.sort_values(["dsid", "real_purity"])
    d[["dsid", "real_purity", "mean", "eplus", "eminus"]].to_csv(OUT / f"fig4_{key}.dat", sep=" ", index=False, float_format="%.5f")
    out4.append(d.assign(source=key))
t4 = pd.concat(out4)
t4[["source", "ds", "cls", "n_train", "n_test", "real_purity", "mean", "min", "max"]].to_csv(OUT / "fig4_long.csv", index=False)

# ---------------- Fig. 7：10 维 vs 完整特征
ff = pd.read_csv(V2 / "results" / "v3_summary" / "fullfeat_eval.csv")
ORDER = [(ds, det) for ds in ("cse2018", "nf_unsw_v3", "cic2017") for det in ("SAORF", "LightGBM", "XGBoost")]
for met in ("rare_f1", "rare_ap"):
    p = ff.pivot_table(index=["dataset", "detector"], columns=["space", "source"], values=met)
    rows = []
    for i, (ds, det) in enumerate(ORDER):
        r = p.loc[(ds, det)]
        aug10 = [r[("d10", s)] for s in ("cw", "ros", "smote")]
        augf = [r[("full", s)] for s in ("cw", "ros", "smote")]
        rows.append(dict(y=len(ORDER) - 1 - i, none10=r[("d10", "none")], lo10=min(aug10), hi10=max(aug10),
                         nonefull=r[("full", "none")], lofull=min(augf), hifull=max(augf),
                         mid10=(min(aug10) + max(aug10)) / 2, half10=(max(aug10) - min(aug10)) / 2,
                         midfull=(min(augf) + max(augf)) / 2, halffull=(max(augf) - min(augf)) / 2))
    pd.DataFrame(rows).to_csv(OUT / f"fig7_{met}.dat", sep=" ", index=False, float_format="%.5f")

pd.set_option("display.width", 200)
print(t1.round(4).to_string(index=False))
print(t3.round(4).to_string(index=False))
print(t3b.round(4).to_string(index=False))
print(t5.round(4).to_string(index=False))
print(t4[["source", "ds", "cls", "real_purity", "mean", "min", "max"]].round(3).to_string(index=False))
print(t6[["prior", "bin", "n"] + [c for c in t6.columns if c.startswith(("df1_", "dap_"))]].round(4).to_string(index=False))
