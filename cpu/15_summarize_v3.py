"""[CPU · 轻量] 汇总 2026-11-11 补充实验（12 / 13b / 14 的结果），输出表格与 summary.md
  A. 概率版评估（原划分）：稀有类 F1、PR-AUC、良性误报率；先验平移基线 none+ps / none+cvps；与 none、ROS 的配对 Wilcoxon
  B. 重复划分：各来源在 R 份划分上的均值、划分间标准差、相对 none / ROS 的胜出份数
  C. 受控实验：下采样规模 n × 纯度 → 各方法相对 none 的 F1 变化
  D. 完整特征：d10 与 full 两个特征空间上的 none / cw / ros / smote
缺哪部分结果就跳过哪部分。
用法：python cpu/15_summarize_v3.py        输出：results/v3_summary/
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS  # noqa: E402

BASES = ["cse2018", "nf_unsw_v3", "cic2017"]
OUT = RESULTS / "v3_summary"
OUT.mkdir(parents=True, exist_ok=True)
md = ["# 补充实验汇总（自动生成：cpu/15_summarize_v3.py）", ""]


def wil(a, b):
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 6 or np.allclose(a, b):
        return np.nan
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return np.nan


def load_proba(ds):
    rows = []
    for f in (RESULTS / ds / "augment_proba").glob("*__s*.json"):
        j = json.load(open(f, encoding="utf-8"))
        base = dict(dataset=ds, detector=j["detector"], seed=j["seed"], target=j["target"])
        rows.append(dict(base, source=j["source"], rare_f1=j["rare_macro_f1"], rare_ap=j["rare_macro_ap"],
                         fp10k=j["fp_benign_per_10k"], rare_prec=j["rare_macro_precision"], rare_rec=j["rare_macro_recall"],
                         **{f"F1[{c}]": v["f1"] for c, v in j["per_class"].items() if c in j["ap"]}))
        for v in ("ps", "cvps"):
            if v in j:
                k = j[v]
                rows.append(dict(base, source=f"none+{v}", rare_f1=k["rare_macro_f1"], rare_ap=j["rare_macro_ap"],
                                 fp10k=k["fp_benign_per_10k"], rare_prec=k["rare_macro_precision"], rare_rec=k["rare_macro_recall"],
                                 **{f"F1[{c}]": k["per_class"][c]["f1"] for c in j["ap"]}))
    return pd.DataFrame(rows)


def part_a():
    tabs = []
    for ds in BASES:
        d = load_proba(ds)
        if d.empty:
            continue
        for (det,), g in d.groupby(["detector"]):
            piv = g.pivot_table(index="seed", columns="source", values="rare_f1")
            for src, h in g.groupby("source"):
                r = dict(dataset=ds, detector=det, source=src, n_seeds=len(h), rare_f1=h.rare_f1.mean(), sd=h.rare_f1.std(ddof=0),
                         rare_ap=h.rare_ap.mean(), rare_prec=h.rare_prec.mean(), rare_rec=h.rare_rec.mean(), fp10k=h.fp10k.mean())
                for ref in ("none", "ros"):
                    if ref in piv and src in piv and src != ref:
                        both = piv[[src, ref]].dropna()
                        r[f"d_vs_{ref}"] = float((both[src] - both[ref]).mean())
                        r[f"p_vs_{ref}"] = wil(both[src], both[ref])
                tabs.append(r)
    if not tabs:
        return
    t = pd.DataFrame(tabs)
    t.to_csv(OUT / "proba_main.csv", index=False)
    md.append("## A. 原划分：稀有类 macro-F1 / PR-AUC / 每万条良性流误报（均值）\n")
    for (ds, det), g in t.groupby(["dataset", "detector"]):
        md.extend([f"### {ds} · {det}", g.drop(columns=["dataset", "detector"]).sort_values("rare_f1", ascending=False).round(4).to_markdown(index=False), ""])


def part_b():
    rows = []
    for ds in BASES:
        ps = [load_proba(p.name).assign(rep=int(p.name.rsplit("_r", 1)[1]))
              for p in sorted(RESULTS.glob(f"{ds}_r*")) if (p / "augment_proba").exists()]
        ps = [p for p in ps if not p.empty]
        if not ps:
            continue
        d = pd.concat(ps)
        fcols = [c for c in d.columns if c.startswith("F1[")]
        for det, g in d.groupby("detector"):
            rep = g.groupby(["rep", "source"])[["rare_f1", "rare_ap", "fp10k"] + fcols].mean().reset_index()
            piv = rep.pivot(index="rep", columns="source", values="rare_f1")
            for src, h in rep.groupby("source"):
                r = dict(dataset=ds, detector=det, source=src, n_reps=len(h), rare_f1=h.rare_f1.mean(), sd_across_splits=h.rare_f1.std(ddof=0),
                         rare_ap=h.rare_ap.mean(), fp10k=h.fp10k.mean())
                for ref in ("none", "ros"):
                    if ref in piv and src != ref:
                        both = piv[[src, ref]].dropna()
                        r[f"d_vs_{ref}"] = float((both[src] - both[ref]).mean())
                        r[f"wins_vs_{ref}"] = f"{int((both[src] > both[ref]).sum())}/{len(both)}"
                r.update({c: h[c].mean() for c in fcols if h[c].notna().any()})
                rows.append(r)
    if not rows:
        return
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "resplit.csv", index=False)
    md.append("## B. 重复划分（每份划分先对种子取均值；sd 为划分间标准差；wins = 该来源高于参照的划分份数）\n")
    for (ds, det), g in t.groupby(["dataset", "detector"]):
        md.extend([f"### {ds} · {det}", g.drop(columns=["dataset", "detector"]).dropna(axis=1, how="all").sort_values("rare_f1", ascending=False).round(4).to_markdown(index=False), ""])


def part_c():
    """受控实验：controlled2 有先验匹配（matched，主结果）与先验不一致（unmatched）；旧版 controlled/ 只有 unmatched"""
    for sub, mode, title in (("controlled2", "matched", "先验匹配（主结果：测试集中该类也按同比例下采样，与真实稀有类设定一致）"),
                             ("controlled2", "unmatched", "先验不一致（测试集保持原样：对应部署时攻击比例高于训练时）"),
                             ("controlled", "unmatched", "旧版（作业 98005，先验不一致）")):
        rows, ref = [], {}
        for ds in BASES:
            for f in (RESULTS / ds / sub).glob("*.json"):
                j = json.load(open(f, encoding="utf-8"))
                if "__full__" in f.name:
                    ref[(ds, j["class"], j["detector"], j["seed"])] = j["none"]["f1"]
                    continue
                r = dict(dataset=ds, cls=j["class"], n=j["n"], detector=j["detector"], seed=j["seed"], purity=j["purity"])
                for m in ("none", "ps", "cw", "ros", "smote"):
                    k = j[m]["matched"] if mode == "matched" else j[m]
                    r[f"f1_{m}"], r[f"ap_{m}"] = k["f1"], k["ap"]
                rows.append(r)
        if not rows or (sub == "controlled" and (RESULTS / BASES[0] / "controlled2").exists()):
            continue
        d = pd.DataFrame(rows)
        d["f1_full"] = [ref.get((r.dataset, r.cls, r.detector, r.seed), np.nan) for r in d.itertuples()]
        g = d.groupby(["dataset", "cls", "n", "detector"]).mean(numeric_only=True).drop(columns="seed").reset_index()
        for m in ("ps", "cw", "ros", "smote"):
            g[f"d_{m}"] = g[f"f1_{m}"] - g["f1_none"]
            g[f"dap_{m}"] = g[f"ap_{m}"] - g["ap_none"]
        g["headroom"] = g["f1_full"] - g["f1_none"]
        g.to_csv(OUT / f"{sub}_{mode}.csv", index=False)
        g["purity_bin"] = pd.cut(g.purity, [-0.01, 0.5, 0.8, 1.0], labels=["重叠 <0.5", "中 0.5–0.8", "可分 >0.8"])
        md.append(f"## C. 受控实验 · {title}\n")
        md.append("相对不增强的稀有类 F1 变化（d_*）与 PR-AUC 变化（dap_*），按下采样后纯度与规模 n 分组取均值；count = 类 × 检测器\n")
        t = g.groupby(["purity_bin", "n"], observed=True).agg(
            count=("cls", "size"), f1_none=("f1_none", "mean"), d_ps=("d_ps", "mean"), d_cw=("d_cw", "mean"),
            d_ros=("d_ros", "mean"), d_smote=("d_smote", "mean"), dap_ros=("dap_ros", "mean"), dap_smote=("dap_smote", "mean")).reset_index()
        md.extend([t.round(3).to_markdown(index=False), ""])
        from scipy.stats import spearmanr
        cor = []
        for m in ("ps", "cw", "ros", "smote"):
            for x in ("purity", "n"):
                rho, p = spearmanr(g[x], g[f"d_{m}"])
                cor.append(dict(method=m, x=x, spearman_rho=rho, p=p, points=len(g)))
        md.extend(["Spearman 相关（每个点 = 数据集 × 类 × n × 检测器）：", "", pd.DataFrame(cor).round(4).to_markdown(index=False), ""])
        md.extend(["各方法增益之间的 Spearman 相关：", "", g[["d_ps", "d_cw", "d_ros", "d_smote"]].corr(method="spearman").round(3).to_markdown(), ""])


def part_d():
    rows = []
    for ds in BASES:
        for f in (RESULTS / ds / "fullfeat").glob("*__s*.json"):
            j = json.load(open(f, encoding="utf-8"))
            rows.append(dict(dataset=ds, space=j["space"], n_features=j["n_features"], source=j["source"], detector=j["detector"],
                             seed=j["seed"], rare_f1=j["rare_macro_f1"], rare_ap=j["rare_macro_ap"],
                             **{f"F1[{c}]": j["per_class"][c]["f1"] for c in j["ap"]}))
    pur = [pd.read_csv(p) for p in RESULTS.glob("*/fullfeat/purity.csv") if re.fullmatch("|".join(BASES), p.parts[-3])]
    if pur:
        p = pd.concat(pur).pivot_table(index=["dataset", "class", "n_train"], columns="space", values="purity_mean").reset_index()
        p.to_csv(OUT / "fullfeat_purity.csv", index=False)
        md.extend(["## D. 完整特征：真实稀有类的 kNN 同类纯度（d10 = 10 维，full = 全部流特征）\n", p.round(3).to_markdown(index=False), ""])
    if rows:
        t = pd.DataFrame(rows).groupby(["dataset", "detector", "space", "source"]).mean(numeric_only=True).drop(columns="seed").reset_index()
        t.to_csv(OUT / "fullfeat_eval.csv", index=False)
        md.extend(["完整特征上的增强基线（稀有类 macro-F1 / PR-AUC，种子均值）：", "", t.dropna(axis=1, how="all").round(4).to_markdown(index=False), ""])


for part in (part_a, part_b, part_c, part_d):
    part()
(OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md)[:6000])
