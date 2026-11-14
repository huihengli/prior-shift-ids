"""[CPU · 轻量] 汇总 + 统计检验 + 生成论文表格（Markdown 与 LaTeX）
  - 检测器对比：均值±标准差（10 种子）
  - 调参方法对比：最佳 CV、耗时、收敛曲线数据
  - 增强方法对比：均值±标准差、相对 'none' 与最强传统基线的 Wilcoxon 符号秩检验（按种子配对）
    Holm 校正只在事先限定的主比较族内做（config: experiment.stats_family，可按数据集覆盖，或 --family 指定）；
    族外来源（消融、中间轮次等）只给未校正 p，作描述性报告
  - 稀有类 F1 的 bootstrap 95% CI（测试集重采样，合并 10 个种子的预测）
用法：python cpu/05_stats.py [--family none ros smote llm_sft_filt ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import exp_param, load_cfg, res_dir

METRICS = ["macro_f1", "rare_macro_f1", "rare_macro_recall", "weighted_f1", "accuracy"]


def load_dir(d: Path) -> pd.DataFrame:
    rows = []
    for f in d.glob("*.json"):
        if f.name == "meta.json":
            continue
        r = json.load(open(f, encoding="utf-8"))
        per = r.pop("per_class", {})
        for c, v in per.items():
            r[f"f1::{c}"] = v["f1"]
        r.pop("history", None)
        rows.append(r)
    return pd.DataFrame(rows)


def ms(x):
    return f"{np.mean(x):.4f} ± {np.std(x, ddof=1) if len(x) > 1 else 0:.4f}"


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0
    for i, k in enumerate(order):
        run = max(run, (len(ps) - i) * ps[k])
        adj[k] = min(1, run)
    return adj


def wilcox(a, b):
    from scipy.stats import wilcoxon
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b, zero_method="zsplit").pvalue)
    except ValueError:
        return 1.0


def bootstrap_rare_f1(aug_dir: Path, tag_prefix: str, meta, n_boot=1000, seed=0):
    """合并所有种子的预测，对测试集样本做 bootstrap，返回稀有类 macro-F1 的 95% CI"""
    from sklearn.metrics import f1_score
    y = np.load(aug_dir / "y_test.npy")
    preds = [np.load(f) for f in sorted((aug_dir / "preds").glob(f"{tag_prefix}__s*.npy"))]
    if not preds:
        return (np.nan, np.nan)
    rare_idx = [meta["labels"].index(c) for c in meta["rare"]]
    sub = np.isin(y, rare_idx) | np.isin(np.stack(preds), rare_idx).any(0)
    yi, P = y[sub], np.stack(preds)[:, sub]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        b = rng.integers(0, len(yi), len(yi))
        vals.append(np.mean([f1_score(yi[b], p[b], labels=rare_idx, average="macro", zero_division=0) for p in P]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", nargs="+", default=None, help="主比较族（Holm 校正范围），覆盖配置")
    a = ap.parse_args()
    cfg = load_cfg()
    try:
        family = a.family or list(exp_param(cfg, "stats_family"))
    except KeyError:
        family = None  # 未配置：退回为对全部来源校正
    base = res_dir(cfg)
    out = res_dir(cfg, "tables")
    md = []

    # ---------- 检测器
    d = base / "detectors/final"
    if d.exists() and any(d.glob("*.json")):
        df = load_dir(d)
        g = df.groupby("model")
        t = pd.DataFrame({m: g[m].apply(ms) for m in ["macro_f1", "rare_macro_f1", "weighted_f1"]})
        t["train_time_s"] = g["train_time_s"].apply(ms)
        t["latency_batch_us"] = g["latency_batch_us_per_flow"].mean().round(2)
        t["model_size_mb"] = g["model_size_mb"].mean().round(2)
        t = t.loc[g["macro_f1"].mean().sort_values(ascending=False).index]
        ref = df[df.model == "SAORF"].sort_values("seed")["macro_f1"].to_numpy()
        if len(ref):
            t["p_vs_SAORF"] = [np.nan if m == "SAORF" else
                               wilcox(ref, df[df.model == m].sort_values("seed")["macro_f1"].to_numpy()[: len(ref)])
                               for m in t.index]
        t.to_csv(out / "detectors.csv")
        md += [f"## 检测器对比（{cfg['_ds']} 测试集，10 种子）", t.to_markdown(), ""]
        (out / "detectors.tex").write_text(t.to_latex(escape=True), encoding="utf-8")

    # ---------- 调参方法
    d = base / "detectors/tune"
    if d.exists() and any(d.glob("*.json")):
        rs = [json.load(open(f, encoding="utf-8")) for f in d.glob("*.json")]
        df = pd.DataFrame([{k: r[k] for k in ("method", "seed", "best_cv", "time_s")} for r in rs])
        t = df.groupby("method").agg(best_cv=("best_cv", ms), time_s=("time_s", ms))
        t.to_csv(out / "tuners.csv")
        md += ["## 调参方法对比（相同评估预算）", t.to_markdown(), ""]
        curves = {f"{r['method']}_s{r['seed']}": [h[2] for h in r["history"]] for r in rs}
        json.dump(curves, open(out / "tuner_curves.json", "w"))

    # ---------- 增强方法
    d = base / "augment"
    if d.exists() and any(d.glob("*.json")):
        meta = json.load(open(d / "meta.json", encoding="utf-8"))
        df = load_dir(d)
        for det, dd in df.groupby("detector"):
            g = dd.groupby("source")
            t = pd.DataFrame({m: g[m].apply(ms) for m in ["rare_macro_f1", "rare_macro_recall", "macro_f1"]})
            for c in meta["rare"]:
                t[f"F1 {c}"] = g[f"f1::{c}"].mean().round(4)
            t = t.loc[g["rare_macro_f1"].mean().sort_values(ascending=False).index]
            piv = dd.pivot_table(index="seed", columns="source", values="rare_macro_f1")
            classic = [s for s in piv.columns if s in ("ros", "smote", "bsmote", "adasyn", "ctgan", "tvae")]
            best_classic = piv[classic].mean().idxmax() if classic else None
            for refname, ref in [("none", "none"), ("best_classic", best_classic)]:
                if ref is None or ref not in piv:
                    continue
                ps = [np.nan if s == ref else wilcox(piv[s].dropna(), piv[ref].loc[piv[s].dropna().index])
                      for s in t.index]
                ps = np.asarray(ps, float)
                fam = np.array([family is None or s in family for s in t.index])
                mask = ~np.isnan(ps) & fam
                adj = np.full(len(ps), np.nan)
                adj[mask] = holm(ps[mask])
                t[f"p_holm_vs_{refname}({ref})"] = adj
                t[f"p_raw_vs_{refname}"] = ps
            t.insert(0, "main_family", ["✓" if (family is None or s in family) else "" for s in t.index])
            t["rareF1_95CI"] = [
                "[{:.3f}, {:.3f}]".format(*bootstrap_rare_f1(d, f"{s}__{det}__t{int(dd.target.iloc[0])}", meta))
                for s in t.index]
            t.to_csv(out / f"augment_{det}.csv")
            (out / f"augment_{det}.tex").write_text(t.to_latex(escape=True), encoding="utf-8")
            md += [f"## 增强方法对比 · 检测器 {det}", t.to_markdown(), ""]

    # ---------- 增强规模消融 / 噪声 / 保真度
    for sub, title in [("augment_sweep", "增强规模消融"), ("noise", "噪声鲁棒性")]:
        d = base / sub
        if d.exists() and any(d.glob("*.json")):
            df = load_dir(d)
            key = ["source", "target"] if sub == "augment_sweep" else ["model", "noise_type", "level"]
            col = "rare_macro_f1" if sub == "augment_sweep" else "macro_f1"
            t = df.groupby(key)[col].mean().unstack(key[-1]).round(4)
            t.to_csv(out / f"{sub}.csv")
            md += [f"## {title}（{col} 均值）", t.to_markdown(), ""]
    f = base / "fidelity/fidelity_summary.csv"
    if f.exists():
        md += ["## 合成数据质量", pd.read_csv(f, index_col=0).round(3).to_markdown(), ""]

    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
