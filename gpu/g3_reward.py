"""[CPU · 中等，在实验机上跑即可] 检测器反馈奖励 → 样本筛选 + DPO 偏好对（闭环的“反馈”一步）

对每条生成样本 x（目标类别 c）：
  R_valid = 1[解析成功 且 满足物理约束]                     （不满足则总奖励直接为 0）
  R_fid   = exp(-max(0, d_k(x, real_c)/τ_c - 1)) × (0.5 + 0.5·purity_k(x, c))
            d_k: 到 c 类真实样本的 k 近邻平均距离；τ_c: 真实样本自身 k 近邻距离的 95% 分位
            purity_k: 全体真实训练样本中 x 的 k 近邻属于 c 的比例（防止生成到别的类别区域）
  R_hard  = (1 - p_D(c|x)) × R_fid     D = 当前检测器（真实数据 + 之前各轮已接受样本训练）
            —— 只奖励“真实可信、但当前检测器还没学会”的样本，这就是检测器反馈
  P_mem   = 1[与训练集最近距离 ≤ c 类真实样本最近邻距离的 1% 分位]（抄袭/记忆惩罚）
  R = w_valid·R_valid + w_fid·R_fid + w_hard·R_hard − w_mem·P_mem
接受条件：有效、非抄袭、批内不重复、R ≥ accept_threshold
v2.1 类间重叠约束（reward.overlap_filter，默认开启；依据：v2 评估中 SQLi 合成样本占据 Benign 区域导致大量误报）：
  ov_purity = 10 维特征 log1p 标准化空间中，x 在全部真实训练样本里 overlap_k 个近邻属于 c 的比例
  ov_purity < overlap_tau 的样本：R_hard 置 0（hard_gate，困难度只对“未越界”的样本计分），且不接受
  —— 否则“检测器判错 = 高分”会奖励落在其他类区域的样本，闭环持续制造标签噪声
偏好对：同一 prompt 的 m 条输出中奖励最高者为 chosen、最低者为 rejected（差值 ≥ 0.15）

用法：python gpu/g3_reward.py --gen llm_sft --out llm_sft --prev ""            （第 0 轮）
      python gpu/g3_reward.py --gen llm_loop_r1 --out llm_loop_r1 --prev llm_sft_filt
      消融：加 --ablate hard / fid / mem（对应权重置 0；正向项消融时接受阈值按剩余权重等比缩放；mem 消融时不再过滤抄袭）
            --ablate overlap：关闭类间重叠约束（等同 v2 行为）
输出：data/<ds>/synth/<out>_filt.csv（含 reward/accept 列，03 评估时只用 accept=1 的样本）
      data/<ds>/llm/pairs_<out>.jsonl，data/<ds>/llm/reward_<out>_stats.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (FEATURES, PRIMARY, ds_dir, exp_param, finalize_synth, load_cfg, load_split, make_rf, rare_classes,  # noqa
                    res_dir, synth_dir, validity_mask, xy)


def sa_params(cfg):
    fs = sorted(res_dir(cfg, "detectors/tune").glob("sa_s*.json"))
    return max([json.load(open(f)) for f in fs], key=lambda r: r["best_cv"])["best_params"] if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="g2 输出名（读取 llm/gen_<gen>.jsonl）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prev", nargs="*", default=[], help="之前各轮的 *_filt 合成文件名（用于训练当前检测器）")
    ap.add_argument("--ablate", default=None, choices=[None, "hard", "fid", "mem", "overlap"])
    a = ap.parse_args()
    cfg = load_cfg()
    rc = dict(cfg["llm"]["reward"])
    ov_on = bool(rc.get("overlap_filter", True)) and a.ablate != "overlap"
    hard_gate = ov_on and bool(rc.get("hard_gate", True))
    ov_k, ov_tau = int(rc.get("overlap_k", 10)), float(rc.get("overlap_tau", 0.5))
    if a.ablate and a.ablate != "overlap":
        pos0 = rc["w_valid"] + rc["w_fid"] + rc["w_hard"]
        rc[f"w_{a.ablate}"] = 0.0
        # 去掉某个正向奖励项后，接受阈值按剩余正向权重等比例缩放，否则消融组几乎不可能达到原阈值
        #（v1 中去掉保真项后接受率为 0，比较失去意义）
        rc["accept_threshold"] *= (rc["w_valid"] + rc["w_fid"] + rc["w_hard"]) / pos0
    k = rc["knn_k"]
    train = load_split(cfg, "train")
    rare = rare_classes(cfg, train)

    # ---- 当前检测器 D_r
    parts = [train]
    for p in [p for p in a.prev if p]:
        s = pd.read_csv(synth_dir(cfg) / f"{p}.csv")
        parts.append(s[s["accept"] == 1][FEATURES + ["label"]])
    dtrain = pd.concat(parts, ignore_index=True)
    D = make_rf(sa_params(cfg), 0).fit(*xy(dtrain))
    cls_index = {c: i for i, c in enumerate(D.classes_)}
    print(f"detector trained on {len(dtrain)} rows ({len(dtrain) - len(train)} accepted synthetic)")

    # ---- 参考统计量（只用真实训练集）
    L = np.log1p(train[PRIMARY].to_numpy(float))
    mu, sd = L.mean(0), L.std(0) + 1e-9
    Z = (L - mu) / sd
    ylab = train["label"].to_numpy()
    nn_all = NearestNeighbors(n_neighbors=k).fit(Z)
    # 类间重叠用的空间：10 维特征（含派生特征），与 cpu/08_overlap_filter.py 完全一致
    Lf = np.log1p(train[FEATURES].clip(lower=0).to_numpy(float))
    mu_f, sd_f = Lf.mean(0), Lf.std(0) + 1e-9
    nn_ov = NearestNeighbors(n_neighbors=ov_k).fit((Lf - mu_f) / sd_f)
    per = {}
    for c in rare:
        Zc = Z[ylab == c]
        kk = min(k, len(Zc) - 1)
        nn_c = NearestNeighbors(n_neighbors=kk + 1).fit(Zc)
        dd, _ = nn_c.kneighbors(Zc)
        per[c] = {"nn": nn_c, "k": kk, "tau": np.quantile(dd[:, 1:].mean(1), 0.95) + 1e-9,
                  "mem_thr": np.quantile(dd[:, 1], rc["mem_threshold_quantile"])}

    # ---- 读取生成记录
    recs = [json.loads(l) for l in open(ds_dir(cfg) / "llm" / f"gen_{a.gen}.jsonl", encoding="utf-8")]
    R = np.zeros(len(recs))
    comp = {"valid": np.zeros(len(recs)), "fid": np.zeros(len(recs)), "hard": np.zeros(len(recs)),
            "mem": np.zeros(len(recs))}
    ok_idx = [i for i, r in enumerate(recs) if r["parsed"] is not None]
    if not ok_idx:
        sys.exit(f"gen_{a.gen}.jsonl 中没有可解析的样本，无法打分")
    raw = pd.DataFrame([{**recs[i]["parsed"], "label": recs[i]["label"], "prompt_id": recs[i]["prompt_id"],
                         "sample_id": recs[i]["sample_id"]} for i in ok_idx])
    valid = validity_mask(raw).to_numpy()
    fin = finalize_synth(raw.assign(raw_valid=valid.astype(int)), a.out)  # 修复后（行顺序与 raw 一致）
    assert len(fin) == len(raw), "finalize_synth 改变了行数，样本与奖励会错位"
    Zs = (np.log1p(fin[PRIMARY].to_numpy(float)) - mu) / sd
    Zs = np.nan_to_num(Zs, nan=0.0, posinf=50.0, neginf=-50.0)
    proba = D.predict_proba(fin[FEATURES].to_numpy(float))
    d_glob, i_glob = nn_all.kneighbors(Zs)
    dup = fin.duplicated(subset=PRIMARY + ["label"]).to_numpy()
    Zf = (np.log1p(fin[FEATURES].clip(lower=0).to_numpy(float)) - mu_f) / sd_f
    Zf = np.nan_to_num(Zf, nan=0.0, posinf=50.0, neginf=-50.0)
    _, i_ov = nn_ov.kneighbors(Zf)
    ov_pur = (ylab[i_ov] == fin["label"].to_numpy()[:, None]).mean(1)
    crossing = ov_pur < ov_tau

    for j, i in enumerate(ok_idx):
        c = recs[i]["label"]
        if not valid[j]:
            continue
        P = per[c]
        dk, _ = P["nn"].kneighbors(Zs[j:j + 1], n_neighbors=P["k"])
        purity = np.mean(ylab[i_glob[j]] == c)
        r_fid = float(np.exp(-max(0.0, dk.mean() / P["tau"] - 1.0)) * (0.5 + 0.5 * purity))
        r_hard = (1.0 - proba[j, cls_index[c]]) * r_fid
        if hard_gate and crossing[j]:
            r_hard = 0.0  # 越界样本不计困难度（检测器判错是因为它本来就在别的类区域）
        p_mem = float(d_glob[j, 0] <= P["mem_thr"] or d_glob[j, 0] < 1e-9)
        comp["valid"][i], comp["fid"][i], comp["hard"][i], comp["mem"][i] = 1, r_fid, r_hard, p_mem
        R[i] = rc["w_valid"] + rc["w_fid"] * r_fid + rc["w_hard"] * r_hard - rc["w_mem"] * p_mem

    fin["reward"] = R[ok_idx]
    for kname in ("fid", "hard", "mem"):
        fin[f"r_{kname}"] = comp[kname][ok_idx]
    no_mem = (comp["mem"][ok_idx] == 0) if rc["w_mem"] > 0 else np.ones(len(ok_idx), bool)  # 去掉记忆惩罚的消融：不再过滤近似抄袭
    fin["ov_purity"] = ov_pur
    no_cross = ~crossing if ov_on else np.ones(len(ok_idx), bool)
    acc0 = (valid) & no_mem & (~dup) & (fin["reward"] >= rc["accept_threshold"])
    fin["accept"] = (acc0 & no_cross).astype(int)
    ov_info = {c: {"accepted_before_overlap": int((acc0 & (fin["label"].to_numpy() == c)).sum()),
                   "rejected_by_overlap": int((acc0 & ~no_cross & (fin["label"].to_numpy() == c)).sum()),
                   # 越界比例（有效样本中）；开启 hard_gate 时部分越界样本已因 R_hard=0 低于阈值，不计入上一项
                   "crossing_rate_valid": float(crossing[valid & (fin["label"].to_numpy() == c)].mean())
                   if (valid & (fin["label"].to_numpy() == c)).any() else None,
                   "mean_ov_purity_valid": float(ov_pur[valid & (fin["label"].to_numpy() == c)].mean())
                   if (valid & (fin["label"].to_numpy() == c)).any() else None}
               for c in rare}
    print("[overlap]", "on" if ov_on else "off", f"k={ov_k} tau={ov_tau} hard_gate={hard_gate}",
          json.dumps(ov_info, ensure_ascii=False), flush=True)
    # ---- 锚点均衡筛选：把每条候选样本归到最近的同类真实样本（锚点），每个锚点最多接受 cap 条（奖励高者优先）。
    # 使接受集合在各形态上的比例与真实数据一致（v1 中未微调 LLM 的 XSS 样本 0% 为短流，而真实训练集 33%、测试集 58%）。
    fin["anchor_nn"] = -1
    fin["accept_pre_balance"] = fin["accept"]
    balance = rc.get("anchor_balance", True)
    tmax = max(exp_param(cfg, "aug_target_sweep") + [exp_param(cfg, "aug_target_per_rare_class")])
    bal_info = {}
    for c in rare:
        P = per[c]
        sel = np.where((fin["label"].to_numpy() == c))[0]
        if len(sel) == 0:
            continue
        _, nn1 = P["nn"].kneighbors(Zs[sel], n_neighbors=1)
        fin.iloc[sel, fin.columns.get_loc("anchor_nn")] = nn1[:, 0]
        n_real = int((ylab == c).sum())
        cap = int(np.ceil(max(1, tmax - n_real) / n_real))
        if balance:
            cand = fin.iloc[sel]
            cand = cand[cand["accept"] == 1].sort_values("reward", ascending=False)
            keep = cand.groupby("anchor_nn").head(cap).index
            drop = cand.index.difference(keep)
            fin.loc[drop, "accept"] = 0
        acc = fin.iloc[sel]
        acc = acc[acc["accept"] == 1]
        bal_info[c] = {"n_real": n_real, "cap_per_anchor": cap, "accepted_before": int(fin.iloc[sel]["accept_pre_balance"].sum()),
                       "accepted_after": int(len(acc)), "anchors_covered": int(acc["anchor_nn"].nunique()),
                       "anchor_coverage": float(acc["anchor_nn"].nunique() / n_real)}
    print("[balance]", json.dumps(bal_info, ensure_ascii=False), flush=True)
    fin.to_csv(synth_dir(cfg) / f"{a.out}_filt.csv", index=False)

    # ---- DPO 偏好对
    by_prompt = {}
    for i, r in enumerate(recs):
        by_prompt.setdefault(r["prompt_id"], []).append(i)
    pairs = []
    for pid, idx in by_prompt.items():
        hi, lo = max(idx, key=lambda i: R[i]), min(idx, key=lambda i: R[i])
        if R[hi] - R[lo] >= 0.15:
            pairs.append({"label": recs[hi]["label"], "gap": float(R[hi] - R[lo]), "prompt": recs[hi]["prompt"],
                          "chosen": recs[hi]["completion"], "rejected": recs[lo]["completion"]})
    cap = cfg["llm"]["loop"]["dpo_pairs_per_class"]
    pairs = (pd.DataFrame(pairs).sort_values("gap", ascending=False).groupby("label").head(cap)
             if pairs else pd.DataFrame())
    with open(ds_dir(cfg) / "llm" / f"pairs_{a.out}.jsonl", "w", encoding="utf-8") as f:
        for _, p in pairs.iterrows():
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    st = {"gen": a.gen, "out": a.out, "ablate": a.ablate, "n_records": len(recs), "parse_rate": len(ok_idx) / len(recs),
          "valid_rate": float(comp["valid"].mean()), "mem_rate_of_valid": float(comp["mem"][comp["valid"] == 1].mean()),
          "accept_rate": float(fin["accept"].sum() / len(recs)), "mean_reward": float(R.mean()),
          "accepted_per_class": fin[fin.accept == 1].label.value_counts().to_dict(),
          "n_pairs": len(pairs), "weights": rc, "anchor_balance": bal_info,
          "overlap": {"on": ov_on, "k": ov_k, "tau": ov_tau, "hard_gate": hard_gate, "per_class": ov_info}}
    json.dump(st, open(ds_dir(cfg) / "llm" / f"reward_{a.out}_stats.json", "w"), indent=1)
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
