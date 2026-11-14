"""v2 公共模块：配置、数据集映射、标签规范化、检测器、调参器、评估指标。
CPU / GPU 脚本都会 import 本文件。只依赖 numpy / pandas / pyyaml（检测器相关函数延迟导入 sklearn）。
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


# ----------------------------------------------------------------- config
def load_cfg() -> dict:
    # IDS_CONFIG 可指定另一份配置（实验机 CPU 作业用 config_cpu.yaml，避免覆盖实验机上的 GPU 配置）
    with open(os.environ.get("IDS_CONFIG", ROOT / "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_ds"] = os.environ.get("IDS_DATASET", cfg["dataset"])
    # 重复划分数据集 <base>_r<k>（cpu/11_resplit.py 生成）沿用 <base> 的数据集级配置
    m = re.fullmatch(r"(.+)_r\d+", cfg["_ds"])
    if m and cfg["_ds"] not in cfg["datasets"] and m.group(1) in cfg["datasets"]:
        cfg["datasets"][cfg["_ds"]] = cfg["datasets"][m.group(1)]
    # 第二个 LLM 的敏感性分析：LLM_BASE_MODEL 覆盖 llm.base_model（不改 config.yaml，避免影响 Qwen 的结果复现）
    if os.environ.get("LLM_BASE_MODEL"):
        cfg["llm"]["base_model"] = os.environ["LLM_BASE_MODEL"]
    return cfg


def ds_dir(cfg) -> Path:
    d = DATA / cfg["_ds"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def res_dir(cfg, sub: str = "") -> Path:
    d = RESULTS / (cfg["_ds"] + cfg.get("_tag", "")) / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


FEATURES = ["dst_port", "protocol", "flow_duration", "fwd_pkts", "bwd_pkts",
            "fwd_bytes", "bwd_bytes", "flow_bytes_s", "flow_pkts_s", "pkt_size_avg"]
PRIMARY = ["dst_port", "protocol", "flow_duration", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes"]
DERIVED = ["flow_bytes_s", "flow_pkts_s", "pkt_size_avg"]
INT_COLS = ["dst_port", "protocol", "flow_duration", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """派生特征统一重新计算（保证真实数据与生成数据遵循同一物理定义，约束可精确校验）。"""
    dur_s = np.maximum(df["flow_duration"].astype(float), 1.0) / 1e6
    tot_b = df["fwd_bytes"].astype(float) + df["bwd_bytes"].astype(float)
    tot_p = df["fwd_pkts"].astype(float) + df["bwd_pkts"].astype(float)
    df["flow_bytes_s"] = tot_b / dur_s
    df["flow_pkts_s"] = tot_p / dur_s
    df["pkt_size_avg"] = tot_b / np.maximum(tot_p, 1.0)
    return df


# ------------------------------------------------------ dataset column maps
CSE_MAP = {  # CSE-CIC-IDS2018 CICFlowMeter 列名 -> 统一名
    "Dst Port": "dst_port", "Protocol": "protocol", "Flow Duration": "flow_duration",
    "Tot Fwd Pkts": "fwd_pkts", "Tot Bwd Pkts": "bwd_pkts",
    "TotLen Fwd Pkts": "fwd_bytes", "TotLen Bwd Pkts": "bwd_bytes", "Label": "label",
}
CIC17_MAP = {  # CIC-IDS2017（GeneratedLabelledFlows / Engelen 修正版，列名已 strip）
    "Destination Port": "dst_port", "Protocol": "protocol", "Flow Duration": "flow_duration",
    "Total Fwd Packets": "fwd_pkts", "Total Backward Packets": "bwd_pkts",
    "Total Length of Fwd Packets": "fwd_bytes", "Total Length of Bwd Packets": "bwd_bytes",
    "Label": "label",
}
NF_MAP = {  # NetFlow v2 版本（NF-UNSW-NB15-v2 等），时长为毫秒
    "L4_DST_PORT": "dst_port", "PROTOCOL": "protocol", "FLOW_DURATION_MILLISECONDS": "flow_duration",
    "IN_PKTS": "fwd_pkts", "OUT_PKTS": "bwd_pkts", "IN_BYTES": "fwd_bytes", "OUT_BYTES": "bwd_bytes",
    "Attack": "label",
}


# ------------------------------------------------------- label canonicalise
_CANON = {
    "benign": "Benign", "normal": "Benign",
    "ddosattackhoic": "DDoS-HOIC", "ddosattackshoic": "DDoS-HOIC",
    "ddosattackloicudp": "DDoS-LOIC-UDP", "ddosattacksloicudp": "DDoS-LOIC-UDP",
    "ddosattacksloichttp": "DDoS-LOIC-HTTP", "ddosattackloichttp": "DDoS-LOIC-HTTP",
    "ftpbruteforce": "FTP-BruteForce", "sshbruteforce": "SSH-BruteForce",
    "bruteforceweb": "Web-BruteForce", "bruteforcexss": "Web-XSS", "sqlinjection": "Web-SQLi",
    "infilteration": "Infiltration", "infiltration": "Infiltration",
    "botnet": "Bot", "bot": "Bot",
    # CIC-IDS2017
    "webattackbruteforce": "Web-BruteForce", "webattackxss": "Web-XSS", "webattacksqlinjection": "Web-SQLi",
    "ftppatator": "FTP-BruteForce", "sshpatator": "SSH-BruteForce", "portscan": "PortScan", "ddos": "DDoS",
    "heartbleed": "Heartbleed",
}


def canon_label(x: str) -> str:
    k = re.sub(r"[^a-z0-9]", "", str(x).lower())
    if k in _CANON:
        return _CANON[k]
    if k.startswith("dosattacks"):
        return "DoS-" + str(x).split("-", 1)[-1].strip()
    return str(x).strip()


# ----------------------------------------------------------------- loading
def load_split(cfg, name: str) -> pd.DataFrame:
    """name: train | test | train_pool"""
    return pd.read_csv(ds_dir(cfg) / f"{name}.csv.gz")


def xy(df: pd.DataFrame):
    return df[FEATURES].to_numpy(dtype=np.float64), df["label"].astype(str).to_numpy()


def rare_classes(cfg, train: pd.DataFrame) -> list[str]:
    vc = train["label"].value_counts()
    return sorted(vc[vc < rare_threshold(cfg)].index.tolist())


def exp_param(cfg, key):
    """实验参数：数据集级覆盖（datasets.<ds>.<key>）优先，否则用 experiment.<key>"""
    return cfg["datasets"].get(cfg["_ds"], {}).get(key, cfg["experiment"][key])


def rare_threshold(cfg) -> int:
    """数据集可单独指定稀有类阈值（config.yaml 中 datasets.<ds>.rare_threshold），否则用全局值"""
    return cfg["datasets"].get(cfg["_ds"], {}).get("rare_threshold", cfg["split"]["rare_threshold"])


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)


# ---------------------------------------------------------------- metrics
def evaluate(y_true, y_pred, labels, rare: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per = {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
           for i, lab in enumerate(labels)}
    rare_f1 = [per[c]["f1"] for c in rare if c in per]
    rare_rec = [per[c]["recall"] for c in rare if c in per]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "rare_macro_f1": float(np.mean(rare_f1)) if rare_f1 else float("nan"),
        "rare_macro_recall": float(np.mean(rare_rec)) if rare_rec else float("nan"),
        "per_class": per,
    }


# -------------------------------------------------------------- detectors
def make_rf(params: dict | None = None, seed: int = 0, n_jobs: int = -1):
    from sklearn.ensemble import RandomForestClassifier
    p = dict(n_estimators=100, n_jobs=n_jobs, random_state=seed)
    if params:
        p.update(params)
    return RandomForestClassifier(**p)


def make_detector(name: str, seed: int = 0, params: dict | None = None, n_jobs: int = -1):
    """统一的检测器工厂。xgboost / lightgbm 需要标签编码，由 LabelEncodedModel 包装。"""
    if name == "rf":
        return make_rf(params, seed, n_jobs)
    if name == "xgb":
        from xgboost import XGBClassifier
        return LabelEncodedModel(XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                               tree_method="hist", n_jobs=n_jobs, random_state=seed))
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LabelEncodedModel(LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05,
                                                min_child_samples=5, reg_lambda=1.0,
                                                # 默认 min_sum_hessian=1e-3 在极稀有类上会数值发散
                                                min_sum_hessian_in_leaf=1e-2,
                                                n_jobs=n_jobs, random_state=seed, verbose=-1))
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import QuantileTransformer
        return make_pipeline(QuantileTransformer(n_quantiles=200, random_state=seed),
                             MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200,
                                           early_stopping=True, random_state=seed))
    raise ValueError(name)


class LabelEncodedModel:
    def __init__(self, m):
        self.m = m

    def fit(self, X, y, **kw):
        from sklearn.preprocessing import LabelEncoder
        self.le = LabelEncoder().fit(y)
        self.m.fit(X, self.le.transform(y), **kw)
        return self

    def predict(self, X):
        return self.le.inverse_transform(np.asarray(self.m.predict(X)).astype(int).ravel())

    def predict_proba(self, X):
        return self.m.predict_proba(X)

    @property
    def classes_(self):
        return self.le.classes_


# ---------------------------------------------------- hyper-parameter search
INT_PARAMS = {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"}


def _decode(u: np.ndarray, space: dict) -> dict:
    """u ∈ [0,1]^d -> 超参数字典"""
    out = {}
    for ui, (k, (lo, hi)) in zip(u, space.items()):
        v = lo + float(np.clip(ui, 0, 1)) * (hi - lo)
        out[k] = int(round(v)) if k in INT_PARAMS else round(v, 4)
    return out


def cv_score(params, X, y, cfg, seed) -> float:
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    sa = cfg["detector"]["sa"]
    m = make_rf(params, seed, cfg["experiment"]["n_jobs"])
    cv = StratifiedKFold(sa["cv_folds"], shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(m, X, y, cv=cv, scoring=sa["scoring"])))


def tune_rf(method: str, X, y, cfg, seed: int = 0, budget: int | None = None):
    """所有调参方法使用**相同的评估预算**（budget 次 CV 评估），保证公平。
    method: sa | random | grid | optuna | ga
    返回 (best_params, best_score, history[list of (iter, score, best_so_far)])"""
    space = {k: v for k, v in cfg["detector"]["space"].items()}
    sa = cfg["detector"]["sa"]
    budget = budget or sa["maxiter"]
    rng = np.random.default_rng(seed)
    d = len(space)
    hist, best = [], (None, -1.0)

    def ev(u):
        nonlocal best
        p = _decode(u, space)
        s = cv_score(p, X, y, cfg, seed)
        if s > best[1]:
            best = (p, s)
        hist.append((len(hist), s, best[1]))
        return s

    if method == "sa":  # 模拟退火（Metropolis 准则，几何降温，邻域步长随温度收缩）
        T = sa["T0"]
        cur = rng.uniform(0, 1, d)
        cur_s = ev(cur)
        while len(hist) < budget:
            step = 0.25 * max(T / sa["T0"], 0.1)
            cand = np.clip(cur + rng.normal(0, step, d), 0, 1)
            s = ev(cand)
            delta = s - cur_s
            # 分数在 [0,1]，用 100*delta 放大到合适量级
            if delta >= 0 or rng.random() < math.exp(100 * delta / max(T, 1e-8)):
                cur, cur_s = cand, s
            T *= sa["alpha"]
    elif method == "random":
        while len(hist) < budget:
            ev(rng.uniform(0, 1, d))
    elif method == "grid":
        per = max(2, int(round(budget ** (1 / d))))
        grid = np.array(np.meshgrid(*[np.linspace(0, 1, per)] * d)).reshape(d, -1).T
        idx = rng.permutation(len(grid))[:budget]
        for g in grid[idx]:
            ev(g)
    elif method == "optuna":
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        st.optimize(lambda t: ev(np.array([t.suggest_float(k, 0, 1) for k in space])), n_trials=budget)
    elif method == "ga":  # 简单遗传算法：锦标赛选择 + 均匀交叉 + 高斯变异
        pop_n = 8
        pop = rng.uniform(0, 1, (pop_n, d))
        fit = np.array([ev(p) for p in pop])
        while len(hist) < budget:
            kids = []
            for _ in range(pop_n):
                a, b = rng.choice(pop_n, 2, replace=False), rng.choice(pop_n, 2, replace=False)
                pa, pb = pop[a[np.argmax(fit[a])]], pop[b[np.argmax(fit[b])]]
                mask = rng.random(d) < 0.5
                kids.append(np.clip(np.where(mask, pa, pb) + rng.normal(0, 0.1, d), 0, 1))
            kids = np.array(kids)[: budget - len(hist)]
            kf = np.array([ev(k) for k in kids])
            allp, allf = np.vstack([pop, kids]), np.concatenate([fit, kf])
            keep = np.argsort(-allf)[:pop_n]
            pop, fit = allp[keep], allf[keep]
    else:
        raise ValueError(method)
    return best[0], best[1], hist


def stratified_cap(df: pd.DataFrame, n: int, seed: int, label="label", caps: dict | None = None) -> pd.DataFrame:
    """每类最多抽 n 条（caps 可对个别类单独指定上限）"""
    idx = []
    for lab, g in df.groupby(label):
        k = (caps or {}).get(lab, n)
        idx.extend(g.sample(min(len(g), k), random_state=seed).index)
    return df.loc[idx].reset_index(drop=True)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t


# ------------------------------------------------- synthetic post-processing
VALID_PROTOS = np.array([0, 1, 6, 17])


def validity_mask(df: pd.DataFrame) -> pd.Series:
    """硬性物理约束（对生成器的原始输出做校验，修复前）：
    端口 0..65535；协议 ∈ {0,1,6,17}；包数≥0 且总包数≥1；字节数≥0；时长≥0；
    每包平均载荷 ≤ 65535（单向），协议 0 时不应有端口；UDP 不应出现 0 字节双向流"""
    ok = df[PRIMARY].notna().all(axis=1)
    for c in PRIMARY:
        ok &= pd.to_numeric(df[c], errors="coerce").notna()
    d = df[PRIMARY].apply(pd.to_numeric, errors="coerce")
    ok &= d["dst_port"].between(0, 65535) & d["protocol"].isin(VALID_PROTOS)
    ok &= (d[["fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "flow_duration"]] >= 0).all(axis=1)
    ok &= (d["fwd_pkts"] + d["bwd_pkts"]) >= 1
    ok &= d["fwd_bytes"] <= d["fwd_pkts"].clip(lower=1) * 65535
    ok &= d["bwd_bytes"] <= d["bwd_pkts"].clip(lower=0) * 65535
    ok &= ~((d["protocol"] == 0) & (d["dst_port"] > 0))
    return ok.fillna(False)


def finalize_synth(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """所有生成方法共用的后处理：数值修复 + 派生特征重算。保证各方法在同一规则下比较。"""
    d = df.copy()
    for c in PRIMARY:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # 不删行（保持与输入逐行对齐，g3 依赖这一点）；非有限值置 0；上限 1e12 防止转 int64 溢出成负数
    #（v2 消融中模型输出过 1e30 级数值 → int64 溢出 → log1p 得 NaN → 打分崩溃）
    d[PRIMARY] = d[PRIMARY].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0, upper=1e12).round()
    d["dst_port"] = d["dst_port"].clip(upper=65535)
    d["protocol"] = VALID_PROTOS[np.abs(d["protocol"].to_numpy()[:, None] - VALID_PROTOS[None]).argmin(1)]
    zero = (d["fwd_pkts"] + d["bwd_pkts"]) < 1
    d.loc[zero, "fwd_pkts"] = 1
    d[PRIMARY] = d[PRIMARY].astype(np.int64)
    d = add_derived(d)
    d["method"] = method
    keep = FEATURES + ["label", "method"] + [c for c in ("reward", "accept", "r_fid", "r_hard", "r_mem", "round", "raw_valid", "prompt_id", "sample_id") if c in d]
    return d[keep].reset_index(drop=True)


def synth_dir(cfg) -> Path:
    d = ds_dir(cfg) / "synth"
    d.mkdir(parents=True, exist_ok=True)
    return d
