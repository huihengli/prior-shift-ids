#!/usr/bin/env bash
# =====================================================================
# 实验机 CPU 节点：投稿前补充实验（2026-11-11）
#   gen        CTGAN / TVAE 生成式基线（三个数据集）+ 重复划分数据集的 ROS / SMOTE 合成池
#   eval       12_eval_proba：全部来源 × 4 检测器 × 10 种子，补 PR-AUC、良性误报率、先验平移基线（ps / cvps）
#   resplit    12_eval_proba 在重复划分 <ds>_r1..r5 上：none / cw / ros / smote（及 GPU 回传后的 LLM 来源）× 4 检测器 × 5 种子
#   controlled 14_controlled：把样本充足的攻击类下采样成稀有类的受控实验
#   fullfeat   13b_fullfeat_eval：完整特征空间上的 none / cw / ros / smote
# 用法（在 v2/ 目录下）：
#   tar xzf v3_cluster_inputs.tgz          # 覆盖 common.py、cpu/、config_cpu.yaml，并补充数据
#   sbatch cpu/run_cpu_cluster_v3.sh       # 或：bash cpu/run_cpu_cluster_v3.sh 2>&1 | tee cpu_v3.log
# 可选环境变量：
#   STAGES="gen eval resplit controlled fullfeat"   只跑其中几段
#   DRYRUN=1       只打印将要运行的任务，不运行
#   THREADS=8      每个任务的线程数（并行任务数 = CPU 数 / THREADS）
#   DATASETS="cse2018 nf_unsw_v3 cic2017"    REPS="1 2 3 4 5"
# 断点：已存在的结果文件会跳过；中断后重新提交即可。GPU 结果回传后再提交一次（STAGES="eval resplit"）即可补上新来源。
# 输出：cpu_v3_results_<作业号>.tgz，拷回本机 v2/ 目录
# =====================================================================
#SBATCH --job-name=ids_cpu_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=joboutput_%j.out
#SBATCH --error=joboutput_%j.err
##SBATCH --partition=请改成CPU分区名   # 去掉一个 # 并填写分区名（若集群要求）

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f common.py ] && [ -f cpu/12_eval_proba.py ] || { echo "!! 请在 v2/ 目录下提交，并先解压 v3_cluster_inputs.tgz"; exit 1; }

# 虚拟环境：依次查找 $VENV、v2/venvs/idsllm、~/venvs/idsllm
for v in "${VENV:-}" "$PWD/venvs/idsllm" "$HOME/venvs/idsllm"; do
  if [ -n "$v" ] && [ -f "$v/bin/activate" ]; then source "$v/bin/activate"; echo "使用虚拟环境：$v"; break; fi
done
[ -n "${VIRTUAL_ENV:-}" ] || echo "!! 没找到虚拟环境（v2/venvs/idsllm），将使用系统 python3：$(command -v python3)"
PY=${PYTHON:-python3}
STAGES=${STAGES:-"gen eval resplit controlled fullfeat"}
need="numpy pandas yaml scipy sklearn xgboost lightgbm imblearn"
miss=$("$PY" - $need <<'PYEOF'
import importlib.util, sys
pip = {"yaml": "pyyaml", "sklearn": "scikit-learn", "imblearn": "imbalanced-learn"}
print(" ".join(pip.get(m, m) for m in sys.argv[1:] if importlib.util.find_spec(m) is None))
PYEOF
)
[ -n "$miss" ] && { echo "安装缺少的包：$miss"; "$PY" -m pip install -q $miss || echo "!! 安装失败：$miss"; }
# CTGAN / TVAE 需要 sdv：装在单独的 venv（v2/venvs/idssdv）里，避免它升级 numpy / torch 影响 GPU 实验环境
SDV_VENV=${SDV_VENV:-$PWD/venvs/idssdv}
PY_SDV=${PY_SDV:-$SDV_VENV/bin/python}
SDV_OK=1
if [[ " $STAGES " == *" gen "* ]] && [ -z "${DRYRUN:-}" ]; then
  if ! "$PY_SDV" -c "import sdv, yaml, sklearn, imblearn" 2>/dev/null; then
    echo "准备 sdv 环境：$SDV_VENV"
    [ -x "$PY_SDV" ] || "$PY" -m venv "$SDV_VENV"
    "$PY_SDV" -m pip install -q --upgrade pip > logs_sdv_install.txt 2>&1
    "$PY_SDV" -m pip install sdv pyyaml scikit-learn imbalanced-learn scipy >> logs_sdv_install.txt 2>&1
  fi
  if "$PY_SDV" -c "import sdv, yaml, sklearn, imblearn; print('sdv', sdv.__version__)"; then :; else
    SDV_OK=0
    echo "!! sdv 环境不可用（安装日志 logs_sdv_install.txt 最后 15 行如下）：CTGAN / TVAE 本次跳过，其余阶段不受影响"
    tail -15 logs_sdv_install.txt | sed 's/^/      /'
  fi
fi
"$PY" -c "import sklearn, xgboost, lightgbm, imblearn; print('sklearn', sklearn.__version__, 'xgboost', xgboost.__version__, 'lightgbm', lightgbm.__version__, 'imblearn', imblearn.__version__)"

export IDS_CONFIG=$PWD/config_cpu.yaml
NCPU=${SLURM_CPUS_PER_TASK:-$(nproc)}
echo "分配到的 CPU：SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-未设置}  nproc=$(nproc)"
if [ -z "${DRYRUN:-}" ] && [ "$NCPU" -lt 16 ] && [ -z "${ALLOW_FEW_CPUS:-}" ]; then
  echo "!! 只分配到 $NCPU 个 CPU，400 多个评估任务会跑好几天。请用 sbatch 提交本脚本（头部已写 --cpus-per-task=128），"
  echo "   或在你自己的提交脚本里申请 --cpus-per-task=128。确实要在少量 CPU 上跑请设 ALLOW_FEW_CPUS=1。已退出。"
  exit 1
fi
THREADS=${THREADS:-8}
P=$(( NCPU / THREADS )); [ "$P" -ge 1 ] || P=1
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DATASETS=${DATASETS:-"cse2018 nf_unsw_v3 cic2017"}
REPS=${REPS:-"1 2 3 4 5"}
DETS=${DETS:-"SAORF RF-grid LightGBM XGBoost"}
L=logs/v3; mkdir -p $L
echo "CPU=$NCPU 每任务线程=$THREADS 并行=$P 阶段=$STAGES"; date

# 每行一个任务："日志名|命令"
run_line() {
  log=${1%%|*}; cmd=${1#*|}
  if bash -c "$cmd" > "$L/$log.log" 2>&1; then echo "OK   $log  $(date +%T)"; else echo "FAIL $log  （见 $L/$log.log），最后 6 行："; grep -v "it/s\]" "$L/$log.log" | tail -6 | sed 's/^/      /'; fi
}
export -f run_line; export L PY THREADS
run_all() {
  if [ -n "${DRYRUN:-}" ]; then cut -d'|' -f2- "$1"; return; fi   # DRYRUN=1：只打印任务，不运行
  [ -s "$1" ] && xargs -P "${2:-$P}" -d '\n' -I{} bash -c 'run_line "$1"' _ {} < "$1" | tee -a $L/progress.txt
}
has() { [[ " $STAGES " == *" $1 "* ]]; }

# 各数据集参与评估的来源（文件不存在的自动跳过；ctgan / tvae 由 gen 阶段生成，great 由 GPU 生成）
sources() {
  case $1 in
    cse2018) echo "none cw ros smote bsmote adasyn ctgan tvae great v2_llm_base_filt v2_llm_sft_filt v2_llm_sft_filt_pure v2_llm_sft_relov_filt v2r_llm_loop_r3_filt_pure" ;;
    *_r*)    echo "none cw ros smote llm_sft_filt llm_sft_relov_filt llm_sft_noneov_filt" ;;
    *)       echo "none cw ros smote bsmote adasyn ctgan tvae great llm_base_filt llm_sft_filt llm_sft_relov_filt llm_sft_noneov_filt" ;;
  esac
}
eval_tasks() {  # $1 数据集  $2 额外参数
  for s in $(sources "$1"); do
    [[ $s == none || $s == cw || -f data/$1/synth/$s.csv ]] || { echo "$1 $s" >> $L/skipped.txt; continue; }
    for det in $DETS; do echo "eval__$1__${det}__$s|IDS_DATASET=$1 $PY cpu/12_eval_proba.py --sources $s --detectors $det --n-jobs $THREADS $2"; done
  done
}

: > $L/progress.txt; : > $L/skipped.txt
# ---------------- 阶段 A：生成（评估依赖它，先跑完）
T=$L/tasks_gen.txt; : > $T
if has gen; then
  for ds in $DATASETS; do for m in ctgan tvae; do
    [ $SDV_OK = 1 ] && [ ! -f data/$ds/synth/$m.csv ] && echo "gen__${ds}__$m|IDS_DATASET=$ds $PY_SDV cpu/02_classic_augment.py --methods $m" >> $T
  done; done
fi
if has gen || has resplit; then
  for ds in $DATASETS; do for r in $REPS; do
    [ -f data/${ds}_r$r/train.csv.gz ] || { echo "!! 缺少 data/${ds}_r$r（先在本机运行 cpu/11_resplit.py 并打包）"; continue; }
    [ -f data/${ds}_r$r/synth/smote.csv ] || echo "gen__${ds}_r${r}__classic|IDS_DATASET=${ds}_r$r $PY cpu/02_classic_augment.py --methods ros smote" >> $T
  done; done
fi
echo "生成任务：$(wc -l < $T)"; run_all $T

# ---------------- 阶段 B：评估
T=$L/tasks_eval.txt; : > $T
if has eval; then for ds in $DATASETS; do eval_tasks "$ds" "" >> $T; done; fi
if has resplit; then for ds in $DATASETS; do for r in $REPS; do
  [ -f data/${ds}_r$r/train.csv.gz ] && eval_tasks "${ds}_r$r" "--seeds 5 --no-cv" >> $T
done; done; fi
if has controlled; then for ds in $DATASETS; do
  IDS_DATASET=$ds $PY cpu/14_controlled.py --list | while IFS= read -r c; do
    for det in SAORF LightGBM; do
      echo "ctrl__${ds}__${det}__${c// /-}|IDS_DATASET=$ds $PY cpu/14_controlled.py --classes \"$c\" --detectors $det --n-jobs $THREADS" >> $T
    done; done
done; fi
if has fullfeat; then for ds in $DATASETS; do
  [ -f data/$ds/fullfeat/full.csv.gz ] || { echo "!! 缺少 data/$ds/fullfeat/full.csv.gz"; continue; }
  for det in SAORF LightGBM XGBoost; do
    echo "full__${ds}__$det|IDS_DATASET=$ds $PY cpu/13b_fullfeat_eval.py --part eval --detectors $det --n-jobs $THREADS" >> $T
  done; done; fi
# RF 类最慢，先排
{ grep -E '__(SAORF|RF-grid)' $T; grep -vE '__(SAORF|RF-grid)' $T; } > $T.sorted
[ -s $L/skipped.txt ] && echo "尚无合成文件、本次未评估的来源 $(wc -l < $L/skipped.txt) 个（正常：GReaT 和重复划分上的 LLM 样本要等 GPU 结果；CTGAN/TVAE 若生成失败也会出现在这里），清单见 $L/skipped.txt"
echo "评估任务：$(wc -l < $T.sorted)"; run_all $T.sorted
date

[ -n "${DRYRUN:-}" ] && exit 0
# ---------------- 检查与打包
echo "失败任务数：$(grep -c '^FAIL' $L/progress.txt || true)"; grep '^FAIL' $L/progress.txt || true
OUT=cpu_v3_results_${SLURM_JOB_ID:-local}.tgz
{ ls -d results/*/augment_proba results/*/controlled results/*/controlled2 results/*/fullfeat 2>/dev/null
  ls data/*/synth/ctgan.csv data/*/synth/tvae.csv 2>/dev/null; echo $L; } > $L/filelist.txt
tar czf "$OUT" -T $L/filelist.txt
echo "全部完成。请把 $OUT（以及 joboutput_${SLURM_JOB_ID:-local}.*）拷回本机 v2/ 目录"
