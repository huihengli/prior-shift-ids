#!/usr/bin/env bash
# =====================================================================
# 实验机 CPU 节点：第二个 LLM（Llama-3.1-8B-Instruct）合成样本的评估（2026-11-12）
#   fidelity  04_fidelity：原划分三个数据集上 Llama 来源的生成质量（W1 / C2ST / 记忆率 / 多样性 / TSTR）
#             同时重算 Qwen 的对应来源作为同口径对照；结果写到 results/<ds>/fidelity_llama/，不覆盖已有汇总
#   eval      12_eval_proba 原划分：Llama 来源 × 4 检测器 × 10 种子（与 5J.1 / F26 相同协议）
#   resplit   12_eval_proba 重复划分 r1–r3：Llama SFT 来源 × 4 检测器 × 5 种子（与 F27 相同协议）
# 前提：GPU 脚本 gpu/run_gpu_llama.sh 已完成。若 CPU 节点与 GPU 节点不共享 v2/ 目录，
#       先在这里解压 gpu_llama_results.tgz（tar xzf gpu_llama_results.tgz）。
# 用法（在 v2/ 目录下）：sbatch cpu/run_cpu_llama.sh   或   bash cpu/run_cpu_llama.sh 2>&1 | tee cpu_llama.log
# 可选环境变量：STAGES="fidelity eval resplit"  DATASETS="cse2018 nf_unsw_v3 cic2017"  REPS="1 2 3"
#               THREADS=8（每任务线程数；并行任务数 = CPU 数 / THREADS）  DRYRUN=1（只打印任务）
# 断点：已存在的结果文件会跳过；中断后重新提交即可。
# 输出：cpu_llama_results_<作业号>.tgz，拷回本机 v2/ 目录解压
# 预计：约 160 个任务（每个任务含该来源 × 检测器的全部种子），128 核约 1–2 小时
# =====================================================================
#SBATCH --job-name=ids_cpu_llama
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=joboutput_%j.out
#SBATCH --error=joboutput_%j.err
##SBATCH --partition=请改成CPU分区名

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f common.py ] && [ -f cpu/12_eval_proba.py ] || { echo "!! 请在 v2/ 目录下提交"; exit 1; }
for v in "${VENV:-}" "$PWD/venvs/idsllm" "$HOME/venvs/idsllm"; do
  if [ -n "$v" ] && [ -f "$v/bin/activate" ]; then source "$v/bin/activate"; echo "使用虚拟环境：$v"; break; fi
done
PY=${PYTHON:-python3}
need="numpy pandas yaml scipy sklearn xgboost lightgbm imblearn"
miss=$("$PY" - $need <<'PYEOF'
import importlib.util, sys
pip = {"yaml": "pyyaml", "sklearn": "scikit-learn", "imblearn": "imbalanced-learn"}
print(" ".join(pip.get(m, m) for m in sys.argv[1:] if importlib.util.find_spec(m) is None))
PYEOF
)
[ -n "$miss" ] && { echo "安装缺少的包：$miss"; "$PY" -m pip install -q $miss || echo "!! 安装失败：$miss"; }
grep -q 'out-sub' cpu/04_fidelity.py || { echo "!! cpu/04_fidelity.py 不是本次的新版本（缺少 --out-sub），请先解压 llama_inputs.tgz"; exit 1; }

[ -f config_cpu.yaml ] && export IDS_CONFIG=$PWD/config_cpu.yaml
NCPU=${SLURM_CPUS_PER_TASK:-$(nproc)}
if [ -z "${DRYRUN:-}" ] && [ "$NCPU" -lt 16 ] && [ -z "${ALLOW_FEW_CPUS:-}" ]; then
  echo "!! 只分配到 $NCPU 个 CPU，约 160 个评估任务（含全部种子）会跑很久。请用 sbatch 提交（头部已写 --cpus-per-task=128），"
  echo "   确实要在少量 CPU 上跑请设 ALLOW_FEW_CPUS=1。已退出。"; exit 1
fi
THREADS=${THREADS:-8}
P=$(( NCPU / THREADS )); [ "$P" -ge 1 ] || P=1
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
STAGES=${STAGES:-"fidelity eval resplit"}
DATASETS=${DATASETS:-"cse2018 nf_unsw_v3 cic2017"}
REPS=${REPS:-"1 2 3"}
DETS=${DETS:-"SAORF RF-grid LightGBM XGBoost"}
L=logs/cpu_llama; mkdir -p $L
has() { [[ " $STAGES " == *" $1 "* ]]; }
echo "CPU=$NCPU 每任务线程=$THREADS 并行=$P 阶段=$STAGES"; date

run_line() {
  log=${1%%|*}; cmd=${1#*|}
  if bash -c "$cmd" > "$L/$log.log" 2>&1; then echo "OK   $log  $(date +%T)"; else echo "FAIL $log  （见 $L/$log.log），最后 6 行："; grep -v "it/s\]" "$L/$log.log" | tail -6 | sed 's/^/      /'; fi
}
export -f run_line; export L PY THREADS
run_all() {
  if [ -n "${DRYRUN:-}" ]; then cut -d'|' -f2- "$1"; return; fi
  [ -s "$1" ] && xargs -P "${2:-$P}" -d '\n' -I{} bash -c 'run_line "$1"' _ {} < "$1" | tee -a $L/progress.txt
}
: > $L/progress.txt; : > $L/skipped.txt

# 原划分上的 Llama 来源；重复划分只有 SFT 来源（与 Qwen 的 F27 协议一致）
SRC_ORIG="llm_sft_llama_filt llm_sft_llama_relov_filt llm_sft_llama_noneov_filt llm_base_llama_filt"
SRC_REP="llm_sft_llama_filt llm_sft_llama_relov_filt llm_sft_llama_noneov_filt"
# 同口径对照：Qwen 的对应来源（CSE 的 Qwen 来源名带 v2_ 前缀）
qwen_src() { case $1 in cse2018) echo "v2_llm_base_filt v2_llm_sft_filt v2_llm_sft_relov_filt" ;; *) echo "llm_base_filt llm_sft_filt llm_sft_relov_filt llm_sft_noneov_filt" ;; esac; }

T=$L/tasks.txt; : > $T
if has fidelity; then for ds in $DATASETS; do
  srcs=""; for s in $SRC_ORIG $(qwen_src $ds) ros smote; do
    [ -f data/$ds/synth/$s.csv ] && srcs="$srcs $s" || echo "$ds fidelity $s" >> $L/skipped.txt
  done
  # 只有当已有汇总包含全部来源时才跳过（否则会漏掉 GPU 之后才生成的来源，如 11-13 的 llm_base_llama / llm_sft_llama_*）
  f=results/$ds/fidelity_llama/fidelity_summary.csv
  if [ ! -f $f ] || ! "$PY" -c "import pandas as pd,sys; s=set(pd.read_csv('$f').source); sys.exit(0 if set('$srcs'.split())<=s else 1)"; then
    echo "fid__$ds|IDS_DATASET=$ds $PY cpu/04_fidelity.py --out-sub fidelity_llama --sources$srcs" >> $T
  fi
done; fi
eval_tasks() {  # $1 数据集  $2 来源列表  $3 额外参数
  for s in $2; do
    [ -f data/$1/synth/$s.csv ] || { echo "$1 $s" >> $L/skipped.txt; continue; }
    for det in $DETS; do echo "eval__$1__${det}__$s|IDS_DATASET=$1 $PY cpu/12_eval_proba.py --sources $s --detectors $det --n-jobs $THREADS $3"; done
  done
}
if has eval; then for ds in $DATASETS; do eval_tasks "$ds" "$SRC_ORIG" "" >> $T; done; fi
if has resplit; then for ds in $DATASETS; do for r in $REPS; do
  [ -f data/${ds}_r$r/train.csv.gz ] && eval_tasks "${ds}_r$r" "$SRC_REP" "--seeds 5 --no-cv" >> $T
done; done; fi
{ grep -E '^fid__' $T; grep -E '__(SAORF|RF-grid)__' $T; grep -vE '^fid__|__(SAORF|RF-grid)__' $T; } > $T.sorted
[ -s $L/skipped.txt ] && echo "缺少合成文件、本次未评估的来源 $(wc -l < $L/skipped.txt) 个，清单见 $L/skipped.txt（未微调 Llama 解析率过低时 llm_base_llama_filt 缺失属正常）"
echo "任务：$(wc -l < $T.sorted)"; run_all $T.sorted
date

[ -n "${DRYRUN:-}" ] && exit 0
echo "失败任务数：$(grep -c '^FAIL' $L/progress.txt || true)"; grep '^FAIL' $L/progress.txt || true
OUT=cpu_llama_results_${SLURM_JOB_ID:-local}.tgz
{ ls results/*/augment_proba/*llama* results/*/augment_proba/meta.json 2>/dev/null
  ls -d results/*/fidelity_llama 2>/dev/null; echo $L; } > $L/filelist.txt
tar czf $OUT -T $L/filelist.txt
echo "已打包 $OUT  $(date)"
