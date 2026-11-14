#!/usr/bin/env bash
# =====================================================================
# 实验机（GPU）：投稿前补充实验（2026-11-11 修订：单卡运行）
#   resplit  在重复划分数据集 <ds>_r<k> 上重做 LoRA SFT + 生成 + 奖励筛选（每份划分都从该划分的训练集重新构造 SFT 数据，
#            测试样本不会进入提示或微调数据），再离线生成“按类相对阈值 / 不加重叠约束”两个变体
#   great    GReaT 基线（原划分，三个数据集）
#
# 单卡：默认只用一张卡（当前可见的第一张）。训练、生成全部在一张卡上完成，不存在任何卡间通信。
#   可选 GPU_LIST="0 1"：两张卡各跑一个**完全独立**的进程，各自处理一半划分（仍然没有卡间通信，
#   每个进程只看得到自己那张卡），总时间约减半。若担心该服务器的双卡问题，保持默认即可。
#
# 用法（在 v2/ 目录下，先解压 v3_cluster_inputs_fix.tgz）：
#   sbatch 或直接：bash gpu/run_gpu_v3.sh
#   日志：gpu_v3.log（总览，自动写入）+ logs/gpu_v3/<数据集>.log（每份划分的完整输出，含报错）
# 可选环境变量：
#   STAGES="resplit great"    DATASETS="cse2018 cic2017 nf_unsw_v3"    REPS="1 2 3"
#   GPU_LIST="0"     GPU 号写法与 CUDA_VISIBLE_DEVICES 相同（默认取当前可见的第一张）
#   KEEP_MODELS=1（保留每份划分约 15 GB 的合并模型；默认生成完即删除，只删本脚本新建的目录）
#   GREAT_LLM=<本地模型目录>   计算节点不能联网时，指向本地的 distilgpt2（默认从 HuggingFace 下载）
# 预计耗时（单卡 A10）：每份划分 CSE ≈ 1.2 h、CIC ≈ 1.6 h、NF ≈ 2.6 h；3 份 × 3 个数据集 ≈ 16 h（GPU_LIST="0 1" 约 8 h）
# 断点：已有输出会跳过，中断后重新运行即可。
# 输出：gpu_v3_results.tgz，拷回本机 v2/ 目录
# =====================================================================
#SBATCH --job-name=ids_gpu_v3
#SBATCH --gres=gpu:1              # 单卡；若用 GPU_LIST="0 1" 双队列，改成 gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=joboutput_%j.out
#SBATCH --error=joboutput_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f gpu/run_gpu.sh ] || { echo "!! 请在 v2/ 目录下运行"; exit 1; }
exec > >(tee -a gpu_v3.log) 2>&1
# 虚拟环境：依次查找 $VENV、v2/venvs/idsllm、~/venvs/idsllm
for v in "${VENV:-}" "$PWD/venvs/idsllm" "$HOME/venvs/idsllm"; do
  if [ -n "$v" ] && [ -f "$v/bin/activate" ]; then source "$v/bin/activate"; echo "使用虚拟环境：$v"; break; fi
done
[ -n "${VIRTUAL_ENV:-}" ] || echo "!! 没找到虚拟环境（v2/venvs/idsllm），将使用系统 python3：$(command -v python3)"
PY=${PYTHON:-python3}
STAGES=${STAGES:-"resplit great"}
DATASETS=${DATASETS:-"cse2018 cic2017 nf_unsw_v3"}
REPS=${REPS:-"1 2 3"}
# 默认用当前可见的第一张卡（sbatch 分配的卡会体现在 CUDA_VISIBLE_DEVICES 里，这里不擅自改成别的卡）
VIS=${CUDA_VISIBLE_DEVICES:-0}
GPU_LIST=${GPU_LIST:-${VIS%%,*}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs/gpu_v3; mkdir -p $L
has() { [[ " $STAGES " == *" $1 "* ]]; }
echo "================ 开始 $(date) ｜ 阶段=$STAGES ｜ GPU=$GPU_LIST ================"

# ---------- 启动前检查：任何一项不通过就立即退出，不再白跑
ok=1
for f in gpu/g1_sft_lora.py gpu/g4_dpo.py; do
  grep -q 'MULTI_GPU_SPLIT' $f || { echo "!! $f 是旧版本（缺少单卡限制，会触发 DataParallel → OOM）。请先解压 v3_cluster_inputs_fix.tgz"; ok=0; }
done
grep -q '_r\\d+' common.py || { echo "!! common.py 是旧版本（不认识 <ds>_r<k> 数据集名）"; ok=0; }
if has resplit; then
  for r in $REPS; do for ds in $DATASETS; do
    [ -f data/${ds}_r$r/train.csv.gz ] || { echo "!! 缺少 data/${ds}_r$r"; ok=0; }
  done; done
fi
"$PY" - <<'PYEOF' || ok=0
import torch, transformers, peft
n = torch.cuda.device_count()
print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "可见 GPU", n)
assert n >= 1, "看不到 GPU"
for i in range(n):  # 只检查本作业可见的卡
    free, tot = torch.cuda.mem_get_info(i)
    print(f"  GPU {i}: 空闲 {free/2**30:.1f} / {tot/2**30:.1f} GiB")
    if free / tot < 0.9:
        print(f"!! GPU {i} 已被其他进程占用 {(tot-free)/2**30:.1f} GiB，训练可能 OOM")
PYEOF
[ $ok = 1 ] || { echo "!! 启动前检查未通过，已退出"; exit 1; }

# ---------- 重复划分：每张卡一个独立队列
run_split() {  # $1 数据集  $2 GPU 号
  local d=$1 g=$2 log=$L/$1.log
  if [ -f data/$d/synth/llm_sft_filt.csv ]; then echo "SKIP $d（已有结果）"; return 0; fi
  echo "START $d  GPU $g  $(date +%T)  日志 $log"
  if CUDA_VISIBLE_DEVICES=$g VLLM_MODE=dp IDS_DATASET=$d STAGE=sft SKIP_BASE=1 bash gpu/run_gpu.sh > "$log" 2>&1; then
    for mode in rel none; do
      [ -f data/$d/synth/llm_sft_${mode}ov_filt.csv ] || IDS_DATASET=$d "$PY" cpu/09_refilter.py --source llm_sft_filt --mode $mode >> "$log" 2>&1
    done
    if [ -z "${KEEP_MODELS:-}" ] && [ -d models/llm/$d/round0_merged ]; then rm -rf "models/llm/$d/round0_merged"; fi
    echo "OK    $d  $(date +%T)"; return 0
  fi
  echo "FAIL  $d  $(date +%T)  最后 25 行："; grep -v "it/s\]\|examples/s\]" "$log" | tail -25 | sed 's/^/      /'; return 1
}
queue() {  # $1 GPU 号  其余：数据集列表
  local g=$1; shift; local nfail=0 nok=0
  for d in "$@"; do
    if run_split "$d" "$g"; then nok=$((nok+1)); nfail=0; else nfail=$((nfail+1)); fi
    if [ $nfail -ge 2 ] && [ $nok -eq 0 ]; then echo "!! GPU $g 上连续 2 份划分失败且没有成功过，判断为系统性问题，停止该队列"; return 1; fi
  done
}
if has resplit; then
  ALL=(); for r in $REPS; do for ds in $DATASETS; do ALL+=("${ds}_r$r"); done; done
  G=($GPU_LIST); NG=${#G[@]}; PIDS=()
  for k in "${!G[@]}"; do
    mine=(); for i in "${!ALL[@]}"; do [ $((i % NG)) -eq $k ] && mine+=("${ALL[$i]}"); done
    queue "${G[$k]}" "${mine[@]}" &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"   # 只等两个队列（不能用不带参数的 wait：它也会等写日志的 tee 进程，导致永远不结束）
fi

# ---------- GReaT（单卡）
if has great; then
  if ! "$PY" -c "import be_great" 2> $L/great_import.txt; then
    echo "be_great 不可用：$(tail -1 $L/great_import.txt)"
    echo "安装 be-great（用当前环境的版本作约束，不会升级 torch / transformers / numpy；冲突时直接失败，不改动环境）"
    "$PY" -m pip freeze | grep -v -i "be.great" > $L/constraints.txt
    "$PY" -m pip install -q be-great -c $L/constraints.txt > $L/great_install.txt 2>&1 \
      || { echo "!! be-great 安装失败，见 $L/great_install.txt（GReaT 跳过，不影响其他结果）"; }
  fi
  if "$PY" -c "import be_great" 2>/dev/null; then
    for ds in $DATASETS; do
      [ -f data/$ds/synth/great.csv ] && { echo "SKIP GReaT $ds"; continue; }
      echo "START GReaT $ds  $(date +%T)"
      if CUDA_VISIBLE_DEVICES=${GPU_LIST%% *} IDS_DATASET=$ds "$PY" gpu/g5_great.py ${GREAT_LLM:+--llm "$GREAT_LLM"} > $L/great_$ds.log 2>&1; then
        echo "OK    GReaT $ds"
      else
        echo "FAIL  GReaT $ds  最后 15 行："; tail -15 $L/great_$ds.log | sed 's/^/      /'
      fi
    done
  fi
fi

# ---------- 汇总与打包
echo -e "\n================ 汇总 ================"
for r in $REPS; do for ds in $DATASETS; do d=${ds}_r$r
  if [ -f data/$d/synth/llm_sft_filt.csv ]; then echo "  $d  完成（接受 $(($(wc -l < data/$d/synth/llm_sft_filt.csv)-1)) 条）"; else echo "  $d  未完成"; fi
done; done
for ds in $DATASETS; do [ -f data/$ds/synth/great.csv ] && echo "  GReaT $ds  完成" || echo "  GReaT $ds  未完成"; done
ls data/*_r*/synth/llm_sft*_filt.csv data/*_r*/llm/*stats.json data/*/synth/great.csv models/llm/*_r*/*_log.json 2>/dev/null > $L/filelist.txt
echo gpu_v3.log >> $L/filelist.txt; echo $L >> $L/filelist.txt
tar czf gpu_v3_results.tgz -T $L/filelist.txt
echo "已打包 gpu_v3_results.tgz（$(($(wc -l < $L/filelist.txt)-2)) 个结果文件 + 日志）  $(date)"
