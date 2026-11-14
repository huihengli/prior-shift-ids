#!/usr/bin/env bash
# =====================================================================
# 实验机（GPU）：第二个 LLM 的敏感性分析（Llama-3.1-8B-Instruct，2026-11-12）
#   与 Qwen2.5-7B 完全相同的流程与超参数（config.yaml 中的 llm.*），只换底座模型：
#     orig     原划分（cse2018 / cic2017 / nf_unsw_v3）：LoRA SFT → 生成 → 奖励筛选 → 离线重筛选（按类相对阈值 / 不加重叠约束）
#     base     原划分：未微调 Llama 的 few-shot 生成 → 奖励筛选（与 Qwen 的 llm_base 对照，可选，很快）
#     resplit  重复划分 <ds>_r1..r3：每份划分重新 SFT → 生成 → 筛选 → 重筛选（与 F27 的 Qwen 协议一致）
#   不跑 DPO 闭环、消融、GReaT（F13 / F15 已说明闭环几乎不学习）。
#
# 输出命名（不会覆盖任何 Qwen 结果）：
#   data/<ds>/synth/llm_sft_llama.csv、llm_sft_llama_filt.csv、llm_sft_llama_relov_filt.csv、llm_sft_llama_noneov_filt.csv
#   data/<ds>/synth/llm_base_llama.csv、llm_base_llama_filt.csv
#   data/<ds>/llm/gen_llm_*_llama_stats.json、reward_llm_*_llama_stats.json
#   models/llm/<ds>_llama/（sft_log.json；合并模型默认生成完即删除）
#
# 用法（在 v2/ 目录下，先解压 llama_inputs.tgz）：
#   LLAMA_PATH=/path/to/Meta-Llama-3.1-8B-Instruct  bash gpu/run_gpu_llama.sh
#   或 sbatch（先把下面的 LLAMA_PATH 默认值改成你的路径，或在 sbatch 前 export LLAMA_PATH=...）
# 可选环境变量：
#   STAGES="check orig base resplit"（默认全部；check = 启动前检查 + 5 步 SFT 冒烟测试 + 小规模生成测试）
#   DATASETS="cse2018 cic2017 nf_unsw_v3"   REPS="1 2 3"
#   GPU_LIST="0 1"   两张卡各跑一个完全独立的队列，总时间约减半；默认只用第一张可见的卡
#                    每个进程都只看得到自己那一张卡（CUDA_VISIBLE_DEVICES=单个卡号，启动时检查），训练与 vLLM 生成都在单卡上，
#                    不做模型切分、不共享显存、没有任何卡间通信（该服务器 P2P 异常，见记录 7.6）
#   KEEP_MODELS=1    保留合并后的模型（每个约 16 GB）
# 预计耗时（单卡 A10，按 Qwen 实测估计）：check ≈ 20 min；orig ≈ 5.5 h；base ≈ 1.5 h；resplit ≈ 16 h
#   合计约 23 h（GPU_LIST="0 1" 约 12 h）
# 断点：已有输出会跳过，中断后重新运行同一命令即可。
# 输出：gpu_llama_results.tgz（拷回本机 v2/ 目录解压；再到 CPU 节点运行 cpu/run_cpu_llama.sh）
# =====================================================================
#SBATCH --job-name=ids_gpu_llama
#SBATCH --gres=gpu:1              # 单卡；若用 GPU_LIST="0 1"，改成 gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=joboutput_%j.out
#SBATCH --error=joboutput_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f gpu/g1_sft_lora.py ] && [ -f common.py ] || { echo "!! 请在 v2/ 目录下运行"; exit 1; }
exec > >(tee -a gpu_llama.log) 2>&1

for v in "${VENV:-}" "$PWD/venvs/idsllm" "$HOME/venvs/idsllm"; do
  if [ -n "$v" ] && [ -f "$v/bin/activate" ]; then source "$v/bin/activate"; echo "使用虚拟环境：$v"; break; fi
done
[ -n "${VIRTUAL_ENV:-}" ] || echo "!! 没找到虚拟环境（v2/venvs/idsllm），将使用系统 python3：$(command -v python3)"
PY=${PYTHON:-python3}

LLAMA_PATH=${LLAMA_PATH:-}          # ← 也可以直接在这里写死路径
STAGES=${STAGES:-"check orig base resplit"}
DATASETS=${DATASETS:-"cse2018 cic2017 nf_unsw_v3"}
REPS=${REPS:-"1 2 3"}
VIS=${CUDA_VISIBLE_DEVICES:-0}
GPU_LIST=${GPU_LIST:-${VIS%%,*}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false NCCL_P2P_DISABLE=1
export VLLM_MODE=dp GEN_TIMEOUT_MIN=${GEN_TIMEOUT_MIN:-240}
export LLM_TAG=llama
L=logs/gpu_llama; mkdir -p $L
has() { [[ " $STAGES " == *" $1 "* ]]; }
echo "================ 开始 $(date) ｜ 阶段=$STAGES ｜ GPU=$GPU_LIST ================"

# ---------- 启动前检查（任何一项不通过就退出）
ok=1
[ -n "$LLAMA_PATH" ] || { echo "!! 请设置 LLAMA_PATH=<Llama-3.1-8B-Instruct 本地目录>"; exit 1; }
BASE=$("$PY" gpu/model_path.py "$LLAMA_PATH") || { echo "!! LLAMA_PATH 不可用：$LLAMA_PATH"; exit 1; }
export LLM_BASE_MODEL=$BASE
echo "Llama 模型目录：$BASE"
grep -q 'LLM_BASE_MODEL' common.py || { echo "!! common.py 不是本次的新版本（缺少 LLM_BASE_MODEL），请先解压 llama_inputs.tgz"; ok=0; }
grep -q 'LLM_TAG' gpu/g1_sft_lora.py || { echo "!! gpu/g1_sft_lora.py 不是本次的新版本（缺少 LLM_TAG）"; ok=0; }
grep -q 'MULTI_GPU_SPLIT' gpu/g1_sft_lora.py || { echo "!! gpu/g1_sft_lora.py 缺少单卡限制"; ok=0; }
"$PY" - "$BASE" <<'PYEOF' || ok=0
import json, sys
from pathlib import Path
c = json.load(open(Path(sys.argv[1]) / "config.json"))
arch = c.get("architectures", ["?"])[0]
print("模型架构:", arch, "| hidden", c.get("hidden_size"), "| layers", c.get("num_hidden_layers"), "| vocab", c.get("vocab_size"))
assert arch == "LlamaForCausalLM", f"不是 Llama 架构：{arch}（确认下载的是 Llama-3.1-8B-Instruct）"
tc = Path(sys.argv[1]) / "tokenizer_config.json"
assert tc.exists() and "chat_template" in tc.read_text(encoding="utf-8"), "tokenizer_config.json 中没有 chat_template：请下载 Instruct 版本"
PYEOF
for ds in $DATASETS; do
  [ -f data/$ds/train.csv.gz ] || { echo "!! 缺少 data/$ds"; ok=0; }
  [ -f results/$ds/detectors/tune/sa_s0.json ] || echo "!! 缺少 results/$ds/detectors/tune/sa_s*.json（奖励中的检测器会退化为默认 RF）"
done
if has resplit; then for r in $REPS; do for ds in $DATASETS; do
  [ -f data/${ds}_r$r/train.csv.gz ] || { echo "!! 缺少 data/${ds}_r$r"; ok=0; }
done; done; fi
"$PY" - <<'PYEOF' || ok=0
import torch, transformers, peft
n = torch.cuda.device_count()
print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "可见 GPU", n)
assert n >= 1, "看不到 GPU"
assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 43), "Llama 3.1 需要 transformers >= 4.43"
for i in range(n):
    free, tot = torch.cuda.mem_get_info(i)
    print(f"  GPU {i}: 空闲 {free/2**30:.1f} / {tot/2**30:.1f} GiB")
    if free / tot < 0.9:
        print(f"!! GPU {i} 已被其他进程占用 {(tot-free)/2**30:.1f} GiB，训练可能 OOM")
PYEOF
[ $ok = 1 ] || { echo "!! 启动前检查未通过，已退出"; exit 1; }

# ---------- 冒烟测试：5 步 SFT（不保存模型）+ 小规模生成（每类 8 条，结果随后删除）
one_gpu() {  # 确认当前进程只看得到 1 张卡（不存在跨卡切分 / 通信）
  "$PY" -c "import torch,sys; n=torch.cuda.device_count(); print(f'[gpu] 本进程可见 GPU 数 = {n}（应为 1）'); sys.exit(0 if n==1 else 1)"
}
if has check; then
  d=${DATASETS%% *}
  [ -f data/$d/llm/sft.jsonl ] || IDS_DATASET=$d "$PY" gpu/g0_build_sft_data.py > $L/check_g0.log 2>&1
  for g in $GPU_LIST; do
  echo -e "\n>>> 冒烟测试（GPU $g，$d，只用这一张卡）$(date +%T)"
  CUDA_VISIBLE_DEVICES=$g one_gpu || { echo "!! GPU $g：进程可见的卡数不是 1，已退出"; exit 1; }
  if CUDA_VISIBLE_DEVICES=$g IDS_DATASET=$d "$PY" gpu/g1_sft_lora.py --max-steps 5 > $L/check_sft_gpu$g.log 2>&1; then
    grep -E "\[check\]|\[gpu\]|\[test\]|显存|peak|\[time\]" $L/check_sft_gpu$g.log | tail -8 | sed 's/^/   /'
  else
    echo "!! SFT 冒烟测试失败（完整输出 $L/check_sft_gpu$g.log），最后 25 行："; tail -25 $L/check_sft_gpu$g.log | sed 's/^/      /'
    echo "   常见原因：显存不足 → 在 config.yaml 中把 llm.sft.batch_size 改为 1、grad_accum 改为 16 后重试"; exit 1
  fi
  if CUDA_VISIBLE_DEVICES=$g IDS_DATASET=$d "$PY" gpu/g2_generate.py --model "$BASE" --name _smoke_llama --n-per-class 8 > $L/check_gen_gpu$g.log 2>&1; then
    grep -E "parse_rate|engine" $L/check_gen_gpu$g.log | tail -2 | cut -c1-300 | sed 's/^/   /'
  else
    echo "!! 生成冒烟测试失败（完整输出 $L/check_gen_gpu$g.log），最后 25 行："; tail -25 $L/check_gen_gpu$g.log | sed 's/^/      /'; exit 1
  fi
  rm -f data/$d/synth/_smoke_llama.csv data/$d/llm/gen__smoke_llama.jsonl data/$d/llm/gen__smoke_llama_stats.json
  echo "   GPU $g 冒烟测试通过 $(date +%T)"
  done
fi

# ---------- 单个任务
sft_job() {  # $1 数据集  $2 GPU
  local d=$1 g=$2 log=$L/sft_$1.log M=models/llm/$1_llama S=data/$1/synth
  if [ -f $S/llm_sft_llama_relov_filt.csv ] && [ -f $S/llm_sft_llama_noneov_filt.csv ]; then echo "SKIP  sft $d（已有结果）"; return 0; fi
  echo "START sft $d  GPU $g  $(date +%T)  日志 $log"
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$g IDS_DATASET=$d
    one_gpu   # 只看得到 1 张卡，否则任务失败
    [ -f data/$d/llm/sft.jsonl ] || "$PY" gpu/g0_build_sft_data.py
    if [ ! -f $S/llm_sft_llama.csv ]; then
      [ -d $M/round0_merged ] || "$PY" gpu/g1_sft_lora.py
      "$PY" gpu/g2_generate.py --model $M/round0_merged --name llm_sft_llama --round 0
    fi
    [ -f $S/llm_sft_llama_filt.csv ] || "$PY" gpu/g3_reward.py --gen llm_sft_llama --out llm_sft_llama
    for mode in rel none; do
      [ -f $S/llm_sft_llama_${mode}ov_filt.csv ] || "$PY" cpu/09_refilter.py --source llm_sft_llama_filt --mode $mode
    done
  ) > "$log" 2>&1
  if [ $? -eq 0 ]; then
    if [ -z "${KEEP_MODELS:-}" ] && [ -d $M/round0_merged ]; then rm -rf "$M/round0_merged"; fi
    echo "OK    sft $d  $(date +%T)"; return 0
  fi
  echo "FAIL  sft $d  $(date +%T)  最后 25 行："; grep -v "it/s\]\|examples/s\]" "$log" | tail -25 | sed 's/^/      /'; return 1
}
base_job() {  # 未微调 Llama 的 few-shot 生成（只在原划分上）
  local d=$1 g=$2 log=$L/base_$1.log S=data/$1/synth
  if [ -f $S/llm_base_llama_filt.csv ]; then echo "SKIP  base $d（已有结果）"; return 0; fi
  echo "START base $d  GPU $g  $(date +%T)  日志 $log"
  (
    set -e
    export CUDA_VISIBLE_DEVICES=$g IDS_DATASET=$d
    one_gpu   # 只看得到 1 张卡，否则任务失败
    [ -f $S/llm_base_llama.csv ] || "$PY" gpu/g2_generate.py --model "$BASE" --name llm_base_llama
    "$PY" gpu/g3_reward.py --gen llm_base_llama --out llm_base_llama
  ) > "$log" 2>&1
  if [ $? -eq 0 ]; then echo "OK    base $d  $(date +%T)"; return 0; fi
  echo "WARN  base $d 未完成（未微调模型解析率过低时属正常，论文中如实报告解析率即可），最后 10 行："
  tail -10 "$log" | sed 's/^/      /'; return 0
}
queue() {  # $1 GPU  其余：任务（sft:<ds> 或 base:<ds>）
  local g=$1; shift; local nfail=0 nok=0
  for t in "$@"; do
    if ${t%%:*}_job "${t#*:}" "$g"; then nok=$((nok+1)); nfail=0; else nfail=$((nfail+1)); fi
    if [ $nfail -ge 2 ] && [ $nok -eq 0 ]; then echo "!! GPU $g 上连续 2 个任务失败且没有成功过，判断为系统性问题，停止该队列"; return 1; fi
  done
}

# ---------- 组织任务：原划分 → 未微调基线 → 重复划分；按 GPU 轮流分配
ALL=()
has orig && for ds in $DATASETS; do ALL+=("sft:$ds"); done
has base && for ds in $DATASETS; do ALL+=("base:$ds"); done
has resplit && for r in $REPS; do for ds in $DATASETS; do ALL+=("sft:${ds}_r$r"); done; done
if [ ${#ALL[@]} -gt 0 ]; then
  G=($GPU_LIST); NG=${#G[@]}; PIDS=()
  echo -e "\n任务 ${#ALL[@]} 个，GPU 队列 $NG 个"
  for k in "${!G[@]}"; do
    mine=(); for i in "${!ALL[@]}"; do [ $((i % NG)) -eq $k ] && mine+=("${ALL[$i]}"); done
    queue "${G[$k]}" "${mine[@]}" &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
fi

# ---------- 汇总与打包
echo -e "\n================ 汇总 ================"
show() { local f=data/$1/synth/$2.csv; if [ -f $f ]; then echo "  $1  $2  接受 $(($(wc -l < $f)-1)) 条"; else echo "  $1  $2  未完成"; fi; }
for ds in $DATASETS; do show $ds llm_sft_llama_filt; show $ds llm_base_llama_filt; done
for r in $REPS; do for ds in $DATASETS; do show ${ds}_r$r llm_sft_llama_filt; done; done
ls data/*/synth/llm_*_llama*.csv data/*/llm/gen_llm_*_llama_stats.json data/*/llm/reward_llm_*_llama_stats.json \
   models/llm/*_llama/sft_log.json 2>/dev/null > $L/filelist.txt
echo gpu_llama.log >> $L/filelist.txt; echo $L >> $L/filelist.txt
tar czf gpu_llama_results.tgz -T $L/filelist.txt
echo "已打包 gpu_llama_results.tgz（$(($(wc -l < $L/filelist.txt)-2)) 个结果文件 + 日志）  $(date)"
echo "下一步：在 CPU 节点上运行  sbatch cpu/run_cpu_llama.sh（或 bash cpu/run_cpu_llama.sh）"
