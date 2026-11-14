#!/usr/bin/env bash
# =====================================================================
# 实验机（GPU）一键脚本：未微调基线 → LoRA SFT → 3 轮“生成-奖励-DPO”闭环 → 奖励消融 → GReaT 基线
# 用法（在 v2/ 目录下）：  bash gpu/run_gpu.sh  2>&1 | tee gpu_run.log
# 可分段：STAGE=sft|loop|ablation|great bash gpu/run_gpu.sh
# 断点：每一步若输出已存在会跳过（删掉对应文件即可重跑该步）
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
# 默认使用两块 GPU（0 和 1）；单卡时可改为 CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}   # PCIe A10 上 P2P 常导致多卡通信卡死
# 生成默认每张卡各跑一个独立 vLLM 进程（VLLM_MODE=dp），卡间无通信；单步生成超过 GEN_TIMEOUT_MIN 分钟会自动终止报错
export VLLM_MODE=${VLLM_MODE:-dp}
export GEN_TIMEOUT_MIN=${GEN_TIMEOUT_MIN:-240}
PY=${PYTHON:-python3}
python() { "$PY" "$@"; }
"$PY" -c "import sys; assert sys.version_info >= (3, 9), sys.version; print('python', sys.version.split()[0])"
"$PY" -c "import torch; n=torch.cuda.device_count(); print('GPUs visible:', n, [torch.cuda.get_device_name(i) for i in range(n)]); assert n>=1, 'no GPU'"
DS=${IDS_DATASET:-cse2018}
M=models/llm/$DS
BASE_CFG=$(python -c "import yaml;print(yaml.safe_load(open('config.yaml',encoding='utf-8'))['llm']['base_model'])")
# 启动前先确认模型目录真的存在（支持多套一层目录 / HF 缓存格式，会自动找到含 config.json 的子目录）
BASE=$(python gpu/model_path.py "$BASE_CFG") || { echo "!! config.yaml 中 llm.base_model 不可用：$BASE_CFG"; exit 1; }
echo "base model: $BASE"
Q=${LOAD_4BIT:+--load-4bit}          # 显存不够时：LOAD_4BIT=1 bash gpu/run_gpu.sh
STAGE=${STAGE:-all}
SYN=data/$DS/synth
ROUNDS=$(python -c "import yaml;print(yaml.safe_load(open('config.yaml',encoding='utf-8'))['llm']['loop']['rounds'])")

run() { echo -e "\n>>> $*"; date; "$@"; }
gen()  { [ -f "$SYN/$2.csv" ]      || run python gpu/g2_generate.py --model "$1" --name "$2" ${3:+--round $3}; }
rew()  { [ -f "$SYN/$2_filt.csv" ] || run python gpu/g3_reward.py --gen "$1" --out "$2" "${@:3}"; }
dpo()  { [ -d "$M/$3_merged" ]     || run python gpu/g4_dpo.py --model "$1" --pairs "$2" --out "$3" $Q; }

if [[ $STAGE == all || $STAGE == sft ]]; then
  [ -f data/$DS/llm/sft.jsonl ] || run python gpu/g0_build_sft_data.py
  # 基线 1：未微调 LLM（仅提示词 + few-shot），对应旧稿 B 组
  if [[ -z ${SKIP_BASE:-} ]]; then   # SKIP_BASE=1：跳过未微调基线（重复划分实验只需要 SFT 样本）
    gen "$BASE" llm_base
    rew llm_base llm_base || echo "!! llm_base 无可解析样本，跳过其打分（未微调模型常见，论文中如实报告解析率即可）"
  fi
  # LoRA 监督微调，对应旧稿 C 组
  [ -d "$M/round0_merged" ] || run python gpu/g1_sft_lora.py $Q
  gen "$M/round0_merged" llm_sft 0
  rew llm_sft llm_sft
fi

if [[ $STAGE == all || $STAGE == loop ]]; then
  # 闭环：第 r 轮用上一轮奖励产生的偏好对做 DPO，再生成、再用更新后的检测器打分
  PREV=(llm_sft_filt); PAIRS=llm_sft; MODEL=$M/round0_merged
  for r in $(seq 1 "$ROUNDS"); do
    dpo "$MODEL" "$PAIRS" "round$r"
    MODEL=$M/round${r}_merged
    gen "$MODEL" "llm_loop_r$r" "$r"
    if ! python -c "
import json,yaml,sys
st=json.load(open('data/$DS/llm/gen_llm_loop_r$r''_stats.json'))
thr=yaml.safe_load(open('config.yaml',encoding='utf-8'))['llm']['loop'].get('min_parse_rate',0.9)
print('round $r parse_rate=%.3f (min %.2f)'%(st['parse_rate'],thr)); sys.exit(0 if st['parse_rate']>=thr else 1)"; then
      echo "!! 第 $r 轮生成解析率过低，模型可能已塌缩，停止闭环（已完成的轮次结果保留）"; break
    fi
    rew "llm_loop_r$r" "llm_loop_r$r" --prev "${PREV[@]}"
    PREV+=("llm_loop_r${r}_filt"); PAIRS=llm_loop_r$r
  done
fi

if [[ $STAGE == all || $STAGE == ablation ]]; then
  # 奖励分项消融：各去掉一项，从 SFT 模型出发做 1 轮 DPO，与 llm_loop_r1 对比
  for ab in hard fid mem overlap; do
    rew llm_sft "abl_no$ab" --ablate $ab
    dpo "$M/round0_merged" "abl_no$ab" "abl_no$ab"
    gen "$M/abl_no${ab}_merged" "abl_no${ab}_r1" 1
    rew "abl_no${ab}_r1" "abl_no${ab}_r1" --prev llm_sft_filt --ablate $ab
  done
fi

if [[ $STAGE == all || $STAGE == great ]]; then
  if python -c "import be_great" 2>/dev/null; then
    [ -f "$SYN/great.csv" ] || run python gpu/g5_great.py || echo "!! GReaT 基线失败（不影响主实验）"
  else
    echo "!! 未安装 be_great，跳过 GReaT 基线。安装：pip install be-great --no-deps，然后 STAGE=great bash gpu/run_gpu.sh"
  fi
fi

echo -e "\n全部完成。请把以下内容拷回本机 v2/ 对应位置：\n  data/$DS/synth/   data/$DS/llm/*.json*（不含模型）   models/llm/$DS/*_log.json"
