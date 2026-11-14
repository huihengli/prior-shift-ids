#!/usr/bin/env bash
# 附录：Westermo 外部验证（本机 CPU）。只调参 sa / grid（SAORF、RF-grid 所需），其余与主实验协议相同。
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2
for ds in westermo westermo_ps; do
  IDS_DATASET=$ds python3 - <<'PY'
import importlib.util, sys
sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location('d01', 'cpu/01_detectors.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.TUNERS = ['sa', 'grid']
m.part_tune(m.load_cfg(), quick=False)
PY
  for src in none cw ros smote; do
    IDS_DATASET=$ds python3 cpu/12_eval_proba.py --sources $src --detectors SAORF RF-grid LightGBM XGBoost --n-jobs 2
  done
done
echo ALL_DONE
