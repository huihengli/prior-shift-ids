# Prior Shift in Disguise — code and results

Code and result files for the paper

> **Prior Shift in Disguise: Rethinking the Evaluation of Data Augmentation for Rare-Attack Network Intrusion Detection** (under review at the *Journal of Information Security and Applications*).

The paper evaluates data augmentation for rare attack classes in flow-based intrusion detection — from SMOTE and generic tabular generators to fine-tuned LLMs (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct) — under a protocol with de-duplication, repeated splits, reweighting baselines and a threshold-only baseline. This repository contains everything needed to rerun the experiments and to rebuild every table and figure from the included result files.

> Code comments and log messages are mostly in Chinese; function names, configuration keys and outputs are in English. Each script starts with a docstring describing its purpose, inputs and outputs.

## Repository layout

```
config.yaml            all settings (datasets, splits, detectors, augmentation, LLM, reward)
common.py              shared helpers: loading, features, detectors, metrics
cpu/                   data preparation, augmentation baselines, evaluation, analysis (CPU only)
gpu/                   LLM pipeline: fine-tuning data, LoRA SFT, vLLM sampling, reward/filter, DPO/RPO, GReaT
figures/               pgfplots/TikZ sources of Figs. 1–8, figure data (figures/data) and prep_figs.py
results/               result files used in the paper (see "Included results")
data/                  per-dataset cleaning statistics; raw data go to data/raw/ (not included)
requirements-cpu.txt   CPU environment
requirements-gpu.txt   GPU environment (tested versions)
```

## Data

The datasets are not redistributed. Download them from their publishers and place them as follows (paths can be changed in `config.yaml`):

| Dataset | Files | Location |
|---|---|---|
| CSE-CIC-IDS2018 | CICFlowMeter CSVs of 14, 21 and 22 February and 1 March 2018 | `data/raw/cse2018/` |
| NF-UNSW-NB15-v3 | `NF-UNSW-NB15-v3.csv` | `data/raw/nf_unsw_v3/` |
| CIC-IDS2017 | `GeneratedLabelledFlows` CSVs (with the `Protocol` column; *not* `MachineLearningCVE`) | `data/raw/cic2017/` |
| Westermo (appendix) | the dataset repository, flows in `data/extended/flows/` | `data/raw/westermo/` |

As a check, `cpu/00_prepare_data.py` reports the number of raw rows; the paper used 3,476,837 (CSE-CIC-IDS2018), 2,365,424 (NF-UNSW-NB15-v3), 3,119,345 (CIC-IDS2017) and 68,729 (Westermo) rows. The cleaning statistics of every dataset are in `data/<dataset>/prep_stats.json` and `class_table.csv`.

## Environment

```bash
pip install -r requirements-cpu.txt        # Python 3.9+; add `sdv` for CTGAN/TVAE
pip install -r requirements-gpu.txt        # GPU part; tested on 2 x NVIDIA A10 (24 GB), CUDA 12.4
```

The base LLMs are loaded from Hugging Face (`llm.base_model` in `config.yaml`, or the environment variable `LLM_BASE_MODEL`). Llama-3.1-8B-Instruct requires accepting Meta's licence.

All scripts take the dataset from the environment variable `IDS_DATASET` (`cse2018`, `nf_unsw_v3`, `cic2017`, `westermo`, `westermo_ps`; repeated splits are `<dataset>_r1` … `<dataset>_r5`) and skip work whose results already exist, so they can be resumed.

## Pipeline and mapping to the paper

| Step | Script | Paper |
|---|---|---|
| Cleaning, de-duplication, conflict removal, stratified split | `cpu/00_prepare_data.py`, `cpu/00b_prepare_westermo.py` | §4.1, Table 1, App. D |
| Repeated splits (`<ds>_r1..r5`) | `cpu/11_resplit.py` | §4.1, §5.2 |
| Tuning-method comparison and detectors | `cpu/01_detectors.py` | §4.3, App. B |
| ROS, SMOTE, Borderline-SMOTE, ADASYN, CTGAN, TVAE pools | `cpu/02_classic_augment.py` | §4.4 |
| GReaT pool | `gpu/g5_great.py` | §4.4 |
| LLM fine-tuning data (anchor prompts) | `gpu/g0_build_sft_data.py`, `gpu/prompts.py` | §4.4, App. C.1 |
| LoRA SFT | `gpu/g1_sft_lora.py` | §4.4 |
| Sampling with vLLM | `gpu/g2_generate.py` | §4.4 |
| Reward, overlap filter, anchor balancing, preference pairs | `gpu/g3_reward.py` | Eq. (1), §4.4 |
| DPO/RPO feedback loop | `gpu/g4_dpo.py` | App. C.4 |
| Overlap-filter variants (offline) | `cpu/08_overlap_filter.py`, `cpu/09_refilter.py` | §4.4 |
| Main evaluation: F1, PR-AUC, false positives, full and threshold-only prior shift | `cpu/12_eval_proba.py` | §4.5–4.6, Tables 4–5, App. A |
| Earlier evaluation (F1 only, original split) | `cpu/03_eval_augmentation.py`, `cpu/05_stats.py` | — |
| Fidelity (W1, C2ST, memorisation, diversity, TSTR) | `cpu/04_fidelity.py` | Table 6 |
| Class purity versus gain | `cpu/10_overlap_gain.py` | Fig. 4, §6.4 |
| Full flow features | `cpu/13_fullfeat_prep.py`, `cpu/13b_fullfeat_eval.py` | Table 7, Fig. 7 |
| Controlled experiment | `cpu/14_controlled.py` | Fig. 6 |
| Duplicates and conflicting labels | `cpu/16a_dup_extract.py`, `cpu/16_dup_artefact.py` | Table 3 |
| Summaries | `cpu/15_summarize_v3.py`, `cpu/15b_summarize_llama.py` | Tables 4–6, Figs. 1, 3 |
| Figures | `figures/prep_figs.py`, `figures/fig*.tex` | Figs. 1–8 |

Batch scripts that run these steps in order are in `cpu/run_cpu_cluster_v3.sh`, `cpu/run_cpu_llama.sh`, `cpu/run_westermo.sh`, `gpu/run_gpu.sh`, `gpu/run_gpu_v3.sh` and `gpu/run_gpu_llama.sh` (written for a SLURM cluster; they also run with `bash`).

A minimal run for one dataset:

```bash
export IDS_DATASET=nf_unsw_v3
python cpu/00_prepare_data.py
python cpu/01_detectors.py
python cpu/02_classic_augment.py
python cpu/12_eval_proba.py --sources none cw ros smote --detectors SAORF RF-grid LightGBM XGBoost
python cpu/11_resplit.py                                   # then repeat the evaluation with IDS_DATASET=nf_unsw_v3_r1 ...
# LLM pipeline (GPU)
python gpu/g0_build_sft_data.py && python gpu/g1_sft_lora.py
python gpu/g2_generate.py --model models/llm/nf_unsw_v3/round0_merged --name llm_sft --round 0
python gpu/g3_reward.py --gen llm_sft --out llm_sft
```

See the docstring of each script for all options.

## Included results

`results/` contains the files from which the paper's numbers were computed:

* `results/<ds>/augment_proba/` — one JSON per source × detector × seed (main evaluation, all splits, Westermo included);
* `results/<ds>/detectors/` — tuning runs; `results/<ds>/tables/` — detector and tuning tables;
* `results/<ds>/fidelity_v3/`, `fidelity_llama/` — fidelity metrics;
* `results/<ds>/controlled2/` — controlled experiment; `results/<ds>/fullfeat/` — full-feature evaluation;
* `results/<ds>/dup_artefact/` — duplicate/conflict comparison (Table 3);
* `results/v3_summary/`, `llama_summary/`, `westermo_summary/`, `overlap_gain/` — summary tables.

The synthetic flow pools, fine-tuned models and the raw datasets are not included because of their size and the datasets' licences.

## Hardware notes

All fine-tuning and DPO runs use a single GPU. On our server, splitting a model across two GPUs produced wrong forward passes because of faulty peer-to-peer communication, so the scripts restrict each training process to one visible GPU; sampling runs one independent vLLM instance per GPU (`VLLM_MODE=dp`). If your GPUs communicate correctly, `MULTI_GPU_SPLIT=1` re-enables model splitting; `gpu/diag_forward.py` checks whether this is safe. Appendix E of the paper lists further pitfalls (vLLM per-request seeds, loss normalisation in Transformers 4.46).

## Citation

If you use this code, please cite the paper (citation details will be added on publication; see `CITATION.cff`).

## Licence

Code: MIT (see `LICENSE`). The datasets are subject to their own licences.
