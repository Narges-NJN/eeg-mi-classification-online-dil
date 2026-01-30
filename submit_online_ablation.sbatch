#!/bin/bash
#SBATCH --job-name=online_ablation
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
mkdir -p slurm_logs

# ---- subjects ----
SUBJECTS=$(seq -s, 1 80)

# ---- single seed (screening phase) ----
SEED=42

# ---- ablations (MUST match run.py keys) ----
ABLATIONS=(
  baseline
  lr_5e-4 lr_1e-3 lr_5e-3 lr_3e-2
  wd_1e-4 wd_1e-3 wd_1e-2
  bs_8 bs_16 bs_32
  drop_0.10 drop_0.25 drop_0.50
  F1_16 F1_32
  D_2 D_4 D_8
  klen_32 klen_64 klen_128
  dwklen_16 dwklen_32 dwklen_64
  pool_max
)

ABL=${ABLATIONS[$SLURM_ARRAY_TASK_ID]}

echo "Running ablation=${ABL} seed=${SEED}"
nvidia-smi

python run.py \
  --exp online \
  --ablation "${ABL}" \
  --tag "online_${ABL}_seed${SEED}" \
  --set seed=${SEED} subjects=${SUBJECTS}
