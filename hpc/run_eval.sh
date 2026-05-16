#!/bin/bash
#SBATCH --job-name=segremover-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Full evaluation suite for SegRemover.
#
# Produces all segment-level, transcript-level, baseline, agreement,
# gold, and explainability (Head B / Head C) plots and reports.
#
# Usage:
#   sbatch hpc/run_eval.sh                    # standard full eval (100-doc sample)
#   sbatch hpc/run_eval.sh --eval-sample 0   # evaluate entire corpus (slow)
#   sbatch hpc/run_eval.sh --no-model        # skip model inference (baselines only)
#
# Run order:
#   run_labels.sh → run_baselines.sh → run_train.sh → run_eval.sh
#
# Output: eval/  (plots/, evaluation_report.md, gold_eval.md, qualitative_examples.md)

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

mkdir -p logs eval/plots

# Offline HuggingFace cache — models pre-downloaded during setup
export HF_HOME=/mnt/beegfsstudents/home/3223837/hf_cache
export SENTENCE_TRANSFORMERS_HOME=/mnt/beegfsstudents/home/3223837/hf_cache/sentence_transformers
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segremover

echo "================================================================"
echo "  SegRemover — Evaluation"
echo "  Node      : $(hostname)"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Started   : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Checkpoint: models/best.pt"
echo "  Output dir: eval/"
echo "================================================================"

python src/eval.py \
    --checkpoint        models/best.pt \
    --data-dir          data/processed \
    --output-dir        eval \
    --transcript-sample 2 \
    --thresholds        0.50 0.60 0.70 0.80 0.90 0.95 \
    --paper-mode \
    --ablate-cascade \
    --pareto \
    "$@"

echo "================================================================"
echo "  Finished  : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Plots generated:"
ls eval/plots/*.png 2>/dev/null | wc -l
echo "  Report    : eval/evaluation_report.md"
echo "================================================================"
