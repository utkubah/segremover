#!/bin/bash
#SBATCH --job-name=segremover-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=16:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Evaluation suite for SegRemover.
#
# Runs ONE eval call covering:
#   - Full model (models/best.pt)
#   - Cascade ablation (--ablate-cascade: zeros Head B/C logits at inference)
#   - No-doc ablation  (--ablate-no-doc models/best_no_doc.pt: loads the retrained variant)
#
# All three appear as rows in outputs/eval/ablation_table.csv.
#
# Usage:
#   sbatch hpc/run_eval.sh                    # standard
#   sbatch hpc/run_eval.sh                    # uses all data (--eval-sample 0 is the default)
#   sbatch hpc/run_eval.sh --no-model        # baselines only
#
# Run order:
#   run_labels.sh → run_train.sh → run_eval.sh

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

mkdir -p logs outputs/eval

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
echo "================================================================"

SPLIT_ARG=""
if [ -f models/test_ids.json ]; then
    SPLIT_ARG="--split-file models/test_ids.json"
    echo "  Using held-out test split: models/test_ids.json"
else
    echo "  WARNING: models/test_ids.json not found — evaluating on all data in data/processed."
    echo "  Run: python src/train.py --save-splits-only  to regenerate without retraining."
fi

python src/eval.py \
    --checkpoint        models/best.pt \
    --data-dir          data/processed \
    --output-dir        outputs/eval \
    $SPLIT_ARG \
    --eval-sample       500 \
    --transcript-sample 10 \
    --thresholds        0.50 0.60 0.70 0.80 0.90 0.95 \
    --paper-mode \
    --ablate-cascade \
    --ablate-no-doc     models/best_no_doc.pt \
    --pareto \
    "$@"

echo "================================================================"
echo "  Finished  : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Report:         outputs/eval/evaluation_report.md"
echo "  Ablation table: outputs/eval/ablation_table.csv"
echo "  Plots:          $(ls outputs/eval/plots/*.png 2>/dev/null | wc -l) PNG files"
echo "================================================================"
