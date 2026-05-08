#!/bin/bash
#SBATCH --job-name=segremover-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Usage:
#   sbatch hpc/run_eval.sh                         # full eval (needs models/best.pt)
#   sbatch hpc/run_eval.sh --no-model              # baselines + transcript only
#   sbatch hpc/run_eval.sh --max-docs 5 --no-model # smoke test
#
# Run order:
#   run_labels.sh → run_baselines.sh → run_train.sh → run_eval.sh

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

echo "=== Evaluation ==="
python src/eval.py "$@"

echo "=== Done ==="
echo "Outputs:"
ls -lh outputs/eval/
echo "Plots:"
ls -lh outputs/eval/plots/*.png 2>/dev/null | wc -l
echo " PNG files"
