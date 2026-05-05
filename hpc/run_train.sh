#!/bin/bash
#SBATCH --job-name=segremover-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1           # single A100/V100 sufficient for roberta-base

# Usage:
#   sbatch hpc/run_train.sh
#   sbatch hpc/run_train.sh --encoder roberta-large
#
# Requires data/processed/{weak_labels,function_labels,disfluency_labels}.jsonl
# Run run_labels.sh first.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs models

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segremover

echo "=== Training segremover ==="
python src/train.py "$@"

echo "=== Training complete. Checkpoint saved to models/ ==="
ls -lh models/
