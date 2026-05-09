#!/bin/bash
#SBATCH --job-name=segremover-baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Usage:
#   sbatch hpc/run_baselines.sh
#   sbatch hpc/run_baselines.sh --max-docs 100   # smoke test

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

mkdir -p logs data/processed

export HF_HOME=/mnt/beegfsstudents/home/3223837/hf_cache
export SENTENCE_TRANSFORMERS_HOME=/mnt/beegfsstudents/home/3223837/hf_cache/sentence_transformers
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segremover

echo "=== Baselines ==="
python src/baselines.py "$@"

echo "=== Done ==="
ls -lh data/processed/baselines.jsonl
