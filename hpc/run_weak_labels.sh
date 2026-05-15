#!/bin/bash
#SBATCH --job-name=segremover-weeklabels
#SBATCH --output=logs/weak_%j.out
#SBATCH --error=logs/weak_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Regenerate ONLY weak_labels.jsonl (Head A targets) using existing
# function_labels.jsonl and disfluency_labels.jsonl.
#
# Use this (instead of the full run_labels.sh) when only the LF logic in
# src/weak_labels.py has changed — function and disfluency labels are unchanged.
#
# Usage:
#   sbatch hpc/run_weak_labels.sh
#   sbatch hpc/run_weak_labels.sh --max-docs 50   # smoke test
#
# Run order:
#   run_weak_labels.sh → run_train.sh → run_eval.sh

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

echo "================================================================"
echo "  SegRemover — Weak Label Regeneration (step 4 only)"
echo "  Node      : $(hostname)"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Started   : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  LF change : LF_function_keep split into 3 per-class LFs"
echo "              (new_info, useful_rep, clarification)"
echo "================================================================"

python src/weak_labels.py "$@"

echo "================================================================"
echo "  Finished  : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Output    : data/processed/weak_labels.jsonl"
echo "================================================================"
