#!/bin/bash
#SBATCH --job-name=segremover-labels
#SBATCH --output=logs/labels_%j.out
#SBATCH --error=logs/labels_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Usage (rule LFs only, no GPU needed — remove --gres line above):
#   sbatch hpc/run_labels.sh
#
# Usage (with open-source LLM for extra LF coverage):
#   sbatch hpc/run_labels.sh microsoft/Phi-3.5-mini-instruct
#   sbatch hpc/run_labels.sh mistralai/Mistral-7B-Instruct-v0.3
#
# Correct run order (weak_labels must come last so it sees all auxiliary files):
#   1. summary_align       → summary_align_votes.jsonl
#   2. function_labels     → function_labels.jsonl      (Head B)
#   3. disfluency_labels   → disfluency_labels.jsonl    (Head C)
#   4. weak_labels         → weak_labels.jsonl           (Head A, uses 1-3 as extra LFs)
#
# Requires segments_topic.jsonl to exist (run run_segment.sh first).

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

LLM_MODEL="${1:-}"

echo "=== Step 1: Summary alignment LF (BART) ==="
python src/summary_align.py

echo "=== Step 2: Function weak labels (Head B — 13 rule LFs$([ -n "$LLM_MODEL" ] && echo " + LLM")) ==="
if [ -n "$LLM_MODEL" ]; then
    python src/function_labels.py --llm-model "$LLM_MODEL"
else
    python src/function_labels.py
fi

echo "=== Step 3: Disfluency weak labels (Head C — 7 rule LFs$([ -n "$LLM_MODEL" ] && echo " + LLM")) ==="
if [ -n "$LLM_MODEL" ]; then
    python src/disfluency_labels.py --llm-model "$LLM_MODEL"
else
    python src/disfluency_labels.py
fi

echo "=== Step 4: Removability weak labels (Head A — picks up all auxiliary LFs) ==="
python src/weak_labels.py

echo "=== All label files ready ==="
ls -lh data/processed/*.jsonl
