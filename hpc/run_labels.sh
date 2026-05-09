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

# Usage (rule LFs only):
#   sbatch hpc/run_labels.sh
#
# Usage (with LLM):
#   sbatch hpc/run_labels.sh --llm Qwen/Qwen2.5-3B-Instruct
#
# Usage (smoke test — first 50 documents):
#   sbatch hpc/run_labels.sh --max-docs 50
#   sbatch hpc/run_labels.sh --llm Qwen/Qwen2.5-3B-Instruct --max-docs 50
#
# Correct run order:
#   1. summary_align       → summary_align_votes.jsonl
#   2. function_labels     → function_labels.jsonl
#   3. disfluency_labels   → disfluency_labels.jsonl
#   4. weak_labels         → weak_labels.jsonl

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

# ── Argument parsing ──────────────────────────────────────────────────────────
LLM_MODEL=""
MAX_DOCS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llm)        LLM_MODEL="$2"; shift 2 ;;
        --max-docs)   MAX_DOCS="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Build shared flags forwarded to every Python script
COMMON_FLAGS=""
if [ -n "$MAX_DOCS" ]; then
    COMMON_FLAGS="--max-docs $MAX_DOCS"
fi

echo "=== Step 1: Summary alignment LF (BART) ==="
python src/summary_align.py $COMMON_FLAGS

echo "=== Step 2: Function weak labels (Head B$([ -n "$LLM_MODEL" ] && echo " + LLM: $LLM_MODEL")) ==="
if [ -n "$LLM_MODEL" ]; then
    python src/function_labels.py --llm-model "$LLM_MODEL" $COMMON_FLAGS
else
    python src/function_labels.py $COMMON_FLAGS
fi

echo "=== Step 3: Disfluency weak labels (Head C$([ -n "$LLM_MODEL" ] && echo " + LLM: $LLM_MODEL")) ==="
if [ -n "$LLM_MODEL" ]; then
    python src/disfluency_labels.py --llm-model "$LLM_MODEL" $COMMON_FLAGS
else
    python src/disfluency_labels.py $COMMON_FLAGS
fi

echo "=== Step 4: Removability weak labels (Head A) ==="
python src/weak_labels.py $COMMON_FLAGS

echo "=== All label files ready ==="
ls -lh data/processed/*.jsonl
