#!/bin/bash
#SBATCH --job-name=segremover-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=20:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Train SegRemover with the cascade "explainer → ranker" architecture.
#
# Architecture: Head B (function) and Head C (disfluency) run first as explainers;
# Head A (p_remove) is conditioned on [h ‖ logit_B ‖ logit_C] — the ranker is
# explicitly driven by the model's own explanations.
#
# Defaults (roberta-base, covers ~75% of segments fully):
#   --encoder roberta-base  --inter-layers 4
#   --max-segs 256          --max-tok 384
#   --seg-batch 32          --batch-size 2   --grad-accum 8
#   --w-b 1.0               --w-c 1.0
#
# Usage:
#   sbatch hpc/run_train.sh                          # standard run (roberta-base)
#   sbatch hpc/run_train.sh --max-docs 50            # smoke test
#
# roberta-large recipe (pre-download model on login node first — see below):
#   sbatch hpc/run_train.sh \
#       --encoder roberta-large \
#       --seg-batch 8 --batch-size 1 --grad-accum 16
#
# To pre-download roberta-large on the login node (run once before sbatch):
#   conda activate segremover
#   HF_HOME=/mnt/beegfsstudents/home/3223837/hf_cache \
#   python -c "from transformers import AutoTokenizer, RobertaModel; \
#              AutoTokenizer.from_pretrained('roberta-large'); \
#              RobertaModel.from_pretrained('roberta-large')"
#
# Requires data/processed/{weak_labels,function_labels,disfluency_labels}.jsonl
# Re-run hpc/run_weak_labels.sh first if LFs changed.
#
# Run order:
#   run_labels.sh → run_train.sh → run_eval.sh

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

mkdir -p logs models

export HF_HOME=/mnt/beegfsstudents/home/3223837/hf_cache
export SENTENCE_TRANSFORMERS_HOME=/mnt/beegfsstudents/home/3223837/hf_cache/sentence_transformers
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segremover

echo "=== Training segremover ==="
python src/train.py "$@"

echo "=== Training complete. Checkpoint saved to models/ ==="
ls -lh models/
