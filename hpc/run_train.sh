#!/bin/bash
#SBATCH --job-name=segremover-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Trains SegRemover (full model) + no-doc-context ablation, then runs a single
# eval that includes both variants in the ablation table.
#
# Variants trained:
#   1. Full model (3 epochs)    → models/best.pt       + models/config.json
#   2. No-doc ablation (3 epochs, no inter-seg Transformer)
#                               → models/best_no_doc.pt + models/config_no_doc.json
#
# Eval (single run) produces:
#   outputs/eval/ablation_table.csv  — full / w/o cascade / w/o inter-seg Transformer
#   outputs/eval/plots/              — all paper figures
#
# Time estimate (3h per full-model epoch):
#   Full model  3 epochs  ~9h
#   No-doc      3 epochs  ~5.5h  (no inter-seg Transformer, fewer ops per doc)
#   Eval (once) ~2h
#   Total       ~16.5h  (within 24h wall time)
#
# Usage:
#   sbatch hpc/run_train.sh               # full run
#   sbatch hpc/run_train.sh --max-docs 20 # smoke test (passed to BOTH train calls)
#
# Run order:
#   run_labels.sh → run_train.sh → (eval is inline below)

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

mkdir -p logs models outputs/eval

export HF_HOME=/mnt/beegfsstudents/home/3223837/hf_cache
export SENTENCE_TRANSFORMERS_HOME=/mnt/beegfsstudents/home/3223837/hf_cache/sentence_transformers
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segremover

# ── 1. Full model ─────────────────────────────────────────────────────────────
echo "================================================================"
echo "  TRAINING  (1/2): Full model"
echo "  Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

python src/train.py "$@"

echo "  Finished full model: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
ls -lh models/best.pt models/config.json

# ── 2. No-doc-context ablation ────────────────────────────────────────────────
echo "================================================================"
echo "  TRAINING  (2/2): No-doc-context ablation"
echo "  Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

python src/train.py --no-doc-context --output-suffix no_doc "$@"

echo "  Finished no-doc model: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
ls -lh models/best_no_doc.pt models/config_no_doc.json

echo "================================================================"
echo "  ALL DONE: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Run: sbatch hpc/run_eval.sh to generate evaluation report."
echo "================================================================"
ls -lh models/
