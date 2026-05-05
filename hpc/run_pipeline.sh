#!/bin/bash
# Submit the full segremover pipeline as a chain of SLURM jobs.
# Each stage is held until its predecessor finishes successfully.
#
# Usage:
#   bash hpc/run_pipeline.sh                                   # rules-only labels
#   bash hpc/run_pipeline.sh microsoft/Phi-3.5-mini-instruct  # + LLM LF
#
# Job chain:
#   [scrape array] → segment → labels → train
#
# Run from the repo root:
#   cd ~/segremover && bash hpc/run_pipeline.sh

set -euo pipefail

LLM_MODEL="${1:-}"

mkdir -p logs

echo "=== Submitting segremover pipeline ==="

# --- Stage 1: Scrape (4-shard array) ---
SCRAPE_JOB=$(sbatch --parsable hpc/run_scrape.sh)
echo "  scrape array job: $SCRAPE_JOB"

# --- Stage 2: Segment (depends on all scrape shards) ---
SEG_JOB=$(sbatch --parsable \
    --dependency=afterok:"${SCRAPE_JOB}" \
    hpc/run_segment.sh)
echo "  segment job: $SEG_JOB"

# --- Stage 3: Labels (depends on segment) ---
if [ -n "$LLM_MODEL" ]; then
    LABEL_JOB=$(sbatch --parsable \
        --dependency=afterok:"${SEG_JOB}" \
        hpc/run_labels.sh "$LLM_MODEL")
else
    LABEL_JOB=$(sbatch --parsable \
        --dependency=afterok:"${SEG_JOB}" \
        hpc/run_labels.sh)
fi
echo "  labels job: $LABEL_JOB  (LLM: ${LLM_MODEL:-none})"

# --- Stage 4: Train (depends on labels) ---
TRAIN_JOB=$(sbatch --parsable \
    --dependency=afterok:"${LABEL_JOB}" \
    hpc/run_train.sh)
echo "  train job: $TRAIN_JOB"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f logs/train_${TRAIN_JOB}.out"
