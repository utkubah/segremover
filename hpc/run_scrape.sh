#!/bin/bash
#SBATCH --job-name=segremover-scrape
#SBATCH --output=logs/scrape_%A_%a.out
#SBATCH --error=logs/scrape_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-3           # 4 parallel shards — adjust to taste

# Usage:
#   sbatch hpc/run_scrape.sh
#
# Splits sources across 4 shards so multiple nodes scrape in parallel.
# The shared checkpoint file prevents duplicate downloads across shards.
# Merge candidate logs afterwards:
#   cat data/candidates_shard_*.csv | sort -u > data/candidates_all.csv

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs data/raw

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segremover

python src/scrape.py \
    --sources data/source_registry.csv \
    --target-total 5000 \
    --out-dir data/raw \
    --checkpoint data/scrape_checkpoint.jsonl \
    --candidate-log "data/candidates_shard_${SLURM_ARRAY_TASK_ID}.csv" \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --sleep-sec 1.5 \
    --max-per-channel 160

echo "Shard ${SLURM_ARRAY_TASK_ID} done."
