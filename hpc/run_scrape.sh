#!/bin/bash
#SBATCH --job-name=segremover-scrape
#SBATCH --partition=defq
#SBATCH --output=scrape_%j.out
#SBATCH --error=scrape_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs data/raw

exec > >(tee -a "logs/scrape_${SLURM_JOB_ID}.out") \
     2> >(tee -a "logs/scrape_${SLURM_JOB_ID}.err" >&2)

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Repo: ${REPO_DIR}"
echo "Started: $(date)"

if [[ ! -f src/scrape.py ]]; then
    echo "ERROR: src/scrape.py not found"
    exit 1
fi

if [[ ! -f data/source_registry.csv ]]; then
    echo "ERROR: data/source_registry.csv not found"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segremover

SHARD_ID=0
NUM_SHARDS=1

python src/scrape.py \
    --sources data/source_registry.csv \
    --target-total 5000 \
    --out-dir data/raw \
    --checkpoint data/scrape_checkpoint.jsonl \
    --candidate-log "data/candidates_shard_${SHARD_ID}.csv" \
    --shard-id "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}" \
    --sleep-sec 1.5 \
    --max-per-channel 160

echo "Finished: $(date)"
echo "Scrape job done."