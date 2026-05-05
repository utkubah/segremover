#!/bin/bash
#SBATCH --job-name=segremover-segment
#SBATCH --output=logs/segment_%j.out
#SBATCH --error=logs/segment_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Usage:
#   sbatch hpc/run_segment.sh
#
# Reads  data/raw/**/*.json
# Writes data/processed/segments_topic.jsonl
#        data/processed/segments_sent.jsonl

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs data/processed

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segremover

echo "=== Segmenting transcripts ==="
python src/segment.py

echo "=== Done ==="
wc -l data/processed/segments_topic.jsonl
