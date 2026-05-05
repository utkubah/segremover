#!/bin/bash
# Bootstrap the segremover conda environment on the HPC.
# Run this ONCE after cloning the repo:
#
#   bash hpc/setup_env.sh
#
# After it finishes every job script can activate with:
#   conda activate segremover

set -euo pipefail

ENV_NAME="segremover"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -y -n "$ENV_NAME" python=3.11

echo "=== Activating ==="
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "=== Installing requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Downloading spaCy model ==="
python -m spacy download en_core_web_sm

echo ""
echo "Done. Add the following line to each job script to activate the environment:"
echo "  source \$(conda info --base)/etc/profile.d/conda.sh && conda activate $ENV_NAME"
