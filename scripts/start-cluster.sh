#!/bin/bash
# SUTD AI Mega Cluster setup and Jupyter startup
# Run once after cloning, or after dependency changes
#
# One-time prerequisites (run manually):
#   git clone https://github.com/sagerstack/deep-learning-doc-classification.git dl-project-gnn
#   echo 'export ENV_PROFILE=cluster' >> ~/.bashrc && source ~/.bashrc

set -e

echo "=== Cluster Setup ==="

# Install dependencies
echo "1/4 Locking dependencies..."
poetry lock --no-update

echo "2/4 Installing dependencies..."
poetry install --no-root

echo "3/4 Installing PyTorch for CUDA 12.x (V100)..."
poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

echo "4/4 Registering Jupyter kernel..."
poetry run python -m ipykernel install --user --name dl-project --display-name "DL Project"

echo ""
echo "=== Setup Complete ==="
echo "ENV_PROFILE=$ENV_PROFILE"
echo "Select 'DL Project' kernel in notebooks"
