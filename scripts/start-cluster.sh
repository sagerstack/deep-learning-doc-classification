#!/bin/bash
# SUTD AI Mega Cluster setup
# Uses the cluster's pre-built Python environment with pip
# Run once after cloning, or after adding new dependencies
#
# Prerequisites:
#   git clone https://github.com/sagerstack/deep-learning-doc-classification.git dl-doc-classification
#   cd dl-doc-classification

set -e

echo "=== Cluster Setup ==="

# Set ENV_PROFILE for Jupyter kernels (runs on every kernel start)
mkdir -p ~/.ipython/profile_default/startup
echo 'import os; os.environ["ENV_PROFILE"] = "cluster"' > ~/.ipython/profile_default/startup/00-env.py
echo "ENV_PROFILE=cluster set for all Jupyter kernels"

# Install missing packages (skips if already installed)
echo "Installing dependencies..."
pip install -q scikit-learn matplotlib torch_geometric python-dotenv datasets "numpy<2" "Pillow>=10.0"

# Verify setup
echo ""
echo "=== Verifying ==="
python -c "
import torch, torchvision, torch_geometric, sklearn, matplotlib, datasets, dotenv
print(f'PyTorch:  {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'PyG:      {torch_geometric.__version__}')
print(f'Device:   {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
print(f'ENV_PROFILE: cluster')
print()
print('All dependencies OK')
"

echo ""
echo "=== Setup Complete ==="
echo "Use the default cluster Jupyter kernel (not DL Kernel)"
