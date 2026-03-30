#!/bin/bash
# Start Jupyter on SUTD cluster with full dataset config
export ENV_PROFILE=cluster
echo "ENV_PROFILE=$ENV_PROFILE (loading config.cluster)"
jupyter notebook
