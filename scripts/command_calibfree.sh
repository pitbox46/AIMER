#!/usr/bin/env bash

# Usage:
#   bash scripts/command_calibfree.sh <cuda_visible_devices> <model_name> aimer <sparsity_ratio> <seed>

bash experiments/calib-free-cli.sh \
  0,1 \
  Qwen/Qwen3-30B-A3B-Instruct-2507 \
  aimer \
  0.25 \
  42

