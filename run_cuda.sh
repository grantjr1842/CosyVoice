#!/bin/bash
set -e
export PATH="/home/grant/github/CosyVoice-1/.pixi/envs/default/bin:$PATH"
export CUDA_HOME="/home/grant/github/CosyVoice-1/.pixi/envs/default"
export LD_LIBRARY_PATH="/home/grant/github/CosyVoice-1/lib_links:/home/grant/github/CosyVoice-1/.pixi/envs/default/lib"
export LIBRARY_PATH="/home/grant/github/CosyVoice-1/lib_links:/home/grant/github/CosyVoice-1/.pixi/envs/default/lib"
export COSYVOICE_MODEL_DIR="/home/grant/github/CosyVoice-1/pretrained_models/Fun-CosyVoice3-0.5B"
export ARTIFACT_PATH="tests/test_artifacts.safetensors"

echo "Running test_native (debug/cuda)..."
/home/grant/github/CosyVoice-1/rust/target/debug/test_native
