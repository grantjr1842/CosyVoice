from safetensors.torch import load_file
import torch

artifacts = load_file("debug_artifacts.safetensors")
mel = artifacts["python_flow_output"]
print(f"Mel shape: {mel.shape}")
print(f"Mel stats: min={mel.min().item():.6f}, max={mel.max().item():.6f}, mean={mel.mean().item():.6f}")
