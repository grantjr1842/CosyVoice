import torch
from safetensors import safe_open
import json

with safe_open("hift_stages_debug.safetensors", framework="pt") as f:
    s_stft_py = f.get_tensor("s_stft")

frame_idx = 112
print(f"Python s_stft Frame {frame_idx} All Bins:")
# Real part
print(f"  Real: {s_stft_py[0, :9, frame_idx].tolist()}")
# Imag part
print(f"  Imag: {s_stft_py[0, 9:, frame_idx].tolist()}")

# Check if frame 113 is also small
print(f"\nPython s_stft Frame 113 All Bins:")
print(f"  Real: {s_stft_py[0, :9, 113].tolist()}")
