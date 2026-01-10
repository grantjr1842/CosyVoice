import torch
from safetensors import safe_open
import json
import numpy as np

def get_stats(t):
    return {
        "min": t.min().item(),
        "max": t.max().item(),
        "mean": t.mean().item(),
        "shape": list(t.shape),
        "first_elements": t.flatten()[:10].tolist()
    }

# Load Python s_stft
with safe_open("hift_stages_debug.safetensors", framework="pt") as f:
    s_stft_py = f.get_tensor("s_stft")

# Compute our own STFT in Python to see if it matches
with safe_open("debug_artifacts.safetensors", framework="pt") as f:
    # We need the excitation signal 's' which is used for s_stft
    # Wait, 'source' in hift_stages_debug is the excitation signal
    pass

with safe_open("hift_stages_debug.safetensors", framework="pt") as f:
    s = f.get_tensor("source") # [B, 1, T]

n_fft = 16
hop_len = 4
window = torch.hann_window(n_fft, periodic=True)

s_stft_calc = torch.stft(s.squeeze(1), n_fft, hop_len, n_fft, window=window, center=True, return_complex=False)
# s_stft_calc shape: [B, Freq, Frames, 2]
# Transform to [B, 2*Freq, Frames] to match Rust
b, f, t, _ = s_stft_calc.shape
s_stft_calc_cat = torch.cat([s_stft_calc[..., 0], s_stft_calc[..., 1]], dim=1)

print("Python internal calc vs Python saved artifact:")
print(f"  Calc min/max: {s_stft_calc_cat.min().item():.6f}, {s_stft_calc_cat.max().item():.6f}")
print(f"  Artif min/max: {s_stft_py.min().item():.6f}, {s_stft_py.max().item():.6f}")
print(f"  Max Diff: {(s_stft_calc_cat - s_stft_py).abs().max().item():.6e}")
