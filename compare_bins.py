import torch
from safetensors import safe_open
import numpy as np

# Load Python s_stft
with safe_open("hift_stages_debug.safetensors", framework="pt") as f:
    s_stft_py = f.get_tensor("s_stft") # [B, 18, T]
    source = f.get_tensor("source") # [B, 1, T]

frame_idx = 112
print(f"Python stats for s_stft (Frame {frame_idx}):")
print(f"  Real: {s_stft_py[0, :9, frame_idx].tolist()}")

# Compute manually
n_fft = 16
hop_len = 4
window = torch.hann_window(n_fft, periodic=True)

# Padding to match center=True
padding = n_fft // 2
source_padded = torch.nn.functional.pad(source, (padding, padding), mode='reflect')

# Extract samples for frame
start = frame_idx * hop_len
frame_samples = source_padded[0, 0, start : start + n_fft]
frame_windowed = frame_samples * window

# FFT
fft_out = torch.fft.rfft(frame_windowed)
print(f"\nManual FFT stats (Frame {frame_idx}):")
print(f"  Real: {fft_out.real.tolist()}")

ratio = s_stft_py[0, :9, frame_idx] / fft_out.real
print(f"\nRatio (Py / Manual): {ratio.tolist()}")
