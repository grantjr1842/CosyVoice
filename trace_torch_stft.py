import torch
import numpy as np

n_fft = 16
window = torch.hann_window(n_fft, periodic=True)

# Test with a constant signal
x = torch.ones(1, 1, 32)
s_stft_norm = torch.stft(x.squeeze(1), n_fft, 4, n_fft, window=window, center=False, return_complex=True, normalized=True)
print(f"Constant signal (1.0) STFT (normalized=True) DC bin: {s_stft_norm[0, 0, 0].item()}")

# Test with a sine wave
freq = 1
t = torch.arange(n_fft).float()
s_vals = torch.cos(2 * np.pi * freq * t / n_fft)
s_stft_sine_norm = torch.stft(s_vals.unsqueeze(0), n_fft, n_fft, n_fft, window=window, center=False, return_complex=True, normalized=True)
print(f"Cosine (bin 1) STFT (normalized=True) bin 1: {s_stft_sine_norm[0, 1, 0].item()}")
