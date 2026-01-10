#!/usr/bin/env python3
"""Test STFT parity between Python torch.stft and Rust StftModule."""

import torch
import numpy as np
from scipy.signal import get_window
from safetensors.torch import save_file

def main():
    # Create test signal - simple sine wave
    sample_rate = 24000
    duration = 0.1  # 100ms
    freq = 440  # A4
    n_samples = int(sample_rate * duration)

    t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
    source = 0.1 * torch.sin(2 * np.pi * freq * t)  # [n_samples]
    source = source.unsqueeze(0)  # [1, n_samples]

    print(f"Source shape: {source.shape}")
    print(f"Source stats: min={source.min():.6f}, max={source.max():.6f}, mean={source.mean():.6f}")

    # STFT parameters (matching HiFT)
    n_fft = 16
    hop_len = 4

    # Hann window (periodic)
    window = torch.from_numpy(get_window("hann", n_fft, fftbins=True).astype(np.float32))

    # Run torch.stft
    spec = torch.stft(
        source.squeeze(0),  # [n_samples]
        n_fft,
        hop_len,
        n_fft,  # win_length
        window=window,
        return_complex=True
    )
    # spec: [n_fft/2+1, frames]

    spec_real = spec.real  # [n_fft/2+1, frames]
    spec_imag = spec.imag

    print(f"\nSTFT output shape: {spec.shape}")
    print(f"Real stats: min={spec_real.min():.6f}, max={spec_real.max():.6f}, mean={spec_real.mean():.6f}")
    print(f"Imag stats: min={spec_imag.min():.6f}, max={spec_imag.max():.6f}, mean={spec_imag.mean():.6f}")

    # Combine into s_stft format [1, n_fft+2, frames] = [real, imag] stacked
    s_stft = torch.cat([spec_real.unsqueeze(0), spec_imag.unsqueeze(0)], dim=1)
    print(f"\ns_stft shape: {s_stft.shape}")
    print(f"s_stft stats: min={s_stft.min():.6f}, max={s_stft.max():.6f}, mean={s_stft.mean():.6f}")

    # Save for Rust comparison
    save_file({
        "source": source.contiguous(),
        "s_stft_real": spec_real.unsqueeze(0).contiguous(),  # [1, n_fft/2+1, frames]
        "s_stft_imag": spec_imag.unsqueeze(0).contiguous(),
        "s_stft": s_stft.contiguous(),
    }, "stft_parity_test.safetensors")

    print("\nSaved to stft_parity_test.safetensors")

    # Also print first few values for debugging
    print(f"\nFirst frame real values: {spec_real[:, 0].numpy()}")
    print(f"First frame imag values: {spec_imag[:, 0].numpy()}")


if __name__ == "__main__":
    main()
