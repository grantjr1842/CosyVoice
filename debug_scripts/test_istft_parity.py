import torch


def test_istft():
    n_fft = 16
    hop_len = 4
    window = torch.hann_window(n_fft, periodic=True)

    # Create a dummy spectrogram
    mag = torch.ones(1, n_fft // 2 + 1, 10).float()
    phase = torch.zeros(1, n_fft // 2 + 1, 10).float()
    spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    # torch.istft
    out_torch = torch.istft(spec, n_fft, hop_len, n_fft, window=window, center=True)
    print(f"Torch out shape: {out_torch.shape}")
    print(f"Torch out first 10: {out_torch[0, :10]}")
    print(f"Torch out max: {out_torch.abs().max().item()}")

    # Manual OLA simulation
    # 1. Inverse DFT
    scale = 1.0 / n_fft
    # For k=0, Nyquist: val * 1 else * 2
    # But wait, torch.istft has specific normalization.

    # Let's try to match a single pulse
    spec_pulse = torch.zeros_like(spec)
    spec_pulse[0, 0, 5] = 1.0  # DC only, frame 5
    out_pulse = torch.istft(
        spec_pulse, n_fft, hop_len, n_fft, window=window, center=True
    )
    print(
        f"Pulse center value: {out_pulse[0, 5 * hop_len + n_fft // 2]}"
    )  # center=True adds n_fft//2 padding?

    # Torch ISTFT with center=True pads the input signal with n_fft//2 on both sides.
    # So the first frame (m=0) is centered at t=0.
    # The reconstruction for m=0 covers -N/2 to N/2.


if __name__ == "__main__":
    test_istft()
