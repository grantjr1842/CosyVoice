#!/usr/bin/env python3
"""Extract source excitation tensor from Python HiFT for Rust injection."""

import torch
import numpy as np
from safetensors.torch import save_file, load_file
from hyperpyyaml import load_hyperpyyaml

def main():
    print("Loading HiFT model...")
    with open('./pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml', 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'flow': None})

    hift = configs['hift']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hift.to(device)
    hift.eval()

    print(f"Device: {device}")

    # Load mel from existing artifacts
    artifacts = load_file("debug_artifacts.safetensors")
    mel = artifacts["python_flow_output"].to(device)  # [1, 80, T]
    print(f"Mel shape: {mel.shape}")

    # Use the model's inference method which handles everything correctly
    with torch.no_grad():
        # HiFT.inference returns (speech, source_excitation)
        speech, source = hift.inference(mel)
        print(f"Speech shape: {speech.shape}")
        print(f"Source shape: {source.shape}")
        print(f"Source stats: min={source.min():.6f}, max={source.max():.6f}, mean={source.mean():.6f}")

        # Compute s_stft for comparison
        s_stft_real, s_stft_imag = hift._stft(source.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
        print(f"s_stft shape: {s_stft.shape}")
        print(f"s_stft stats: min={s_stft.min():.6f}, max={s_stft.max():.6f}")

        print(f"Audio stats: min={speech.min():.6f}, max={speech.max():.6f}")

    # Save source and related tensors
    save_file({
        "mel": mel.cpu().contiguous(),
        "source": source.cpu().contiguous(),
        "s_stft": s_stft.cpu().contiguous(),
        "audio": speech.cpu().unsqueeze(0).contiguous(),
    }, "source_injection_test.safetensors")

    print("\nSaved to source_injection_test.safetensors")


if __name__ == "__main__":
    main()
