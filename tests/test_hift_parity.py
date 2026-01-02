#!/usr/bin/env python3
"""
HiFT parity test - run Python HiFT on exact same input as Rust test_native.
Compare outputs to find the divergence.
"""

import os
import sys
from pathlib import Path

import torch
import torchaudio
from safetensors.torch import load_file, save_file

# Add project root and third_party to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "Matcha-TTS"))

from hyperpyyaml import load_hyperpyyaml


def tensor_stats(t: torch.Tensor, name: str) -> dict:
    """Compute statistics for a tensor."""
    t_flat = t.float().flatten().detach().cpu()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": float(t_flat.min().item()),
        "max": float(t_flat.max().item()),
        "mean": float(t_flat.mean().item()),
        "std": float(t_flat.std().item()) if t_flat.numel() > 1 else 0.0,
        "first_10": t_flat[:10].tolist() if t_flat.numel() >= 10 else t_flat.tolist(),
    }


def main():
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"

    # Load the same test artifacts used by Rust
    print("Loading test artifacts...")
    artifacts = load_file("tests/test_artifacts.safetensors")
    flow_feat_24k = artifacts["flow_feat_24k"]  # [1, 80, 171]

    print(f"Input mel (flow_feat_24k) shape: {flow_feat_24k.shape}")
    print(
        f"Input mel stats: min={flow_feat_24k.min():.4f}, max={flow_feat_24k.max():.4f}, mean={flow_feat_24k.mean():.4f}"
    )

    # Load config
    hyper_yaml_path = os.path.join(model_dir, "cosyvoice3.yaml")
    with open(hyper_yaml_path, "r") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")
            },
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("test_hift_debug.log", "w") as log:
        log.write(f"Device: {device}\n")

    # Load HiFT
    print("\nLoading HiFT model...")
    hift = configs["hift"]
    hift_state_dict = {
        k.replace("generator.", ""): v
        for k, v in torch.load(
            os.path.join(model_dir, "hift.pt"), map_location=device, weights_only=True
        ).items()
    }
    hift.load_state_dict(hift_state_dict, strict=True)
    hift.to(device).eval()
    print("HiFT loaded!")
    with open("test_hift_debug.log", "a") as log:
        log.write(f"HiFT device check: {next(hift.parameters()).device}\n")

    # Move mel to device
    mel = flow_feat_24k.to(device)
    log_path = Path("outputs/logs/test_hift_debug.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"Mel device: {mel.device}\n")

    # Run Python HiFT inference
    print("\n=== Running Python HiFT ===")
    with torch.inference_mode():
        audio, cache_source = hift.inference(mel)

    print(f"Python audio shape: {audio.shape}")
    print(
        f"Python audio stats: min={audio.min():.6f}, max={audio.max():.6f}, mean={audio.mean():.6f}, std={audio.std():.6f}"
    )

    # Save for comparison
    output_dir = Path("outputs/audio/hift_parity")
    output_dir.mkdir(parents=True, exist_ok=True)

    torchaudio.save(str(output_dir / "python_hift_output.wav"), audio.cpu(), 24000)
    print(f"\nSaved Python HiFT output to: {output_dir / 'python_hift_output.wav'}")

    # Load Rust native output if it exists
    rust_output_path = "outputs/audio/native_hift_output.wav"
    if os.path.exists(rust_output_path):
        print(f"\n=== Loading Rust native output: {rust_output_path} ===")
        rust_audio, rust_sr = torchaudio.load(rust_output_path)
        print(f"Rust audio shape: {rust_audio.shape}")
        print(
            f"Rust audio stats: min={rust_audio.min():.6f}, max={rust_audio.max():.6f}, mean={rust_audio.mean():.6f}, std={rust_audio.std():.6f}"
        )

        # Compare
        print("\n=== Comparison ===")
        min_len = min(audio.shape[1], rust_audio.shape[1])
        py_samples = audio[0, :min_len].cpu()
        rust_samples = rust_audio[0, :min_len]

        mae = (py_samples - rust_samples).abs().mean()
        max_diff = (py_samples - rust_samples).abs().max()
        correlation = torch.corrcoef(torch.stack([py_samples, rust_samples]))[0, 1]

        print(f"Samples compared: {min_len}")
        print(f"MAE: {mae:.6f}")
        print(f"Max diff: {max_diff:.6f}")
        print(f"Correlation: {correlation:.6f}")

        # Check for issues
        if audio.abs().max() < 0.1:
            print("\n⚠️  WARNING: Python output is very quiet (max < 0.1)")
        if rust_audio.abs().max() > 0.99:
            print("\n⚠️  WARNING: Rust output appears clipped (max > 0.99)")
        if rust_audio.mean().abs() > 0.1:
            print(
                f"\n⚠️  WARNING: Rust output has significant DC offset (mean = {rust_audio.mean():.4f})"
            )
        if correlation < 0.5:
            print(
                f"\n❌ CRITICAL: Low correlation ({correlation:.4f}) - outputs are fundamentally different"
            )

    # Also capture detailed intermediate values for debugging
    print("\n=== Capturing Intermediate Values ===")
    with torch.inference_mode():
        # F0 prediction
        with open("outputs/logs/test_hift_debug.log", "a") as log:
            log.write(f"Before f0_predictor calling with mel device: {mel.device}\n")
            # log.write(f"f0_predictor device check: {next(hift.f0_predictor.parameters()).device}\n") # Verify this

        # Explicitly ensure f0_predictor is on device
        hift.f0_predictor.to(device)

        f0 = hift.f0_predictor(mel)
        print(f"F0 predictor output: {tensor_stats(f0, 'f0')}")

        # Upsample F0
        s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        print(f"Upsampled F0 (s): {tensor_stats(s, 's')}")

        # Source module
        sine_merge, noise, uv = hift.m_source(s)
        print(f"Source sine_merge: {tensor_stats(sine_merge, 'sine_merge')}")
        print(f"Source noise: {tensor_stats(noise, 'noise')}")
        print(f"Source uv: {tensor_stats(uv, 'uv')}")

        # Transpose for decode
        s_source = sine_merge.transpose(1, 2)

        # Decode (detailed)
        # STFT of source
        s_stft_real, s_stft_imag = hift._stft(s_source.squeeze(1))
        print(f"Source STFT real: {tensor_stats(s_stft_real, 's_stft_real')}")
        print(f"Source STFT imag: {tensor_stats(s_stft_imag, 's_stft_imag')}")

        # Pre-conv
        x = hift.conv_pre(mel)
        print(f"After conv_pre: {tensor_stats(x, 'conv_pre_out')}")

    # Save intermediate tensors
    intermediates = {
        "input_mel": mel.cpu().contiguous(),
        "f0_output": f0.cpu().contiguous(),
        "source_s": s.cpu().contiguous(),
        "sine_merge": sine_merge.cpu().contiguous(),
        "noise": noise.cpu().contiguous(),
        "uv": uv.cpu().contiguous(),
        "s_stft_real": s_stft_real.cpu().contiguous(),
        "s_stft_imag": s_stft_imag.cpu().contiguous(),
        "conv_pre_out": x.cpu().contiguous(),
        "final_audio": audio.cpu().contiguous(),
    }
    debug_dir = Path("outputs/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_file(intermediates, str(debug_dir / "python_intermediates.safetensors"))
    print(
        f"\nSaved intermediate tensors to: {debug_dir / 'python_intermediates.safetensors'}"
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"Python HiFT audio: {audio.shape}, range [{audio.min():.4f}, {audio.max():.4f}]"
    )
    if os.path.exists(rust_output_path):
        print(
            f"Rust native audio: {rust_audio.shape}, range [{rust_audio.min():.4f}, {rust_audio.max():.4f}]"
        )
        if rust_audio.abs().max() > 0.99:
            print(
                "\n🚨 The Rust output is CLIPPING. This explains the 'garbage' audio."
            )
            print("   The issue is likely in:")
            print("   1. Flow model producing wrong mel ranges")
            print("   2. HiFT vocoder implementation (conv weights, ISTFT, etc.)")
            print("   3. Numerical issues (overflow in exp(), wrong phase handling)")


if __name__ == "__main__":
    main()
