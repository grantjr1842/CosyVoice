#!/usr/bin/env python3
"""
Extract intermediate tensors from Python HiFT for comparison with Rust.
"""

import os
import sys
from pathlib import Path

import torch
import torchaudio

# Add project root and third_party to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "Matcha-TTS"))

from hyperpyyaml import load_hyperpyyaml
from safetensors.torch import save_file


def tensor_stats(t: torch.Tensor, name: str) -> dict:
    """Compute statistics for a tensor."""
    t_flat = t.float().flatten().detach().cpu()
    return {
        f"{name}_shape": list(t.shape),
        f"{name}_dtype": str(t.dtype),
        f"{name}_min": float(t_flat.min().item()),
        f"{name}_max": float(t_flat.max().item()),
        f"{name}_mean": float(t_flat.mean().item()),
        f"{name}_std": float(t_flat.std().item()) if t_flat.numel() > 1 else 0.0,
        f"{name}_l2_norm": float(torch.norm(t_flat, p=2).item()),
        f"{name}_first_10": t_flat[:10].tolist()
        if t_flat.numel() >= 10
        else t_flat.tolist(),
        f"{name}_last_10": t_flat[-10:].tolist() if t_flat.numel() >= 10 else [],
    }


def main():
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"

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

    # Load HiFT
    print("Loading HiFT model...")
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

    # Create synthetic mel spectrogram for testing
    # Using values similar to actual mel output range
    torch.manual_seed(42)
    batch_size = 1
    mel_dim = 80
    mel_len = 156  # ~3 seconds at 24kHz/480 hop

    # Typical mel values are in the range -10 to +2 (log mel)
    mel = torch.randn(batch_size, mel_dim, mel_len, device=device) * 2 - 5
    print(f"Input mel shape: {mel.shape}")
    print(
        f"Input mel stats: min={mel.min():.4f}, max={mel.max():.4f}, mean={mel.mean():.4f}"
    )

    # Hook to capture intermediate values
    intermediates = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        intermediates[f"{name}_output_{i}"] = o.detach().clone().cpu()
            elif isinstance(output, torch.Tensor):
                intermediates[f"{name}_output"] = output.detach().clone().cpu()

        return hook

    # Register hooks
    hooks = []
    hooks.append(hift.f0_predictor.register_forward_hook(hook_fn("f0_predictor")))
    hooks.append(hift.m_source.register_forward_hook(hook_fn("m_source")))

    # Run forward
    print("\nRunning HiFT inference...")
    with torch.inference_mode():
        audio, cache_source = hift.inference(mel)

    print(f"\nOutput audio shape: {audio.shape}")
    print(
        f"Output audio stats: min={audio.min():.4f}, max={audio.max():.4f}, mean={audio.mean():.4f}, std={audio.std():.4f}"
    )

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print intermediate stats
    print("\n=== Intermediate Tensor Stats ===")
    for k, v in intermediates.items():
        stats = tensor_stats(v, k)
        print(f"\n{k}:")
        print(f"  Shape: {stats[f'{k}_shape']}")
        print(f"  Min: {stats[f'{k}_min']:.6f}, Max: {stats[f'{k}_max']:.6f}")
        print(f"  Mean: {stats[f'{k}_mean']:.6f}, Std: {stats[f'{k}_std']:.6f}")

    # Also manually capture key internal values
    print("\n=== Detailed F0 and Source Analysis ===")
    with torch.inference_mode():
        # F0 predictor output
        f0 = hift.f0_predictor(mel)
        f0_clipped = f0[:, :, : mel.shape[2]]
        print(f"\nF0 predictor output shape: {f0_clipped.shape}")
        print(
            f"F0 stats: min={f0_clipped.min():.4f}, max={f0_clipped.max():.4f}, mean={f0_clipped.mean():.4f}"
        )

        # Upsample F0
        upsample_scale = 480  # 8*5*3*4
        b, c, l = f0_clipped.shape
        s = (
            f0_clipped.unsqueeze(3)
            .repeat(1, 1, 1, upsample_scale)
            .reshape(b, c, l * upsample_scale)
        )
        print(f"\nUpsampled F0 shape: {s.shape}")
        print(
            f"Upsampled F0 stats: min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}"
        )

        # Source module
        s_source, noise, uv = hift.m_source(s)
        print(f"\nSource output shape: {s_source.shape}")
        print(
            f"Source stats: min={s_source.min():.4f}, max={s_source.max():.4f}, mean={s_source.mean():.4f}, std={s_source.std():.4f}"
        )
        print(f"UV stats: min={uv.min():.4f}, max={uv.max():.4f}, mean={uv.mean():.4f}")

        # Run full decode
        audio_final = hift.decode(mel, s_source)
        print(f"\nFinal audio shape: {audio_final.shape}")
        print(
            f"Final audio stats: min={audio_final.min():.4f}, max={audio_final.max():.4f}, mean={audio_final.mean():.4f}, std={audio_final.std():.4f}"
        )

        # Check for clipping percentage
        clipping_threshold = 0.99
        clipping_count = (
            (audio_final.abs() > clipping_threshold).float().sum()
            / audio_final.numel()
            * 100
        ).item()
        print(f"\nClipping (>{clipping_threshold}): {clipping_count:.2f}% of samples")

    # Save tensors for Rust comparison
    save_dict = {
        "input_mel": mel.cpu().contiguous(),
        "f0_output": f0_clipped.cpu().contiguous(),
        "upsampled_f0": s.cpu().contiguous(),
        "source_output": s_source.cpu().contiguous(),
        "audio_output": audio_final.cpu().contiguous(),
    }
    for k, v in intermediates.items():
        save_dict[k] = v.contiguous()

    output_path = "tests/hift_debug_tensors.safetensors"
    save_file(save_dict, output_path)
    print(f"\nSaved debug tensors to: {output_path}")

    # Also save audio
    audio_path = "tests/python_hift_debug_audio.wav"
    torchaudio.save(audio_path, audio_final.cpu().squeeze(0), 24000)
    print(f"Saved audio to: {audio_path}")


if __name__ == "__main__":
    main()
