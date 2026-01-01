#!/usr/bin/env python3
"""
Matcha-TTS Component Parity Tests

This test suite verifies parity between Rust and Python implementations
of the Matcha-TTS components used in CosyVoice.

Parent Issue: #44
Sub-Issue: #47
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# For testing mel_spectrogram, we need to compare against the Python implementation
from cosyvoice.compat.matcha_compat import mel_spectrogram as py_mel_spectrogram


def test_mel_spectrogram_parity():
    """
    Test native Rust audio.rs mel_spectrogram against Python implementation.

    The Rust implementation uses rustfft/realfft, while Python uses torch.stft.
    We expect L1 error < 1e-4 for identical inputs.
    """
    print("\n=== test_mel_spectrogram_parity ===")

    # Generate test audio signal (1 second of various frequencies)
    sample_rate = 24000
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))

    # Mix of frequencies for comprehensive test
    audio = (
        0.3 * torch.sin(2 * np.pi * 440 * t)  # A4
        + 0.3 * torch.sin(2 * np.pi * 880 * t)  # A5
        + 0.2 * torch.sin(2 * np.pi * 220 * t)  # A3
        + 0.1 * torch.randn_like(t)  # Noise
    )

    # Python mel spectrogram
    print("Computing Python mel spectrogram...")
    py_mel = py_mel_spectrogram(
        audio.unsqueeze(0),
        n_fft=1024,
        num_mels=80,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=8000,
        center=True,
    )

    print(f"Python mel shape: {py_mel.shape}")
    print(
        f"Python mel stats: min={py_mel.min():.4f}, max={py_mel.max():.4f}, mean={py_mel.mean():.4f}"
    )

    # Save test data for Rust
    test_data = {
        "test_audio": audio.contiguous(),
        "expected_mel": py_mel.contiguous(),
    }

    test_path = Path("tests/mel_parity_test.safetensors")
    save_file(test_data, str(test_path))
    print(f"Saved test data to {test_path}")

    # If Rust output exists, compare
    rust_output_path = Path("tests/mel_parity_rust_output.safetensors")
    if rust_output_path.exists():
        print("\nComparing with Rust output...")
        rust_data = load_file(str(rust_output_path))
        rust_mel = rust_data["rust_mel"]

        # Truncate to same length
        min_len = min(py_mel.shape[-1], rust_mel.shape[-1])
        py_mel_crop = py_mel[..., :min_len]
        rust_mel_crop = rust_mel[..., :min_len]

        l1_error = (py_mel_crop - rust_mel_crop).abs().mean().item()
        max_error = (py_mel_crop - rust_mel_crop).abs().max().item()

        print(f"L1 error: {l1_error:.6f}")
        print(f"Max error: {max_error:.6f}")

        if l1_error < 1e-4:
            print("✅ PASS: Mel spectrogram parity verified!")
            return True
        else:
            print("❌ FAIL: L1 error exceeds threshold")
            return False
    else:
        print(f"⚠️  Rust output not found at {rust_output_path}")
        print("   Run Rust mel parity test first, then re-run this test.")
        return None


def test_sinusoidal_embedding_parity():
    """
    Test timestep embedding (sinusoidal positional embedding).

    This is used in DiT for timestep conditioning.
    L1 error should be < 1e-5 (exact math).
    """
    print("\n=== test_sinusoidal_embedding_parity ===")

    # Standard sinusoidal embedding formula
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute sinusoidal embedding for timesteps."""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

    # Test with various timesteps
    t_values = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    dim = 256

    py_emb = sinusoidal_embedding(t_values, dim)
    print(f"Python embedding shape: {py_emb.shape}")
    print(f"Python embedding stats: min={py_emb.min():.6f}, max={py_emb.max():.6f}")

    # Save for Rust comparison
    test_data = {
        "t_values": t_values.contiguous(),
        "expected_emb": py_emb.contiguous(),
    }
    save_file(test_data, "tests/sinusoidal_emb_test.safetensors")
    print("Saved test data to tests/sinusoidal_emb_test.safetensors")

    # If Rust output exists, compare
    rust_path = Path("tests/sinusoidal_emb_rust_output.safetensors")
    if rust_path.exists():
        rust_data = load_file(str(rust_path))
        rust_emb = rust_data["rust_emb"]

        l1_error = (py_emb - rust_emb).abs().mean().item()
        print(f"L1 error: {l1_error:.6f}")

        if l1_error < 1e-5:
            print("✅ PASS: Sinusoidal embedding parity verified!")
            return True
        else:
            print("❌ FAIL: L1 error exceeds threshold")
            return False
    else:
        print("⚠️  Rust output not found. Run Rust test first.")
        return None


def test_snake_activation_parity():
    """
    Test Snake activation function used in HiFT vocoder.

    Snake(x) = x + (1/alpha) * sin(alpha * x)^2
    L1 error should be < 1e-5.
    """
    print("\n=== test_snake_activation_parity ===")

    # Snake activation
    def snake_activation(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Snake activation: x + (1/alpha) * sin(alpha * x)^2"""
        return x + (1 / alpha) * torch.sin(alpha * x).pow(2)

    # Test inputs
    x = torch.randn(1, 256, 100)  # [Batch, Channels, Time]
    alpha = torch.ones(1, 256, 1) * 1.0  # Default alpha

    py_out = snake_activation(x, alpha)
    print(f"Python Snake output shape: {py_out.shape}")
    print(
        f"Stats: min={py_out.min():.6f}, max={py_out.max():.6f}, mean={py_out.mean():.6f}"
    )

    # Save for comparison
    test_data = {
        "input_x": x.contiguous(),
        "alpha": alpha.contiguous(),
        "expected_out": py_out.contiguous(),
    }
    save_file(test_data, "tests/snake_activation_test.safetensors")
    print("Saved test data to tests/snake_activation_test.safetensors")

    # If Rust output exists, compare
    rust_path = Path("tests/snake_activation_rust_output.safetensors")
    if rust_path.exists():
        rust_data = load_file(str(rust_path))
        rust_out = rust_data["rust_out"]

        l1_error = (py_out - rust_out).abs().mean().item()
        print(f"L1 error: {l1_error:.6f}")

        if l1_error < 1e-5:
            print("✅ PASS: Snake activation parity verified!")
            return True
        else:
            print("❌ FAIL: L1 error exceeds threshold")
            return False
    else:
        print("⚠️  Rust output not found. Run Rust test first.")
        return None


def test_flow_matching_parity():
    """
    Test full Flow Matching (ConditionalCFM) parity.

    This is tested more extensively in compare_flow.py.
    Here we just do a quick sanity check.
    """
    print("\n=== test_flow_matching_parity ===")

    rust_dump_path = Path("rust/server/rust_flow_debug.safetensors")
    if not rust_dump_path.exists():
        print(f"⚠️  Rust flow dump not found at {rust_dump_path}")
        print("   Run test_native binary first to generate flow debug output.")
        return None

    print(f"Loading Rust flow dump from {rust_dump_path}...")
    rust_dump = load_file(str(rust_dump_path))

    rust_out = rust_dump.get("flow_output")
    if rust_out is None:
        print("⚠️  flow_output not found in dump")
        return None

    print(f"Rust flow output shape: {rust_out.shape}")
    print(
        f"Stats: min={rust_out.min():.4f}, max={rust_out.max():.4f}, mean={rust_out.mean():.4f}"
    )

    # Check for reasonable values
    if rust_out.abs().max() < 100 and rust_out.std() > 0.01:
        print(
            "✅ PASS: Flow output looks reasonable (detailed parity in compare_flow.py)"
        )
        return True
    else:
        print("❌ FAIL: Flow output has suspicious statistics")
        return False


def run_all_tests():
    """Run all parity tests and report results."""
    print("=" * 60)
    print("MATCHA-TTS COMPONENT PARITY TESTS")
    print("=" * 60)

    results = {}

    # Run each test
    results["mel_spectrogram"] = test_mel_spectrogram_parity()
    results["sinusoidal_embedding"] = test_sinusoidal_embedding_parity()
    results["snake_activation"] = test_snake_activation_parity()
    results["flow_matching"] = test_flow_matching_parity()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results.items():
        if result is True:
            status = "✅ PASS"
            passed += 1
        elif result is False:
            status = "❌ FAIL"
            failed += 1
        else:
            status = "⚠️  SKIP"
            skipped += 1
        print(f"  {name}: {status}")

    print()
    print(f"Passed: {passed}, Failed: {failed}, Skipped: {skipped}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
