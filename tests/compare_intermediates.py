#!/usr/bin/env python3
import torch
from safetensors.torch import load_file


def compare(name, py_t, rust_t, tolerance=1e-4):
    print(f"\n--- Comparing {name} ---")
    py_t = py_t.cpu().float()
    rust_t = rust_t.cpu().float()

    # Handle Transpose (B, C, L) vs (B, L, C)
    if py_t.ndim == 3 and rust_t.ndim == 3:
        if (
            py_t.shape[1] != rust_t.shape[1]
            and py_t.shape[2] == rust_t.shape[1]
            and py_t.shape[1] == rust_t.shape[2]
        ):
            print(
                f"  Transposing Rust tensor from {rust_t.shape} to match Python {py_t.shape}"
            )
            rust_t = rust_t.transpose(1, 2)
        elif (
            py_t.shape[2] == 1
            and rust_t.shape[1] == 1
            and py_t.shape[1] == rust_t.shape[2]
        ):
            print(
                f"  Transposing Rust tensor {rust_t.shape} to match Python {py_t.shape}"
            )
            rust_t = rust_t.transpose(1, 2)

    # Check shapes
    if py_t.shape != rust_t.shape:
        print(f"SHAPE MISMATCH: Python {py_t.shape} vs Rust {rust_t.shape}")
        # Try to slice if close?
        min_len = min(py_t.shape[-1], rust_t.shape[-1])
        if py_t.ndim == 3:
            py_t = py_t[:, :, :min_len]
            rust_t = rust_t[:, :, :min_len]
        elif py_t.ndim == 2:
            py_t = py_t[:, :min_len]
            rust_t = rust_t[:, :min_len]
        print(f"  (Sliced to {min_len} for comparison)")

    diff = (py_t - rust_t).abs()
    mae = diff.mean()
    max_diff = diff.max()

    print(f"MAE: {mae:.6f}")
    print(f"Max Diff: {max_diff:.6f}")

    if mae < tolerance:
        print("✅ MATCH")
    else:
        print("❌ MISMATCH")

    # Check correlation for signals
    if py_t.numel() > 10:
        flat_py = py_t.flatten()
        flat_rust = rust_t.flatten()

        # Avoid correlation calculation on tiny or constant vectors
        if flat_py.std() > 0 and flat_rust.std() > 0:
            corr = torch.corrcoef(torch.stack([flat_py, flat_rust]))[0, 1]
            print(f"Correlation: {corr:.6f}")
        else:
            print("Skipping correlation (constant or zero vector)")


def main():
    py_path = "tests/hift_parity/python_intermediates.safetensors"
    rust_path = "rust_intermediates.safetensors"

    try:
        py_data = load_file(py_path)
        rust_data = load_file(rust_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # F0
    if "f0_output" in py_data and "f0" in rust_data:
        compare("F0", py_data["f0_output"], rust_data["f0"])

    # Upsampled F0
    if "source_s" in py_data and "upsampled_f0" in rust_data:
        compare("Upsampled F0 (s)", py_data["source_s"], rust_data["upsampled_f0"])

    # Source (Sine Merge)
    if "sine_merge" in py_data and "source" in rust_data:
        compare("Source Module Output", py_data["sine_merge"], rust_data["source"])

    # Final Audio
    if "final_audio" in py_data and "audio" in rust_data:
        compare("Audio", py_data["final_audio"], rust_data["audio"])


if __name__ == "__main__":
    main()
