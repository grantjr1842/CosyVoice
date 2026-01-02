#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file


DEFAULT_PY_PATH = "outputs/debug/python_intermediates.safetensors"
DEFAULT_RUST_PATH = "outputs/debug/rust_intermediates.safetensors"

DEFAULT_KEYS = [
    "input_mel",
    "f0_output",
    "f0_layer0",
    "f0_layer1",
    "f0_layer2",
    "f0_layer3",
    "f0_layer4",
    "f0_classifier_pre_abs",
    "source_s",
    "sine_waves",
    "sine_rad_down",
    "sine_merge",
    "noise",
    "uv",
    "rand_ini",
    "sine_noise_cache",
    "source_noise_cache",
    "s_stft_real",
    "s_stft_imag",
    "conv_pre_out",
    "upsample_0_pre",
    "upsample_0_out",
    "source_down_0_out",
    "source_resblock_0_out",
    "fusion_0_out",
    "resblock_0_0_out",
    "resblock_0_1_out",
    "resblock_0_2_out",
    "resblock_0_out",
    "upsample_1_pre",
    "upsample_1_out",
    "source_down_1_out",
    "source_resblock_1_out",
    "fusion_1_out",
    "resblock_1_0_out",
    "resblock_1_1_out",
    "resblock_1_2_out",
    "resblock_1_out",
    "upsample_2_pre",
    "upsample_2_out",
    "source_down_2_out",
    "source_resblock_2_out",
    "fusion_2_out",
    "resblock_2_0_out",
    "resblock_2_1_out",
    "resblock_2_2_out",
    "resblock_2_out",
    "post_lrelu_out",
    "conv_post_out",
    "magnitude",
    "phase",
    "istft_audio",
    "final_audio",
]


def align_tensors(py_t: torch.Tensor, rust_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    py_t = py_t.cpu().float()
    rust_t = rust_t.cpu().float()

    if py_t.ndim == 2 and rust_t.ndim == 3 and rust_t.shape[1] == 1:
        rust_t = rust_t.squeeze(1)
    if rust_t.ndim == 2 and py_t.ndim == 3 and py_t.shape[1] == 1:
        py_t = py_t.squeeze(1)

    if py_t.ndim == 3 and rust_t.ndim == 3:
        if (
            py_t.shape[1] != rust_t.shape[1]
            and py_t.shape[1] == rust_t.shape[2]
            and py_t.shape[2] == rust_t.shape[1]
        ):
            rust_t = rust_t.transpose(1, 2)

    return py_t, rust_t


def slice_to_match(py_t: torch.Tensor, rust_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if py_t.shape == rust_t.shape:
        return py_t, rust_t

    min_len = min(py_t.shape[-1], rust_t.shape[-1])
    py_t = py_t[..., :min_len]
    rust_t = rust_t[..., :min_len]
    return py_t, rust_t


def compare_tensor(name: str, py_t: torch.Tensor, rust_t: torch.Tensor, tolerance: float) -> None:
    print(f"\n--- {name} ---")
    py_t, rust_t = align_tensors(py_t, rust_t)

    if py_t.shape != rust_t.shape:
        print(f"Shape mismatch: python {tuple(py_t.shape)} vs rust {tuple(rust_t.shape)}")
        py_t, rust_t = slice_to_match(py_t, rust_t)
        print(f"Sliced to: {tuple(py_t.shape)}")

    diff = (py_t - rust_t).abs()
    mae = diff.mean().item()
    max_diff = diff.max().item()

    print(f"MAE: {mae:.6f}")
    print(f"Max diff: {max_diff:.6f}")
    print("MATCH" if mae < tolerance else "MISMATCH")

    if py_t.numel() > 10:
        flat_py = py_t.flatten()
        flat_rust = rust_t.flatten()
        if flat_py.std() > 0 and flat_rust.std() > 0:
            corr = torch.corrcoef(torch.stack([flat_py, flat_rust]))[0, 1].item()
            print(f"Correlation: {corr:.6f}")
        else:
            print("Correlation: skipped (constant vector)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HiFT intermediate tensors.")
    parser.add_argument(
        "--py",
        default=DEFAULT_PY_PATH,
        help="Path to python intermediates safetensors.",
    )
    parser.add_argument(
        "--rust",
        default=DEFAULT_RUST_PATH,
        help="Path to rust intermediates safetensors.",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=DEFAULT_KEYS,
        help="Keys to compare.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="MAE tolerance for match.",
    )
    args = parser.parse_args()

    py_path = Path(args.py)
    rust_path = Path(args.rust)

    if not py_path.exists() and args.py == DEFAULT_PY_PATH:
        alt = Path("rust") / DEFAULT_PY_PATH
        if alt.exists():
            py_path = alt
    if not rust_path.exists() and args.rust == DEFAULT_RUST_PATH:
        alt = Path("rust") / DEFAULT_RUST_PATH
        if alt.exists():
            rust_path = alt

    py_data = load_file(str(py_path))
    rust_data = load_file(str(rust_path))

    missing = [k for k in args.keys if k not in py_data or k not in rust_data]
    if missing:
        print(f"Missing keys: {', '.join(missing)}")

    for key in args.keys:
        if key not in py_data or key not in rust_data:
            continue
        compare_tensor(key, py_data[key], rust_data[key], args.tolerance)


if __name__ == "__main__":
    main()
