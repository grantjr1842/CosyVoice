#!/usr/bin/env python3
"""
Focus on the first step: Compare InputEmbedding output between Rust and Python.
"""

import os
import sys

import torch
from hyperpyyaml import load_hyperpyyaml
from safetensors.torch import load_file

sys.path.insert(0, os.path.abspath("."))


def tensor_stats(t, name=""):
    """Print detailed tensor statistics."""
    if t.numel() == 0:
        print(f"{name}: EMPTY tensor")
        return
    print(
        f"{name}: shape={t.shape}, min={t.min():.6f}, max={t.max():.6f}, mean={t.mean():.6f}"
    )
    if t.numel() >= 5:
        t_flat = t.flatten()
        print(f"  First 5: {t_flat[:5].tolist()}")
    if t.shape[-1] > 64:
        # Assuming layout [B, T, D]
        # Check T=0, D=64..69
        slice_vals = t[0, 0, 64:69]
        print(f"  Chan 64-68 (t=0): {slice_vals.tolist()}")


def main():
    print("=" * 60)
    print("INPUT EMBEDDING COMPARISON")
    print("=" * 60)

    # Load Rust debug dump
    rust_dump = load_file("rust/server/rust_flow_debug.safetensors")

    x_init = rust_dump["x_init"]  # [1, 80, 171]
    mu = rust_dump["mu"]  # [1, 80, 171]
    spks = rust_dump["spks"]  # [1, 80]
    cond = rust_dump["cond"]  # [1, 80, 171]

    device = torch.device("cpu")

    print("\n--- Input Shapes ---")
    print(f"x_init: {x_init.shape}")

    # Load Python model
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"
    with open(os.path.join(model_dir, "cosyvoice3.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None, "hift": None})

    # Load weights
    flow_weights = load_file(os.path.join(model_dir, "flow.safetensors"))
    configs["flow"].load_state_dict(flow_weights, strict=False)

    estimator = configs["flow"].decoder.estimator
    estimator.to(device)
    estimator.eval()

    # === Test InputEmbedding directly ===
    print("\n" + "=" * 60)
    print("STEP 1: InputEmbedding (Python side)")
    print("=" * 60)

    x_py = x_init.transpose(1, 2)  # [1, 171, 80]
    mu_py = mu.transpose(1, 2)  # [1, 171, 80]
    cond_py = cond.transpose(1, 2)  # [1, 171, 80]
    spks_py = spks.unsqueeze(1)  # [1, 1, 1, 80] -> [1, 1, 80]

    with torch.inference_mode():
        # Check proj dimensions and output
        # Reconstruct input to proj
        cat_in = torch.cat(
            [
                x_py,
                cond_py,
                mu_py,
                spks_py.squeeze(1).unsqueeze(1).repeat(1, x_py.shape[1], 1),
            ],
            dim=-1,
        )
        print(f"Proj Input Shape: {cat_in.shape}")

        proj_out = estimator.input_embed.proj(cat_in)
        tensor_stats(proj_out, "Python Proj Output")

        # Check ConvPosEmbed
        pos_out = estimator.input_embed.conv_pos_embed(proj_out)
        tensor_stats(pos_out, "Python ConvPosEmbed Output")

        # Check Conv Weights
        print("\n--- Conv Weights Stats ---")
        conv1_w = estimator.input_embed.conv_pos_embed.conv1[0].weight
        conv1_b = estimator.input_embed.conv_pos_embed.conv1[0].bias
        conv2_w = estimator.input_embed.conv_pos_embed.conv2[0].weight
        conv2_b = estimator.input_embed.conv_pos_embed.conv2[0].bias
        tensor_stats(conv1_w, "conv1.0.weight")
        tensor_stats(conv1_b, "conv1.0.bias")
        tensor_stats(conv2_w, "conv2.0.weight")
        tensor_stats(conv2_b, "conv2.0.bias")

    print("\n" + "=" * 60)
    print("RUST COMPARISON")
    print("=" * 60)
    print("Rust Proj Mean: -0.0467")
    print("Rust Pos Mean: 82.93")


if __name__ == "__main__":
    main()
