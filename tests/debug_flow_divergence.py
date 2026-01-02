#!/usr/bin/env python3
"""
Deep Debug: Flow Divergence Analysis
Compares Rust and Python Flow implementations at each stage.
"""

import os
import sys

import torch
from hyperpyyaml import load_hyperpyyaml
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.abspath("."))


def tensor_stats(t, name=""):
    """Print detailed tensor statistics."""
    if t.numel() == 0:
        print(f"{name}: EMPTY tensor")
        return
    print(
        f"{name}: shape={t.shape}, min={t.min():.6f}, max={t.max():.6f}, mean={t.mean():.6f}, std={t.std():.6f}"
    )


def compare_tensors(rust_t, py_t, name):
    """Compare two tensors and report differences."""
    if rust_t.shape != py_t.shape:
        print(f"❌ {name}: SHAPE MISMATCH! Rust={rust_t.shape}, Python={py_t.shape}")
        return False

    rust_t = rust_t.float()
    py_t = py_t.float()

    diff = (rust_t - py_t).abs()
    mae = diff.mean().item()
    max_diff = diff.max().item()

    if mae < 1e-4:
        print(f"✅ {name}: MAE={mae:.6f}, Max={max_diff:.6f}")
        return True
    else:
        print(f"⚠️  {name}: MAE={mae:.6f}, Max={max_diff:.6f}")
        print(
            f"    Rust: min={rust_t.min():.6f}, max={rust_t.max():.6f}, mean={rust_t.mean():.6f}"
        )
        print(
            f"    Python: min={py_t.min():.6f}, max={py_t.max():.6f}, mean={py_t.mean():.6f}"
        )
        return False


def main():
    print("=" * 60)
    print("FLOW DIVERGENCE DEEP DEBUG")
    print("=" * 60)

    rust_dump_path = "rust/server/rust_flow_debug.safetensors"
    if not os.path.exists(rust_dump_path):
        print(f"Error: {rust_dump_path} not found.")
        return

    rust_dump = load_file(rust_dump_path)

    # Input tensors from Rust
    mu = rust_dump["mu"]
    mask = rust_dump["mask"]
    spks = rust_dump["spks"]
    cond = rust_dump["cond"]
    x_init = rust_dump["x_init"]
    rust_out = rust_dump["flow_output"]

    device = torch.device("cpu")

    print("\n--- Input Tensors from Rust ---")
    tensor_stats(mu, "mu")
    tensor_stats(mask, "mask")
    tensor_stats(spks, "spks")
    tensor_stats(cond, "cond")
    tensor_stats(x_init, "x_init")

    # Load Python model
    print("\n--- Loading Python Model ---")
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"
    with open(os.path.join(model_dir, "cosyvoice3.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None, "hift": None})

    decoder = configs["flow"].decoder
    decoder.to(device)
    decoder.eval()

    # Load weights
    flow_weights = load_file(os.path.join(model_dir, "flow.safetensors"))
    configs["flow"].load_state_dict(flow_weights, strict=False)
    print("Weights loaded.")

    # Get the estimator (DiT)
    estimator = decoder.estimator

    # === STEP 1: Test single DiT forward at t=0 ===
    print("\n" + "=" * 60)
    print("STEP 1: Single DiT Estimator Forward (t=0)")
    print("=" * 60)

    # Prepare inputs exactly as Rust does
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)  # [1, 1, T]

    t_curr = torch.tensor([0.0], device=device)

    # For CFG, batch is doubled
    x_in = torch.cat([x_init, x_init], dim=0)  # [2, 80, 171]
    mask_in = torch.cat([mask, mask], dim=0)  # [2, 1, 171]
    mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)  # [2, 80, 171]
    spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)  # [2, 80]
    cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)  # [2, 80, 171]
    t_in = torch.tensor([0.0, 0.0], device=device)

    print(f"x_in: {x_in.shape}")
    print(f"mask_in: {mask_in.shape}")
    print(f"mu_in: {mu_in.shape}")
    print(f"spks_in: {spks_in.shape}")
    print(f"t_in: {t_in.shape}")

    # Run DiT estimator once
    with torch.inference_mode():
        v_py = estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

    tensor_stats(v_py, "Python estimator output (t=0)")

    # === STEP 2: Check internal layers ===
    print("\n" + "=" * 60)
    print("STEP 2: Internal Layer Checks")
    print("=" * 60)

    # Check input_proj
    with torch.inference_mode():
        # Input embedding
        x_for_emb = x_in + mu_in + spks_in.unsqueeze(-1)
        x_embedded = estimator.in_block(x_for_emb)  # Input conv block
        tensor_stats(x_embedded, "input_block output")

        # Time embedding
        t_emb = estimator.time_embed(t_in)
        tensor_stats(t_emb, "time_embed output")

    # === STEP 3: First transformer block ===
    print("\n" + "=" * 60)
    print("STEP 3: First Transformer Block")
    print("=" * 60)

    # Run through first block manually
    with torch.inference_mode():
        # Transpose for transformer [B, D, T] -> [B, T, D]
        x_t = x_embedded.transpose(1, 2)

        # First block's norm
        block0 = estimator.blocks[0]

        # Apply adaln
        norm_x = block0.norm1(x_t)
        tensor_stats(norm_x, "block0 norm1 output")

        # Check adaln modulation
        modulation = block0.adaLN_modulation(t_emb)
        tensor_stats(modulation, "block0 adaLN_modulation output")

    # === STEP 4: Full solve_euler comparison ===
    print("\n" + "=" * 60)
    print("STEP 4: Full solve_euler Comparison")
    print("=" * 60)

    n_timesteps = 10
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=mu.dtype)
    if decoder.t_scheduler == "cosine":
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

    print(f"t_span: {t_span}")

    with torch.inference_mode():
        py_out = decoder.solve_euler(
            x=x_init, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond
        )

    print("\n--- Final Comparison ---")
    tensor_stats(rust_out, "Rust output")
    tensor_stats(py_out, "Python output")

    compare_tensors(rust_out, py_out, "Flow output")

    # === STEP 5: Save Python intermediate for Rust comparison ===
    print("\n" + "=" * 60)
    print("STEP 5: Saving Python intermediates for Rust comparison")
    print("=" * 60)

    intermediates = {
        "py_estimator_v_t0": v_py,
        "py_flow_output": py_out,
        "t_span": t_span,
    }
    save_file(intermediates, "tests/python_flow_debug.safetensors")
    print("Saved to tests/python_flow_debug.safetensors")


if __name__ == "__main__":
    main()
