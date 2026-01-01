import os
import sys

import torch
from hyperpyyaml import load_hyperpyyaml
from safetensors.torch import load_file

# Add project root to path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("third_party/Matcha-TTS"))
import types

from cosyvoice.flow.DiT import dit

# Mock cosyvoice.flow.transformer because config uses old path
mock_transformer = types.ModuleType("cosyvoice.flow.transformer")
mock_transformer.DiT = dit.DiT
sys.modules["cosyvoice.flow.transformer"] = mock_transformer


def tensor_stats(t, name):
    print(
        f"{name}: shape={t.shape}, min={t.min():.6f}, max={t.max():.6f}, mean={t.mean():.6f}"
    )


def main():
    rust_dump_path = "rust/server/rust_flow_debug.safetensors"
    if not os.path.exists(rust_dump_path):
        print(f"Error: {rust_dump_path} not found.")
        return

    print(f"Loading Rust dump from {rust_dump_path}...")
    rust_dump = load_file(rust_dump_path)

    mu = rust_dump["mu"]
    mask = rust_dump["mask"]
    spks = rust_dump["spks"]
    cond = rust_dump["cond"]
    x_init = rust_dump["x_init"]
    rust_out = rust_dump["flow_output"]

    device = torch.device("cpu")  # Rust test ran on CPU

    print("--- Rust Tensors ---")
    tensor_stats(mu, "mu")
    tensor_stats(mask, "mask")
    tensor_stats(spks, "spks")
    tensor_stats(cond, "cond")
    tensor_stats(x_init, "x_init")
    tensor_stats(rust_out, "rust_out")

    # Load Python model
    print("\nLoading Python model...")
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"
    with open(os.path.join(model_dir, "cosyvoice3.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None, "hift": None})

    decoder = configs["flow"].decoder
    decoder.to(device)
    decoder.eval()

    # LOAD WEIGHTS!
    print("Loading weights from flow.safetensors...")
    flow_weights = load_file(os.path.join(model_dir, "flow.safetensors"))
    # The weights might be prefixed with "decoder." or similar if saved from higher level.
    # Check keys.
    # In CosyVoice implementation, `CosyVoiceFlow` has `decoder`.
    # Let's see if we can load directly or need prefix handling.
    # Try strict=False first or check prefixes.

    # Based on rust loading:
    # Rust loads "decoder.estimator.proj_out.weight".
    # So headers have "decoder." prefix?
    # Or "flow.decoder."?
    # Python `configs['flow']` is `CosyVoiceFlow`. It contains `decoder` (ConditionalCFM).
    # If state_dict has "decoder.estimator...", then `configs['flow'].load_state_dict(flow_weights)` matches.

    try:
        configs["flow"].load_state_dict(flow_weights, strict=False)
        print("Weights loaded into 'flow' module.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("First 5 keys in dump:", list(flow_weights.keys())[:5])

    print("\n--- Model Debug ---")
    proj_bias = decoder.estimator.proj_out.bias
    tensor_stats(proj_bias, "proj_out.bias")
    tensor_stats(decoder.estimator.proj_out.weight, "proj_out.weight")

    # Ensure mask is [1, 1, T] if it is [1, T]
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)

    print("\nRunning Python Decoder...")

    n_timesteps = 10
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=mu.dtype)
    if decoder.t_scheduler == "cosine":
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

    with torch.inference_mode():
        # Note: solve_euler expects x (noise). In our case, x_init is already noise * temp.
        # But wait, Rust x_init = noise * temp.
        # Python solve_euler takes `x`.
        # So we pass x_init directly.

        py_out = decoder.solve_euler(
            x=x_init, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond
        )

    print("\n--- Comparison ---")
    tensor_stats(py_out, "py_out")

    diff = (py_out - rust_out).abs()
    print(f"MAE: {diff.mean():.6f}")
    print(f"Max Diff: {diff.max():.6f}")

    print("\n--- Model Debug ---")
    proj_bias = decoder.estimator.proj_out.bias
    tensor_stats(proj_bias, "proj_out.bias")
    tensor_stats(decoder.estimator.proj_out.weight, "proj_out.weight")

    print("\n--- Single Step Debug ---")
    # Verify Step 1 estimator output
    # x = x_init
    # t = 0
    t_curr = 0.0
    t_tensor = torch.tensor([t_curr, t_curr], device=device).float()

    # Prepare batch-doubled inputs like Rust
    x_in = torch.cat([x_init, x_init], dim=0)
    mask_in = torch.cat([mask, mask], dim=0)
    mu_zero = torch.zeros_like(mu)
    mu_in = torch.cat([mu, mu_zero], dim=0)

    spks_zero = torch.zeros_like(spks)
    spks_in = torch.cat([spks, spks_zero], dim=0)

    cond_zero = torch.zeros_like(cond)
    cond_in = torch.cat([cond, cond_zero], dim=0)

    with torch.inference_mode():
        v = decoder.estimator(x_in, mask_in, mu_in, t_tensor, spks_in, cond_in)
        tensor_stats(v, "estimator_step1_v")

        # Check v manually against what? We don't have Rust v.
        # But we can see if mean is weird.

    if diff.mean() < 1e-4:
        print("✅ Parity Verified!")
    else:
        print("❌ Parity Failed!")
        # If failed, we might want to step debug.


if __name__ == "__main__":
    main()
