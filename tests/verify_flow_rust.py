import json
import logging
import os
import sys

import numpy as np
import torch

# Add project root to path
sys.path.append(os.getcwd())

from safetensors.torch import save_file

from cosyvoice.flow.DiT.dit import DiT

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def verify():
    logging.basicConfig(level=logging.INFO)
    print("Initializing models...")

    # Config matching Rust DiTBlock::new(..., 1024, 16, 64) and FeedForward::new(..., 1024, 2)
    dim = 1024
    depth = 22
    heads = 16
    dim_head = 64
    mel_dim = 80
    ff_mult = 2  # Match Rust flow.rs:137

    # 1. Initialize Python DiT
    dit_py = DiT(
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        ff_mult=ff_mult,
        mel_dim=mel_dim,
        mu_dim=mel_dim,
        spk_dim=mel_dim,
    )
    dit_py.eval()

    # 2. Save weights and config for Rust
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.safetensors")
    config_path = os.path.join(model_dir, "config.json")

    print(f"Saving dummy model weights and config to {model_dir}...")
    state_dict = dit_py.state_dict()

    # Remap name logic
    rust_state_dict = {}
    for name, tensor in state_dict.items():
        new_name = name
        # Time embed
        if "time_embed.time_mlp.0" in name:
            new_name = name.replace("time_embed.time_mlp.0", "time_embed.linear_1")
        elif "time_embed.time_mlp.2" in name:
            new_name = name.replace("time_embed.time_mlp.2", "time_embed.linear_2")
        # FeedForward
        elif "ff.ff.0.0" in name:
            new_name = name.replace("ff.ff.0.0", "ff.project_in")
        elif "ff.ff.2" in name:
            new_name = name.replace("ff.ff.2", "ff.project_out")

        rust_state_dict[f"flow.{new_name}"] = tensor.contiguous()

    # Add missing LayerNorm weights that Rust expects
    for i in range(depth):
        rust_state_dict[f"flow.transformer_blocks.{i}.attn_norm.weight"] = torch.ones(
            dim
        )
        rust_state_dict[f"flow.transformer_blocks.{i}.ff_norm.weight"] = torch.ones(dim)

    save_file(rust_state_dict, model_path)

    config = {
        "dim": dim,
        "depth": depth,
        "heads": heads,
        "dim_head": dim_head,
        "mel_dim": mel_dim,
    }
    with open(config_path, "w") as f:
        json.dump(config, f)

    # 3. Initialize Rust Flow (PyO3)
    print(
        "Loading FlowRust (PyO3)... %s" % os.path.abspath("./cosyvoice_rust_backend.so")
    )
    try:
        sys.path.append(".")
        import cosyvoice_rust_backend
    except ImportError as e:
        print(f"Error: Could not import cosyvoice_rust_backend: {e}")
        return

    flow_rust = cosyvoice_rust_backend.FlowRust(model_dir)

    # 4. Inputs
    B = 1
    N = 10

    # Matching Rust CFM loop logic:
    # x_init = 0.1 * temperature
    temperature = 1.0
    x_init = torch.ones(B, mel_dim, N) * 0.1 * temperature

    mask = torch.ones(B, 1, N)
    mu = torch.randn(B, mel_dim, N)
    t_val = 0.0  # Initial t in Rust loop
    t = torch.tensor([t_val])
    spks = torch.randn(B, mel_dim)
    cond = torch.randn(B, mel_dim, N)

    # 5. Python Forward (Manual CFG to match Rust Loop)
    print("Running PyTorch reference (manual CFG step)...")
    with torch.no_grad():
        input_x = torch.cat([x_init, x_init], dim=0)
        input_mask = torch.cat([mask, mask], dim=0)
        input_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        input_t = torch.tensor([t_val, t_val])
        input_spks = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        input_cond = torch.cat([cond, torch.zeros_like(cond)], dim=0)

        v = dit_py(input_x, input_mask, input_mu, input_t, input_spks, input_cond)

        v1, v2 = v.chunk(2, dim=0)
        cfg_strength = 0.7
        v_cfg = (1.0 + cfg_strength) * v1 - cfg_strength * v2

        expected_out = x_init + v_cfg * 1.0

    # 6. Rust Forward
    print("Running Rust inference (1 step)...")
    out_rust = flow_rust.inference(
        mu.numpy(),
        mask.numpy(),
        1,  # n_timesteps
        temperature,
        spks.numpy(),
        cond.numpy(),
    )
    out_rust = torch.from_numpy(out_rust)

    # 7. Comparison
    print(f"Py shape: {expected_out.shape}, Rust shape: {out_rust.shape}")

    l1 = (expected_out - out_rust).abs().mean().item()
    print(f"L1 Error: {l1}")

    if l1 < 1e-4:
        print("Verification SUCCESSFUL!")
    else:
        print("Verification FAILED!")
        print("Py first 5:", expected_out.flatten()[:5])
        print("Rust first 5:", out_rust.flatten()[:5])


if __name__ == "__main__":
    verify()
