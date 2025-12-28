import os
import sys

import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Mock ConditionalCFM to avoid matcha dependency
class MockConditionalCFM:
    def __init__(self, estimator, inference_cfg_rate=0.7):
        self.estimator = estimator
        self.inference_cfg_rate = inference_cfg_rate
        self.t_scheduler = "cosine"

    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        z = (
            torch.ones_like(mu).to(mu.device).to(mu.dtype) * 0.1 * temperature
        )  # Use deterministic "noise"

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        return self.solve_euler(
            z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond
        )

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            # CFG
            x_in = torch.cat([x, x], dim=0)
            mask_in = torch.cat([mask, mask], dim=0)
            mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
            t_in = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)
            spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
            cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)

            dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

            dphi_dt1, dphi_dt2 = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = (
                1.0 + self.inference_cfg_rate
            ) * dphi_dt1 - self.inference_cfg_rate * dphi_dt2

            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x


import cosyvoice_rust_backend

from cosyvoice.flow.DiT.dit import DiT


def verify_flow():
    print("Initializing models...")
    device = torch.device("cpu")
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"

    # Initialize Python Model
    dit_py = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80).to(
        device
    )
    dit_py.eval()

    cfm_py = MockConditionalCFM(estimator=dit_py, inference_cfg_rate=0.7)

    # Prepare dummy weights
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        import json

        with open(config_path, "w") as f:
            json.dump(
                {"dim": 1024, "depth": 22, "heads": 16, "dim_head": 64, "mel_dim": 80},
                f,
            )

    safetensors_path = os.path.join(model_dir, "model.safetensors")
    print(f"Creating dummy model weights at {safetensors_path}...")
    import safetensors.torch

    weights = {}

    for name, param in dit_py.named_parameters():
        rn = name
        # AdaLayerNormZero mapping
        if ".attn_norm.norm." in name:
            rn = rn.replace(".attn_norm.norm.", ".attn_norm.")
        if ".ff.ff.0.0." in name:
            rn = name.replace(".ff.ff.0.0.", ".ff.project_in.")
        elif ".ff.ff.2." in name:
            rn = name.replace(".ff.ff.2.", ".ff.project_out.")
        if "norm_out.norm." in name:
            rn = rn.replace("norm_out.norm.", "norm_out.")

        # TimestepEmbedding mapping
        if "time_embed.time_mlp.0." in name:
            rn = rn.replace("time_embed.time_mlp.0.", "time_embed.linear_1.")
        elif "time_embed.time_mlp.2." in name:
            rn = rn.replace("time_embed.time_mlp.2.", "time_embed.linear_2.")

        weights[f"flow.{rn}"] = param.data

    # Add missing mandatory weights for Rust
    for i in range(22):
        for suffix in ["attn_norm.weight", "ff_norm.weight"]:
            k = f"flow.transformer_blocks.{i}.{suffix}"
            if k not in weights:
                weights[k] = torch.ones(1024)
        for suffix in ["attn_norm.linear.weight", "attn_norm.linear.bias"]:
            k = f"flow.transformer_blocks.{i}.{suffix}"
            if k not in weights:
                weights[k] = (
                    torch.randn(6144, 1024) if "weight" in k else torch.zeros(6144)
                )

    # FINAL NORM_OUT
    weights["flow.norm_out.weight"] = torch.ones(1024)
    weights["flow.norm_out.linear.weight"] = torch.randn(6144, 1024)
    weights["flow.norm_out.linear.bias"] = torch.zeros(6144)

    # Input Proj
    weights["flow.input_embed.proj.weight"] = torch.randn(1024, 320)
    weights["flow.input_embed.proj.bias"] = torch.randn(1024)

    safetensors.torch.save_file(weights, safetensors_path)

    # NOW initialize Rust Model
    print("Loading FlowRust...")
    flow_rust = cosyvoice_rust_backend.FlowRust(model_dir)

    # Patch Python Model
    print("Patching Python DiT...")

    # Run inference...
    batch_size = 1
    seq_len = 10
    mu = (
        torch.linspace(-1, 1, batch_size * 80 * seq_len)
        .reshape(batch_size, 80, seq_len)
        .to(device)
    )
    mask = torch.ones(batch_size, 1, seq_len).to(device)
    spks = torch.linspace(-1, 1, batch_size * 80).reshape(batch_size, 80).to(device)
    cond = (
        torch.linspace(-1, 1, batch_size * 80 * seq_len)
        .reshape(batch_size, 80, seq_len)
        .to(device)
    )
    n_timesteps = 2
    temperature = 1.0

    print("Running PyTorch inference...")
    with torch.no_grad():
        output_py = cfm_py.forward(
            mu=mu,
            mask=mask,
            n_timesteps=n_timesteps,
            temperature=temperature,
            spks=spks,
            cond=cond,
        ).numpy()

    print("Running Rust inference...")
    output_rust = flow_rust.inference(
        mu.numpy(), mask.numpy(), n_timesteps, temperature, spks.numpy(), cond.numpy()
    )

    print(f"Py shape: {output_py.shape}, Rust shape: {output_rust.shape}")

    if output_py.shape != output_rust.shape:
        if output_py.shape[2] == 80 and output_rust.shape[1] == 80:
            output_py = output_py.transpose(0, 2, 1)

    l1_error = np.mean(np.abs(output_py - output_rust))
    print(f"L1 Error: {l1_error}")

    if l1_error < 1e-4:
        print("Verification SUCCESSFUL!")
    else:
        print("Verification FAILED!")


if __name__ == "__main__":
    verify_flow()
