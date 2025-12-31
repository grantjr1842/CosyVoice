import os
import sys

import torch
from safetensors.torch import load_file

# Add project root to path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("third_party/Matcha-TTS"))

from cosyvoice.flow.DiT.dit import DiT


def tensor_stats(t, name):
    print(
        f"{name}: shape={t.shape}, min={t.min():.6f}, max={t.max():.6f}, mean={t.mean():.6f}"
    )


def main():
    device = torch.device("cpu")
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"
    flow_path = os.path.join(model_dir, "flow.safetensors")

    # 1. Instantiate DiT
    print("Instantiating DiT...")
    dit = DiT(
        dim=1024,
        depth=22,
        heads=16,
        dim_head=64,
        ff_mult=2,
        mel_dim=80,
        mu_dim=80,
        spk_dim=80,
        out_channels=80,
        static_chunk_size=50,
        num_decoding_left_chunks=-1,
    )
    dit.to(device)
    dit.eval()

    # 2. Load Weights
    print("Loading weights...")
    state_dict = load_file(flow_path)

    # Check keys to find prefix
    keys = list(state_dict.keys())
    # Expect "decoder.estimator." prefix based on config structure
    prefix = "decoder.estimator."

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix) :]
            new_state_dict[new_key] = v

    if not new_state_dict:
        print("Error: No keys matched prefix 'decoder.estimator.'.")
        # Try finding prefix
        print("First 5 keys:", keys[:5])
        return

    print("Loading extracted state dict into DiT...")
    keys_missing, keys_unexpected = dit.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", keys_missing)
    print("Unexpected keys:", keys_unexpected)

    # 3. Check proj_out stats
    print("\n--- Model Debug ---")
    tensor_stats(dit.proj_out.bias, "proj_out.bias")
    tensor_stats(dit.proj_out.weight, "proj_out.weight")

    # 4. Load Rust inputs for parity check
    rust_dump_path = "rust/server/rust_flow_debug.safetensors"
    if os.path.exists(rust_dump_path):
        print("\nLoading Rust inputs...")
        rust_dump = load_file(rust_dump_path)
        x_init = rust_dump["x_init"]
        mask = rust_dump["mask"]
        mu = rust_dump["mu"]
        spks = rust_dump["spks"]
        cond = rust_dump["cond"]

        # Prepare inputs for DiT (Step 0)
        t_curr = 0.0
        # DiT expects t as tensor of shape [B]?
        # In DiT.forward: if t.ndim == 0: t = t.repeat(batch)
        t = torch.tensor([t_curr, t_curr], device=device).float()

        x_in = torch.cat([x_init, x_init], dim=0)
        # mask: [1, 120] -> [2, 1, 120]
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        mask_in = torch.cat([mask, mask], dim=0)

        mu_zero = torch.zeros_like(mu)
        mu_in = torch.cat([mu, mu_zero], dim=0)

        spks_zero = torch.zeros_like(spks)  # [1, 80]
        spks_in = torch.cat([spks, spks_zero], dim=0)  # [2, 80]

        cond_zero = torch.zeros_like(cond)
        cond_in = torch.cat([cond, cond_zero], dim=0)

        print("\nRunning DiT forward (Step 0)...")
        with torch.inference_mode():
            # forward: x, mask, mu, t, spks, cond
            v = dit(x_in, mask_in, mu_in, t, spks_in, cond_in)
            tensor_stats(v, "DiT_output_step0")

            # v should be comparable to Rust DIT FINAL (if we had saved it).
            # But mostly we check if mean is -21 or ~0.


if __name__ == "__main__":
    main()
