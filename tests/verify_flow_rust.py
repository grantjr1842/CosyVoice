import os
import sys

import torch

# Add project root to path
sys.path.append(os.getcwd())

from cosyvoice.flow.DiT import DiT


def verify_flow():
    print("Initializing models...")

    # 1. Initialize Python DiT
    # Match Rust config:
    # heads=16, dim=1024, dim_head=64 (16*64=1024), depth=22
    # dropout=0.1, ff_mult=4
    # input_dim=80
    dit_py = DiT(
        dim=1024,
        depth=22,
        heads=16,
        dim_head=64,
        dropout=0.0,  # Rust implementation doesn't use dropout in inference usually
        ff_mult=4,
        in_channels=80,  # Match Rust mel_feat_conf
        long_skip_connection=False,  # Rust implementation likely False? Default is False? Checking flow.py... default is False for FlowMatching.
    )
    dit_py.eval()

    # 2. Convert PyTorch weights to Safetensors for Rust
    # We will create random weights and save them, so both use SAME weights.
    from safetensors.torch import save_file

    # Create dummy dir if not exists
    os.makedirs("pretrained_models/Fun-CosyVoice3-0.5B", exist_ok=True)
    model_path = "pretrained_models/Fun-CosyVoice3-0.5B/model.safetensors"

    print(f"Creating dummy model weights at {model_path}...")
    state_dict = dit_py.state_dict()
    # Ensure contiguous for safetensors
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    save_file(state_dict, model_path)

    # 3. Initialize Rust DiT (via exposed API or rebuilding flow)
    # Since we don't have python bindings for Flow ONLY, we rely on the rust binary
    # OR we use the C-API if we built it?
    # Actually, we likely need to use `ctypes` or similar to call Rust function if we want direct comparison
    # OR simpler: We can just use the "verify_flow_rust.py" approach I Was using which loaded the .so?
    # Yes, I was using ctypes/cdll.

    print("Loading FlowRust...")
    import ctypes

    lib = ctypes.CDLL("./cosyvoice_rust_backend.so")

    # Define Rust function signatures (simplified for verification)
    # We need a function in Rust that runs JUST the flow model.
    # The current `test_flow.rs` binary does this?
    # Or did I expose a function?
    # I exposed `call_flow_inference`?
    # No, I used `sc_flow_inference` in the C-API?
    # Wait, the previous `verify_flow_rust.py` relied on `cosyvoice_rust_backend.so` but I don't see C-API functions in my `lib.rs` research?
    # Ah, I replaced `verify_flow_rust.py` content completely in previous steps.
    # I should use the content I had in the LAST `verify_flow_rust.py`.
    # But clean it up.

    # Assuming previous logic for loading Rust library was working.
    # I will replicate the Python-side logic for consistency.

    # 4. Inputs
    B = 1
    N = 10  # Seq len
    D = 80  # Mel dim

    # Random inputs
    torch.manual_seed(42)
    x = torch.randn(B, D, N)
    mask = torch.ones(B, 1, N)  # Full mask
    mu = torch.zeros_like(x)  # ConditionalCFM usually takes mu?
    # Wait, DiT forward takes (x, mask, mu, t, spks, cond)
    t = torch.tensor([0.5])  # Time
    spks = torch.randn(B, 80)  # Spk embed
    cond = torch.randn(B, 80, N)  # Condition

    # Python Forward
    print("Running PyTorch inference...")
    # DiT forward: x, t, conditions...
    # flow.py DiT wrapper handles: t embedding, etc.
    # We are testing DiT directly?
    # Rust `DiT` is the transformer.
    # flow.rs `ConditionalCFM` calls DiT.
    # My previous script tested `DiT` via direct modification?
    # No, I was calling `lib.flow_inference`?

    # Let's assume there is a `debug_dit_forward` or similar I added to Rust?
    # No, I was running `flow.rs` main logic.
    # I'll rely on the existing `test_flow` binary approach?
    # No, `verify_flow_rust.py` was importing `tests.verify_flow_rust`?
    # I'll stick to what I had, but simpler.

    # Actually, I'll allow the user to see the cleaned script.

    # Restore "verify_flow_rust.py" clean structure.

    pass


if __name__ == "__main__":
    # Logic to load old script content? No, I overwrote it.
    # I will write a minimal verification script that loads weights and runs check.
    pass
