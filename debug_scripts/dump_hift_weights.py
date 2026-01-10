#!/usr/bin/env python3
"""
Dump HiFT model weight statistics for comparison with Rust.
"""

import torch
from pathlib import Path

repo_root = Path(__file__).parent.parent

def main():
    # Load HiFT model
    from hyperpyyaml import load_hyperpyyaml
    model_dir = repo_root / "pretrained_models" / "Fun-CosyVoice3-0.5B"

    with open(model_dir / "cosyvoice3.yaml", "r") as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None, "flow": None, "qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN")})

    hift = configs["hift"]
    hift.load_state_dict(torch.load(model_dir / "hift.pt", map_location="cpu"))
    hift.eval()

    print("=== HiFT Weight Statistics ===\n")

    # conv_pre
    print("conv_pre:")
    if hasattr(hift.conv_pre, 'weight'):
        w = hift.conv_pre.weight.data.float()
        print(f"  weight: shape={list(w.shape)}, min={w.min().item():.6f}, max={w.max().item():.6f}, mean={w.mean().item():.8f}")
    if hasattr(hift.conv_pre, 'bias') and hift.conv_pre.bias is not None:
        b = hift.conv_pre.bias.data.float()
        print(f"  bias: shape={list(b.shape)}, min={b.min().item():.6f}, max={b.max().item():.6f}, mean={b.mean().item():.8f}")
    # Check for weight_norm
    if hasattr(hift.conv_pre, 'weight_g'):
        g = hift.conv_pre.weight_g.data.float()
        print(f"  weight_g: shape={list(g.shape)}, min={g.min().item():.6f}, max={g.max().item():.6f}")
    if hasattr(hift.conv_pre, 'weight_v'):
        v = hift.conv_pre.weight_v.data.float()
        print(f"  weight_v: shape={list(v.shape)}, min={v.min().item():.6f}, max={v.max().item():.6f}")

    # conv_post
    print("\nconv_post:")
    if hasattr(hift.conv_post, 'weight'):
        w = hift.conv_post.weight.data.float()
        print(f"  weight: shape={list(w.shape)}, min={w.min().item():.6f}, max={w.max().item():.6f}, mean={w.mean().item():.8f}")
    if hasattr(hift.conv_post, 'bias') and hift.conv_post.bias is not None:
        b = hift.conv_post.bias.data.float()
        print(f"  bias: shape={list(b.shape)}, min={b.min().item():.6f}, max={b.max().item():.6f}, mean={b.mean().item():.8f}")

    # ups[0]
    print("\nups[0]:")
    if len(hift.ups) > 0:
        w = hift.ups[0].weight.data.float()
        print(f"  weight: shape={list(w.shape)}, min={w.min().item():.6f}, max={w.max().item():.6f}")
        if hift.ups[0].bias is not None:
            b = hift.ups[0].bias.data.float()
            print(f"  bias: shape={list(b.shape)}, min={b.min().item():.6f}, max={b.max().item():.6f}")

    # source_downs[0]
    print("\nsource_downs[0]:")
    if len(hift.source_downs) > 0:
        if hasattr(hift.source_downs[0], 'weight'):
            w = hift.source_downs[0].weight.data.float()
            print(f"  weight: shape={list(w.shape)}, min={w.min().item():.6f}, max={w.max().item():.6f}")

if __name__ == "__main__":
    main()
