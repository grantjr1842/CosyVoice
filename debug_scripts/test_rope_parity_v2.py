#!/usr/bin/env python3
"""Compare Python x_transformers RoPE against expected Rust implementation."""

import torch
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

def main():
    dim_head = 64
    seq_len = 10
    batch = 1
    heads = 16

    # Create input tensor [B, H, N, D]
    x = torch.randn(batch, heads, seq_len, dim_head)

    # Python x_transformers RotaryEmbedding
    rope = RotaryEmbedding(dim_head)
    freqs, scale = rope.forward_from_seq_len(seq_len)

    print("=== Python x_transformers ===")
    print(f"freqs shape: {freqs.shape}")
    print(f"scale: {scale}")
    print(f"freqs first 8 values (pos=0): {freqs[0, 0, :8].tolist()}")
    print(f"freqs first 8 values (pos=1): {freqs[0, 1, :8].tolist()}")

    # Apply RoPE
    x_rotated = apply_rotary_pos_emb(x, freqs, scale)
    print(f"\nInput x[:, 0, 0, :8]: {x[0, 0, 0, :8].tolist()}")
    print(f"Rotated x[:, 0, 0, :8]: {x_rotated[0, 0, 0, :8].tolist()}")

    # Now simulate what Rust does
    print("\n=== Simulated Rust Implementation ===")

    # Rust inv_freqs: 1.0 / 10000^(i / dim) for i in 0..dim step 2
    inv_freqs = torch.tensor([1.0 / (10000.0 ** (i / dim_head)) for i in range(0, dim_head, 2)])
    print(f"inv_freqs: {inv_freqs[:4].tolist()}")

    # Rust freqs: for each position, interleave: [f0, f0, f1, f1, ...]
    rust_freqs = []
    for pos in range(seq_len):
        t = float(pos)
        row = []
        for inv_f in inv_freqs:
            freq_val = t * inv_f.item()
            row.append(freq_val)
            row.append(freq_val)  # Duplicate
        rust_freqs.append(row)
    rust_freqs = torch.tensor(rust_freqs)  # [seq_len, dim_head]
    print(f"rust_freqs shape: {rust_freqs.shape}")
    print(f"rust_freqs first 8 values (pos=0): {rust_freqs[0, :8].tolist()}")
    print(f"rust_freqs first 8 values (pos=1): {rust_freqs[1, :8].tolist()}")

    # Compare with Python freqs
    # Python freqs are [batch, seq, dim] but we have [1, seq, dim]
    py_freqs_squeezed = freqs.squeeze(0)  # [seq, dim]
    print(f"\npy_freqs_squeezed first 8 (pos=0): {py_freqs_squeezed[0, :8].tolist()}")
    print(f"py_freqs_squeezed first 8 (pos=1): {py_freqs_squeezed[1, :8].tolist()}")

    diff = (rust_freqs - py_freqs_squeezed).abs()
    print(f"\nMax diff between freqs: {diff.max().item()}")
    print(f"Mean diff between freqs: {diff.mean().item()}")

    # If freqs are the same, test rotation
    if diff.max().item() < 1e-5:
        print("\n✅ Freqs match! Testing rotation...")

        # Test rotate_half equivalence
        from x_transformers.x_transformers import rotate_half as py_rotate_half

        x_test = torch.randn(1, 4, 64)  # [B, N, D]

        # Python rotate_half
        py_rotated = py_rotate_half(x_test)
        print(f"\nPython rotate_half output[:4]: {py_rotated[0, 0, :8].tolist()}")

        # Simulate Rust rotate_half
        # Rust: reshape to [..., half, 2], then [-x2, x1]
        x_pairs = x_test.reshape(1, 4, 32, 2)  # [B, N, half, 2]
        x1 = x_pairs[..., 0]  # even indices
        x2 = x_pairs[..., 1]  # odd indices
        rust_rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)  # [-x2, x1]
        print(f"Rust rotate_half output[:4]: {rust_rotated[0, 0, :8].tolist()}")

        rotate_diff = (py_rotated - rust_rotated).abs()
        print(f"\nrotate_half Max diff: {rotate_diff.max().item()}")

        if rotate_diff.max().item() < 1e-6:
            print("✅ rotate_half matches!")
        else:
            print("❌ rotate_half DIFFERS!")

        # Test full apply_rotary_pos_emb
        print("\n=== Testing Full RoPE Application ===")
        x_full = torch.randn(1, 8, 10, 64)  # [B, H, N, D]

        # Get freqs for this sequence
        freqs_full, scale_full = rope.forward_from_seq_len(10)

        # Python application
        py_result = apply_rotary_pos_emb(x_full, freqs_full, scale_full)
        print(f"Python result[:, 0, 0, :8]: {py_result[0, 0, 0, :8].tolist()}")

        # Simulate Rust application
        # Rust works with [B, N, inner_dim] then reshapes back to [B, H, N, D]
        # But let's test the core math: x * cos + rotate_half(x) * sin

        freqs_expanded = freqs_full.unsqueeze(1)  # [1, 1, 10, 64]
        cos_full = freqs_expanded.cos()
        sin_full = freqs_expanded.sin()

        rust_result = (x_full * cos_full) + (py_rotate_half(x_full) * sin_full)
        print(f"Rust result[:, 0, 0, :8]: {rust_result[0, 0, 0, :8].tolist()}")

        full_diff = (py_result - rust_result).abs()
        print(f"\nFull RoPE Max diff: {full_diff.max().item()}")
        print(f"Full RoPE Mean diff: {full_diff.mean().item()}")

        if full_diff.max().item() < 1e-5:
            print("✅ Full RoPE application matches!")
        else:
            print("❌ Full RoPE application DIFFERS!")
    else:
        print("\n❌ Freqs DIFFER! This is the source of parity issue.")

if __name__ == "__main__":
    main()
