import torch
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb


def test_rope():
    dim_head = 64
    heads = 8
    seq_len = 10
    batch = 1

    rope = RotaryEmbedding(dim_head)
    freqs = rope.forward_from_seq_len(seq_len)
    if isinstance(freqs, tuple):
        freqs, scale = freqs
    else:
        scale = 1.0

    print(f"Freqs shape: {freqs.shape}")

    q_concat = torch.randn(batch, seq_len, heads * dim_head)
    try:
        q_out = apply_rotary_pos_emb(q_concat, freqs, scale)
        print(f"Case 1 (Concatenated) Success! Output shape: {q_out.shape}")

        # Check if tail is modified
        # First 64 elements should change
        diff_head0 = (q_out[..., :dim_head] - q_concat[..., :dim_head]).abs().max()
        print(f"Diff Head 0: {diff_head0}")

        # Next 64 elements (Head 1)
        diff_head1 = (
            (
                q_out[..., dim_head : 2 * dim_head]
                - q_concat[..., dim_head : 2 * dim_head]
            )
            .abs()
            .max()
        )
        print(f"Diff Head 1: {diff_head1}")

        if diff_head1 < 1e-5:
            print("CRITICAL: Head 1 was NOT rotated!")
        else:
            print("Head 1 WAS rotated.")

    except Exception as e:
        print(f"Case 1 (Concatenated) Failed: {e}")

    # Case 2: Separate heads (what Rust does currently, but manually)
    q_heads = torch.randn(batch, seq_len, heads, dim_head)
    try:
        # Applying to last dim standardly
        # But apply_rotary_pos_emb expects (..., seq, dim)?
        # If we pass (b, n, h, d), seq is h? No, seq needs to match freqs.
        # freqs is (n, d).
        # We need to unsqueeze freqs to (n, 1, d)?
        q_out = apply_rotary_pos_emb(q_heads, freqs, scale)
        print(f"Case 2 (Heads) Success! Output shape: {q_out.shape}")
    except Exception as e:
        print(f"Case 2 (Heads) Failed: {e}")


test_rope()
