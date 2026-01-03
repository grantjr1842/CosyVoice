import torch
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb


def test_rope():
    dim_head = 4  # Small dim for easy inspection
    heads = 2
    seq_len = 2

    rope = RotaryEmbedding(dim_head)
    rope_out = rope.forward_from_seq_len(seq_len)
    freqs, scale = rope_out
    print(f"Freqs shape: {freqs.shape}")
    if isinstance(freqs, tuple):
        print(f"Freqs tuple len: {len(freqs)}")
        print(f"Freqs[0] shape: {freqs[0].shape}")
        # x_transformers usually returns (freqs, scale) or similar?
        # Let's assume it returns what apply_rotary_pos_emb needs as 2nd arg.
        pass
    else:
        print(f"Freqs shape: {freqs.shape}")
    print(f"Freqs (cos, sin): {freqs}")

    # Create a simple Q
    # Shape expected by x_transformers apply_rotary_pos_emb?
    # It usually expects (batch, seq_len, heads, dim_head) OR (batch, seq_len, dim) depending on usage
    # But DiT passes: query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    # WAIT! In modules.py:
    # query = attn.to_q(x)
    # ...
    # query = apply_rotary_pos_emb(query, freqs, ...)
    # THEN it does query.view(...).transpose(1, 2)
    # So `query` passed to `apply_rotary_pos_emb` is [Batch, SeqLen, Heads*DimHead] (flattend!)

    # THIS IS THE KEY!
    # If query is [B, N, H*D], and freqs is [N, D_head], what does x_transformers do?

    dim_total = heads * dim_head
    q = torch.ones(1, seq_len, dim_total)
    # Make head 0 different from head 1
    # At t=1
    q[0, 1, :dim_head] = 1.0  # Head 0
    q[0, 1, dim_head:] = 2.0  # Head 1

    print(f"Q (before) shape: {q.shape}")
    print(f"Q (before) values: {q}")

    q_out = apply_rotary_pos_emb(q, freqs)
    print(f"Q (after) shape: {q_out.shape}")
    print(f"Q (after) values: {q_out}")

    # Check if Head 1 is rotated?
    # If apply_rotary_pos_emb treats the last dim as 'feature dimension' and applies rotation...
    # Freqs is [1, 4] for example.
    # Q is [1, 1, 8].
    # It probably broadcasts!
    # If it broadcasts, does it rotate the first 4 elements using freqs?
    # And the next 4 elements?


if __name__ == "__main__":
    test_rope()
