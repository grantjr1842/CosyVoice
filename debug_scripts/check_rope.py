
import torch
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

def check_rope():
    dim_head = 64
    heads = 4
    dim = dim_head * heads # 256
    seq_len = 10

    rotary = RotaryEmbedding(dim_head)
    rope_res = rotary.forward_from_seq_len(seq_len) # (freqs, scale)
    freqs, scale = rope_res

    # Create query [1, seq_len, dim]
    q = torch.randn(1, seq_len, dim)
    q_orig = q.clone()

    # Apply RoPE
    # We need to know if apply_rotary_pos_emb expects [b, n, heads, d_head] or [b, n, d]
    # CosyVoice modules.py passes [b, n, d] (rank 3) and then reshapes later.
    # So we pass rank 3.

    try:
        q_rotated = apply_rotary_pos_emb(q, freqs)
    except Exception as e:
        print(f"Error applying to rank 3: {e}")
        # Maybe it needs to be broadcast manually or reshaped?
        # If modules.py passes rank 3, it must work.
        return

    print(f"Original Q shape: {q.shape}")
    print(f"Rotated Q shape: {q_rotated.shape}")

    # Check first head (0..64)
    diff_h0 = (q_rotated[..., :64] - q_orig[..., :64]).abs().max()
    print(f"Head 0 diff: {diff_h0}")

    # Check second head (64..128)
    diff_h1 = (q_rotated[..., 64:128] - q_orig[..., 64:128]).abs().max()
    print(f"Head 1 diff: {diff_h1}")

    if diff_h1 > 1e-5:
        print("CONCLUSION: RoPE is applied to ALL heads (Standard RoPE). Rust is WRONG.")
    else:
        print("CONCLUSION: RoPE is applied to FIRST head only (Partial RoPE). Rust is CORRECT (maybe).")

if __name__ == "__main__":
    check_rope()
