
import torch
import torch.nn as nn

def check():
    b, t, d = 1, 10, 64
    x = torch.randn(b, t, d)
    ln = nn.LayerNorm(d)

    y = ln(x)

    # Check Frame 0 mean
    f0_mean = y[0, 0, :].mean()
    print(f"Frame 0 mean: {f0_mean:.8f}")

    # Check whole tensor mean
    whole_mean = y.mean()
    print(f"Whole tensor mean: {whole_mean:.8f}")

    if abs(f0_mean) < 1e-6:
        print("CONCLUSION: LayerNorm(D) normalizes per-frame.")
    else:
        print("CONCLUSION: LayerNorm(D) normalizes sequence-wide (WAIT WHAT?)")

if __name__ == "__main__":
    check()
