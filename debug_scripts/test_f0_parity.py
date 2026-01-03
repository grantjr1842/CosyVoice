import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, ".")
from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor


def log_stats(name, tensor):
    print(
        f"    [F0Predictor] {name} stats: min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}"
    )


def main():
    # Load model
    model_path = "pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors"
    sd = load_file(model_path)

    # Filter for f0_predictor
    f0_sd = {
        k.replace("f0_predictor.", ""): v
        for k, v in sd.items()
        if k.startswith("f0_predictor.")
    }

    # Initialize model
    model = CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
    model.load_state_dict(f0_sd)
    model.eval()

    # Log weight/bias stats for each layer (to match Rust logging)
    for i in range(5):
        conv = model.condnet[i * 2]
        log_stats(f"Layer {i} weights", conv.weight)
        if conv.bias is not None:
            log_stats(f"Layer {i} bias", conv.bias)

    # Load real mel
    mel_path = "debug_mel.safetensors"
    if not Path(mel_path).exists():
        print(
            f"Error: {mel_path} not found. Run debug_scripts/debug_flow_mel.py first."
        )
        return

    mel_data = load_file(mel_path)
    mel = mel_data["mel"]

    log_stats("Input Mel", mel)

    # Forward pass with intermediate logging
    with torch.no_grad():
        x = mel
        for i in range(5):
            # CausalConv1d
            x = model.condnet[i * 2](x)
            # ELU
            x = model.condnet[i * 2 + 1](x)
            log_stats(f"Layer {i} (after ELU)", x)

        # Final classifier
        x = x.transpose(1, 2)
        f0 = torch.abs(model.classifier(x).squeeze(-1))
        log_stats("Final F0", f0)


if __name__ == "__main__":
    main()
