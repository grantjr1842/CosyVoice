#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

# Add project root and third_party to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "Matcha-TTS"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export HiFT f0 predictor to torchscript."
    )
    parser.add_argument(
        "--model-dir",
        default="pretrained_models/Fun-CosyVoice3-0.5B",
        help="Path to model directory containing cosyvoice3.yaml and hift.pt.",
    )
    parser.add_argument(
        "--output",
        help="Output torchscript path (default: <model-dir>/f0_predictor.ts).",
    )
    parser.add_argument(
        "--example-len",
        type=int,
        default=171,
        help="Example mel length for tracing.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_path = (
        Path(args.output)
        if args.output
        else model_dir / "f0_predictor.ts"
    )

    with open(model_dir / "cosyvoice3.yaml", "r") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                "llm": None,
                "flow": None,
                "qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN"),
            },
        )

    hift = configs["hift"]
    state = torch.load(model_dir / "hift.pt", map_location="cpu", weights_only=True)
    state = {k.replace("generator.", ""): v for k, v in state.items()}
    hift.load_state_dict(state, strict=True)
    hift.eval()

    class F0Wrapper(torch.nn.Module):
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor

        def forward(self, mel):
            return self.predictor(mel, finalize=True)

    wrapper = F0Wrapper(hift.f0_predictor)
    example = torch.zeros(1, 80, args.example_len, dtype=torch.float32)
    traced = torch.jit.trace(wrapper, example, strict=False)
    traced.save(output_path)
    print(f"Saved torchscript f0 predictor to {output_path}")


if __name__ == "__main__":
    main()
