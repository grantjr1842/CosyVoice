#!/usr/bin/env python3
import argparse
import os

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a CosyVoice LLM .pt checkpoint to safetensors."
    )
    parser.add_argument("input", help="Path to llm.pt or llm.rl.pt")
    parser.add_argument(
        "--output",
        help="Output safetensors path (default: replace .pt with .safetensors)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise SystemExit(f"Input not found: {input_path}")

    output_path = args.output or input_path.replace(".pt", ".safetensors")
    state = torch.load(input_path, map_location="cpu")
    state = break_shared_tensors(state)
    save_file(state, output_path)
    print(f"Saved: {output_path}")


def storage_id(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().data_ptr()
    except AttributeError:
        return tensor.storage().data_ptr()


def break_shared_tensors(state):
    seen = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        sid = storage_id(value)
        if sid in seen:
            state[key] = value.clone()
        else:
            seen[sid] = key
    return state


if __name__ == "__main__":
    main()
