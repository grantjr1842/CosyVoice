#!/usr/bin/env python3
"""Extract SineGen noise tensors from Python HiFT for deterministic Rust injection."""

import torch
import numpy as np
from safetensors.torch import save_file, load_file
from hyperpyyaml import load_hyperpyyaml

def main():
    print("Loading HiFT model...")
    with open('./pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml', 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'flow': None})

    hift = configs['hift']
    # Load weights!
    ckpt_path = './pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors'
    print(f"Loading weights from {ckpt_path}...")
    hift.load_state_dict(load_file(ckpt_path))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hift.to(device)
    hift.eval()

    print(f"Device: {device}")

    # Load mel from existing artifacts
    artifacts = load_file("debug_artifacts.safetensors")
    mel = artifacts["python_flow_output"].to(device)  # [1, 80, T]
    print(f"Mel shape: {mel.shape}")

    # Hook into SineGen to capture noise tensors
    captured_tensors = {}

    def sinegen_hook(module, inputs, output):
        """Hook to capture SineGen intermediate values."""
        # output is (sine_waves, uv, noise)
        sine_waves, uv, noise = output
        captured_tensors["sine_waves"] = sine_waves.detach().cpu().clone()
        captured_tensors["uv"] = uv.detach().cpu().clone()
        captured_tensors["sinegen_noise"] = noise.detach().cpu().clone()

    def source_module_hook(module, inputs, output):
        """Hook to capture SourceModuleHnNSF values."""
        # output is (sine_merge, noise, uv)
        sine_merge, noise, uv = output
        captured_tensors["sine_merge"] = sine_merge.detach().cpu().clone()
        captured_tensors["source_noise"] = noise.detach().cpu().clone()
        captured_tensors["source_uv"] = uv.detach().cpu().clone()

    # Register hooks
    sinegen_handle = hift.m_source.l_sin_gen.register_forward_hook(sinegen_hook)
    source_handle = hift.m_source.register_forward_hook(source_module_hook)

    with torch.no_grad():
        # Run inference to capture
        speech, source = hift.inference(mel)
        print(f"Speech shape: {speech.shape}")
        print(f"Source shape: {source.shape}")

    # Remove hooks
    sinegen_handle.remove()
    source_handle.remove()

    # Print captured values
    for key, tensor in captured_tensors.items():
        print(f"{key}: shape={tensor.shape}, min={tensor.min():.6f}, max={tensor.max():.6f}")

    # Also capture the rand_ini values from SineGen
    if hasattr(hift.m_source.l_sin_gen, 'rand_ini'):
        rand_ini = hift.m_source.l_sin_gen.rand_ini
        captured_tensors["rand_ini"] = rand_ini.cpu()
        print(f"rand_ini: {rand_ini.cpu().numpy()}")

    # Save for Rust
    tensors_to_save = {
        "mel": mel.cpu().contiguous(),
        "expected_audio": speech.cpu().unsqueeze(0).contiguous(),
    }
    for key, tensor in captured_tensors.items():
        tensors_to_save[key] = tensor.cpu().contiguous()

    save_file(tensors_to_save, "sinegen_parity_test.safetensors")
    print("\nSaved to sinegen_parity_test.safetensors")


if __name__ == "__main__":
    main()
