"""
Test HiFT parity between Python reference and Rust implementation.
Uses the same mel input from test_artifacts.safetensors.
"""

import sys

import torch
import torchaudio

sys.path.insert(0, ".")

from safetensors.torch import load_file

from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
from cosyvoice.hifigan.generator import CausalHiFTGenerator


def main():
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"

    # Load test artifact
    artifacts = load_file("tests/test_artifacts.safetensors")
    mel = artifacts["flow_feat_24k"].float()  # [1, 80, T]
    print(f"Mel shape: {mel.shape}")

    # Create F0 predictor
    f0_predictor = CausalConvRNNF0Predictor(
        num_class=1,
        in_channels=80,
        cond_channels=512,
    )

    # Create HiFT
    hift = CausalHiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        conv_pre_look_right=4,
        f0_predictor=f0_predictor,
    )

    # Load weights
    hift_weights = load_file(f"{model_dir}/hift.safetensors")
    hift.load_state_dict(hift_weights)
    hift.eval()

    print("HiFT model loaded successfully")

    # Run inference
    with torch.no_grad():
        # HiFT inference method for mel -> audio
        result = hift.inference(mel)
        audio = result[0] if isinstance(result, tuple) else result

    print(f"Audio shape: {audio.shape}")
    print(f"Audio max: {audio.abs().max().item()}")
    print(f"Audio mean: {audio.mean().item()}")
    print(f"Audio std: {audio.std().item()}")

    # Save to file - audio is [B, L], need [C, L]
    if audio.dim() == 2:
        audio_save = audio  # [1, L] is valid for mono
    else:
        audio_save = audio.squeeze(0) if audio.dim() == 3 else audio
    torchaudio.save("python_hift_output.wav", audio_save, 24000)
    print("Saved to python_hift_output.wav")


if __name__ == "__main__":
    main()
