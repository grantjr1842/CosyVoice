"""
Capture intermediate tensors from HiFT decode pipeline for parity testing.

Saves tensors at each stage:
- conv_pre output
- up/resblock outputs
- source fusion
- conv_post output (mag/phase)
- ISTFT output
"""

import sys
sys.path.insert(0, ".")

import torch
from safetensors.torch import load_file, save_file
from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
from cosyvoice.hifigan.generator import CausalHiFTGenerator
import torch.nn.functional as F


def log_stats(name, t):
    print(f"  {name}: shape={tuple(t.shape)}, min={t.min().item():.6f}, max={t.max().item():.6f}, mean={t.mean().item():.6f}")


def main():
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"

    # Load test artifacts
    artifacts = load_file("debug_artifacts.safetensors")
    mel = artifacts["python_flow_output"].float()  # [1, 80, T]
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

    saved = {}

    with torch.no_grad():
        speech_feat = mel

        # F0 predictor
        f0 = hift.f0_predictor(speech_feat, finalize=True)
        log_stats("f0", f0)
        saved["f0"] = f0.cpu()

        # F0 upsample
        s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, T_up, 1]
        log_stats("f0_upsampled", s)
        saved["f0_upsampled"] = s.cpu()

        # Source module
        s_out, noise, uv = hift.m_source(s)
        s_out = s_out.transpose(1, 2)  # [B, 1, T_up]
        log_stats("source", s_out)
        saved["source"] = s_out.cpu()

        # Now capture decode stages
        x = speech_feat
        s_source = s_out

        # STFT of source
        s_stft_real, s_stft_imag = hift._stft(s_source.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
        log_stats("s_stft", s_stft)
        saved["s_stft"] = s_stft.cpu()

        # conv_pre
        x = hift.conv_pre(x)
        log_stats("conv_pre", x)
        saved["conv_pre"] = x.cpu()

        # Upsample loop
        for i in range(hift.num_upsamples):
            x = F.leaky_relu(x, hift.lrelu_slope)
            x = hift.ups[i](x)

            if i == hift.num_upsamples - 1:
                x = hift.reflection_pad(x)

            # Source fusion
            si = hift.source_downs[i](s_stft)
            si = hift.source_resblocks[i](si)

            # Match lengths
            min_len = min(x.shape[2], si.shape[2])
            x = x[:, :, :min_len] + si[:, :, :min_len]

            log_stats(f"after_fusion_{i}", x)
            saved[f"after_fusion_{i}"] = x.cpu()

            # ResBlocks
            xs = None
            for j in range(hift.num_kernels):
                if xs is None:
                    xs = hift.resblocks[i * hift.num_kernels + j](x)
                else:
                    xs += hift.resblocks[i * hift.num_kernels + j](x)
            x = xs / hift.num_kernels

            log_stats(f"after_resblocks_{i}", x)
            saved[f"after_resblocks_{i}"] = x.cpu()

        # conv_post
        x = F.leaky_relu(x)
        x = hift.conv_post(x)
        log_stats("conv_post", x)
        saved["conv_post"] = x.cpu()

        # Magnitude/Phase split
        magnitude = torch.exp(x[:, :hift.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, hift.istft_params["n_fft"] // 2 + 1:, :])

        log_stats("magnitude", magnitude)
        log_stats("phase", phase)
        saved["magnitude"] = magnitude.cpu()
        saved["phase"] = phase.cpu()

        # ISTFT
        magnitude_clipped = torch.clip(magnitude, max=1e2)
        audio = hift._istft(magnitude_clipped, phase)
        log_stats("pre_clamp_audio", audio)
        saved["pre_clamp_audio"] = audio.cpu()

        # Clamp
        audio = torch.clamp(audio, -hift.audio_limit, hift.audio_limit)
        log_stats("final_audio", audio)
        saved["final_audio"] = audio.cpu()

    # Save all intermediates
    save_file(saved, "hift_stages_debug.safetensors")
    print(f"\nSaved {len(saved)} tensors to hift_stages_debug.safetensors")


if __name__ == "__main__":
    main()
