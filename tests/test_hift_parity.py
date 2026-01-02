#!/usr/bin/env python3
"""
HiFT parity test - run Python HiFT on exact same input as Rust test_native.
Compare outputs to find the divergence.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from safetensors.torch import load_file, save_file

# Add project root and third_party to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "Matcha-TTS"))

from hyperpyyaml import load_hyperpyyaml


def tensor_stats(t: torch.Tensor, name: str) -> dict:
    """Compute statistics for a tensor."""
    t_flat = t.float().flatten().detach().cpu()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": float(t_flat.min().item()),
        "max": float(t_flat.max().item()),
        "mean": float(t_flat.mean().item()),
        "std": float(t_flat.std().item()) if t_flat.numel() > 1 else 0.0,
        "first_10": t_flat[:10].tolist() if t_flat.numel() >= 10 else t_flat.tolist(),
    }


def compute_sinegen2_debug(l_sin_gen, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    harmonic_num = l_sin_gen.harmonic_num
    sampling_rate = l_sin_gen.sampling_rate
    upsample_scale = l_sin_gen.upsample_scale
    causal = getattr(l_sin_gen, "causal", False)

    fn = f0 * torch.FloatTensor([[range(1, harmonic_num + 2)]]).to(f0.device)
    rad_values = (fn / sampling_rate) % 1
    rad_values = rad_values.clone()

    if not l_sin_gen.training and causal:
        rand_ini = l_sin_gen.rand_ini.to(rad_values.device)
    else:
        rand_ini = torch.rand(
            rad_values.shape[0], rad_values.shape[2], device=rad_values.device
        )
        rand_ini[:, 0] = 0
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

    rad_down = F.interpolate(
        rad_values.transpose(1, 2),
        scale_factor=1 / upsample_scale,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    if causal:
        bsz, down_len, harmonics = rad_down.shape
        phase_up = torch.zeros(
            (bsz, down_len * upsample_scale, harmonics),
            device=rad_down.device,
            dtype=rad_down.dtype,
        )
        scale = torch.tensor(
            2 * torch.pi * upsample_scale,
            device=rad_down.device,
            dtype=rad_down.dtype,
        )
        for b in range(bsz):
            for h in range(harmonics):
                acc = torch.tensor(0.0, device=rad_down.device, dtype=rad_down.dtype)
                for t in range(down_len):
                    acc = acc + rad_down[b, t, h]
                    phase_val = acc * scale
                    start = t * upsample_scale
                    phase_up[b, start : start + upsample_scale, h] = phase_val
    else:
        phase = torch.cumsum(rad_down, dim=1) * 2 * torch.pi
        phase_input = phase.transpose(1, 2) * upsample_scale
        phase_up = F.interpolate(
            phase_input,
            scale_factor=upsample_scale,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    return rad_down, phase_up


def main():
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"

    # Load the same test artifacts used by Rust
    print("Loading test artifacts...")
    artifact_path = os.environ.get("ARTIFACT_PATH", "tests/test_artifacts.safetensors")
    artifacts = load_file(artifact_path)
    flow_feat_24k = artifacts["flow_feat_24k"]  # [1, 80, 171]

    print(f"Input mel (flow_feat_24k) shape: {flow_feat_24k.shape}")
    print(
        f"Input mel stats: min={flow_feat_24k.min():.4f}, max={flow_feat_24k.max():.4f}, mean={flow_feat_24k.mean():.4f}"
    )

    # Load config
    hyper_yaml_path = os.path.join(model_dir, "cosyvoice3.yaml")
    with open(hyper_yaml_path, "r") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")
            },
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("test_hift_debug.log", "w") as log:
        log.write(f"Device: {device}\n")

    # Load HiFT
    print("\nLoading HiFT model...")
    hift = configs["hift"]
    hift_state_dict = {
        k.replace("generator.", ""): v
        for k, v in torch.load(
            os.path.join(model_dir, "hift.pt"), map_location=device, weights_only=True
        ).items()
    }
    hift.load_state_dict(hift_state_dict, strict=True)
    hift.to(device).eval()
    print("HiFT loaded!")
    with open("test_hift_debug.log", "a") as log:
        log.write(f"HiFT device check: {next(hift.parameters()).device}\n")

    # Move mel to device
    mel = flow_feat_24k.to(device)
    log_path = Path("outputs/logs/test_hift_debug.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"Mel device: {mel.device}\n")

    # Run Python HiFT and capture detailed intermediates
    print("\n=== Running Python HiFT (deterministic path) ===")
    with torch.inference_mode():
        def run_f0_with_debug(predictor, mel_cpu, finalize=True):
            condnet = predictor.condnet
            x = mel_cpu
            layer_outputs = {}
            layer_idx = 0

            if finalize:
                x = condnet[0](x)
            else:
                pad = condnet[0].causal_padding
                x = condnet[0](x[:, :, :-pad], x[:, :, -pad:])
            x = condnet[1](x)
            layer_outputs[f"f0_layer{layer_idx}"] = x.detach()
            layer_idx += 1

            for i in range(2, len(condnet), 2):
                x = condnet[i](x)
                x = condnet[i + 1](x)
                layer_outputs[f"f0_layer{layer_idx}"] = x.detach()
                layer_idx += 1

            x = x.transpose(1, 2)
            classifier_pre_abs = predictor.classifier(x)
            f0 = torch.abs(classifier_pre_abs).squeeze(-1)
            return f0, layer_outputs, classifier_pre_abs

        # F0 prediction
        with open("outputs/logs/test_hift_debug.log", "a") as log:
            log.write(f"Before f0_predictor calling with mel device: {mel.device}\n")
            # log.write(f"f0_predictor device check: {next(hift.f0_predictor.parameters()).device}\n") # Verify this

        hift.f0_predictor.to("cpu")
        f0_cpu, f0_layers, f0_classifier_pre_abs = run_f0_with_debug(
            hift.f0_predictor, mel.cpu(), finalize=True
        )
        f0 = f0_cpu.to(mel)
        print(f"F0 predictor output: {tensor_stats(f0, 'f0')}")

        # Upsample F0
        s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        print(f"Upsampled F0 (s): {tensor_stats(s, 's')}")

        # Source module
        sine_merge, noise, uv = hift.m_source(s)
        print(f"Source sine_merge: {tensor_stats(sine_merge, 'sine_merge')}")
        print(f"Source noise: {tensor_stats(noise, 'noise')}")
        print(f"Source uv: {tensor_stats(uv, 'uv')}")

        rand_ini = None
        sine_noise_cache = None
        source_noise_cache = None
        sine_waves = None
        sine_rad_down = None
        sine_phase_up = None

        if hasattr(hift.m_source, "l_sin_gen"):
            l_sin_gen = hift.m_source.l_sin_gen
            if hasattr(l_sin_gen, "rand_ini"):
                rand_ini = l_sin_gen.rand_ini.detach().cpu()
            if hasattr(l_sin_gen, "sine_waves"):
                sine_noise_cache = l_sin_gen.sine_waves[:, : s.shape[1], :].detach().cpu()
            with torch.no_grad():
                sine_waves, _, _ = l_sin_gen(s)
            if hasattr(l_sin_gen, "upsample_scale"):
                sine_rad_down, sine_phase_up = compute_sinegen2_debug(l_sin_gen, s)
        if hasattr(hift.m_source, "uv"):
            source_noise_cache = hift.m_source.uv[:, : uv.shape[1], :].detach().cpu()

        # Transpose for decode
        s_source = sine_merge.transpose(1, 2)

        # Decode (detailed)
        # STFT of source
        s_stft_real, s_stft_imag = hift._stft(s_source.squeeze(1))
        print(f"Source STFT real: {tensor_stats(s_stft_real, 's_stft_real')}")
        print(f"Source STFT imag: {tensor_stats(s_stft_imag, 's_stft_imag')}")
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        # Pre-conv
        x = hift.conv_pre(mel)
        conv_pre_out = x.detach()
        print(f"After conv_pre: {tensor_stats(conv_pre_out, 'conv_pre_out')}")

        decode_intermediates = {}

        for i in range(hift.num_upsamples):
            x = F.leaky_relu(x, hift.lrelu_slope)
            decode_intermediates[f"upsample_{i}_pre"] = x.detach()
            x = hift.ups[i](x)
            if i == hift.num_upsamples - 1:
                x = hift.reflection_pad(x)
            decode_intermediates[f"upsample_{i}_out"] = x.detach()

            si = hift.source_downs[i](s_stft)
            decode_intermediates[f"source_down_{i}_out"] = si.detach()
            si = hift.source_resblocks[i](si)
            decode_intermediates[f"source_resblock_{i}_out"] = si.detach()
            x = x + si
            decode_intermediates[f"fusion_{i}_out"] = x.detach()

            xs = None
            for j in range(hift.num_kernels):
                out = hift.resblocks[i * hift.num_kernels + j](x)
                decode_intermediates[f"resblock_{i}_{j}_out"] = out.detach()
                if xs is None:
                    xs = out
                else:
                    xs = xs + out
            x = xs / hift.num_kernels
            decode_intermediates[f"resblock_{i}_out"] = x.detach()

        x = F.leaky_relu(x)
        decode_intermediates["post_lrelu_out"] = x.detach()
        conv_post_out = hift.conv_post(x)

        n_fft = hift.istft_params["n_fft"]
        cutoff = n_fft // 2 + 1
        magnitude = torch.exp(conv_post_out[:, :cutoff, :])
        magnitude = torch.clip(magnitude, max=1e2)
        phase = torch.sin(conv_post_out[:, cutoff:, :])
        istft_audio = hift._istft(magnitude, phase)
        audio = torch.clamp(istft_audio, -hift.audio_limit, hift.audio_limit)

    print(f"Python audio shape: {audio.shape}")
    print(
        f"Python audio stats: min={audio.min():.6f}, max={audio.max():.6f}, mean={audio.mean():.6f}, std={audio.std():.6f}"
    )

    # Save for comparison
    output_dir = Path("outputs/audio/hift_parity")
    output_dir.mkdir(parents=True, exist_ok=True)

    torchaudio.save(str(output_dir / "python_hift_output.wav"), audio.cpu(), 24000)
    print(f"\nSaved Python HiFT output to: {output_dir / 'python_hift_output.wav'}")

    # Load Rust native output if it exists
    rust_output_path = "outputs/audio/native_hift_output.wav"
    skip_audio_compare = os.environ.get("SKIP_AUDIO_COMPARE", "0") == "1"
    if not skip_audio_compare and os.path.exists(rust_output_path):
        print(f"\n=== Loading Rust native output: {rust_output_path} ===")
        rust_audio, rust_sr = torchaudio.load(rust_output_path)
        print(f"Rust audio shape: {rust_audio.shape}")
        print(
            f"Rust audio stats: min={rust_audio.min():.6f}, max={rust_audio.max():.6f}, mean={rust_audio.mean():.6f}, std={rust_audio.std():.6f}"
        )

        # Compare
        print("\n=== Comparison ===")
        min_len = min(audio.shape[1], rust_audio.shape[1])
        py_samples = audio[0, :min_len].cpu()
        rust_samples = rust_audio[0, :min_len]

        mae = (py_samples - rust_samples).abs().mean()
        max_diff = (py_samples - rust_samples).abs().max()
        correlation = torch.corrcoef(torch.stack([py_samples, rust_samples]))[0, 1]

        print(f"Samples compared: {min_len}")
        print(f"MAE: {mae:.6f}")
        print(f"Max diff: {max_diff:.6f}")
        print(f"Correlation: {correlation:.6f}")

        # Check for issues
        if audio.abs().max() < 0.1:
            print("\n⚠️  WARNING: Python output is very quiet (max < 0.1)")
        if rust_audio.abs().max() > 0.99:
            print("\n⚠️  WARNING: Rust output appears clipped (max > 0.99)")
        if rust_audio.mean().abs() > 0.1:
            print(
                f"\n⚠️  WARNING: Rust output has significant DC offset (mean = {rust_audio.mean():.4f})"
            )
        if correlation < 0.5:
            print(
                f"\n❌ CRITICAL: Low correlation ({correlation:.4f}) - outputs are fundamentally different"
            )

    # Save intermediate tensors
    intermediates = {
        "input_mel": mel.cpu().contiguous(),
        "f0_output": f0.cpu().contiguous(),
        "source_s": s.cpu().contiguous(),
        "sine_merge": sine_merge.cpu().contiguous(),
        "noise": noise.cpu().contiguous(),
        "uv": uv.cpu().contiguous(),
        "s_stft_real": s_stft_real.cpu().contiguous(),
        "s_stft_imag": s_stft_imag.cpu().contiguous(),
        "conv_pre_out": conv_pre_out.cpu().contiguous(),
        "conv_post_out": conv_post_out.cpu().contiguous(),
        "magnitude": magnitude.cpu().contiguous(),
        "phase": phase.cpu().contiguous(),
        "istft_audio": istft_audio.cpu().contiguous(),
        "final_audio": audio.cpu().contiguous(),
    }
    for name, tensor in decode_intermediates.items():
        intermediates[name] = tensor.cpu().contiguous()
    for name, tensor in f0_layers.items():
        intermediates[name] = tensor.cpu().contiguous()
    if f0_classifier_pre_abs is not None:
        intermediates["f0_classifier_pre_abs"] = f0_classifier_pre_abs.cpu().contiguous()
    if rand_ini is not None:
        intermediates["rand_ini"] = rand_ini.contiguous()
    if sine_noise_cache is not None:
        intermediates["sine_noise_cache"] = sine_noise_cache.contiguous()
    if source_noise_cache is not None:
        intermediates["source_noise_cache"] = source_noise_cache.contiguous()
    if sine_waves is not None:
        intermediates["sine_waves"] = sine_waves.cpu().contiguous()
    if sine_rad_down is not None:
        intermediates["sine_rad_down"] = sine_rad_down.cpu().contiguous()
    if sine_phase_up is not None:
        intermediates["sine_phase_up"] = sine_phase_up.cpu().contiguous()
    debug_dir = Path("outputs/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_file(intermediates, str(debug_dir / "python_intermediates.safetensors"))
    print(
        f"\nSaved intermediate tensors to: {debug_dir / 'python_intermediates.safetensors'}"
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"Python HiFT audio: {audio.shape}, range [{audio.min():.4f}, {audio.max():.4f}]"
    )
    if not skip_audio_compare and os.path.exists(rust_output_path):
        print(
            f"Rust native audio: {rust_audio.shape}, range [{rust_audio.min():.4f}, {rust_audio.max():.4f}]"
        )
        if rust_audio.abs().max() > 0.99:
            print(
                "\n🚨 The Rust output is CLIPPING. This explains the 'garbage' audio."
            )
            print("   The issue is likely in:")
            print("   1. Flow model producing wrong mel ranges")
            print("   2. HiFT vocoder implementation (conv weights, ISTFT, etc.)")
            print("   3. Numerical issues (overflow in exp(), wrong phase handling)")


if __name__ == "__main__":
    main()
