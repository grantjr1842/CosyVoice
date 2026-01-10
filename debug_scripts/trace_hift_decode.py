"""Capture HiFT decode intermediates for parity comparison with Rust.

This traces through the HiFT decode loop to capture x_mean and si_mean
at each fusion point, matching the Rust debug output.
"""

import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml
from safetensors.torch import load_file, save_file

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def tensor_stats(t, name):
    t = t.float()
    print(f"  {name}: min={t.min():.6f}, max={t.max():.6f}, mean={t.mean():.6f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_dir = repo_root / "pretrained_models" / "Fun-CosyVoice3-0.5B"
    with open(model_dir / "cosyvoice3.yaml") as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})

    hift = configs["hift"]
    hift_weights = load_file(model_dir / "hift.safetensors")
    hift.load_state_dict(hift_weights, strict=False)
    hift.eval()
    hift.to(device)

    # Load flow output mel from rust debug data
    data_path = repo_root / "rust" / "server" / "rust_flow_debug.safetensors"
    if not data_path.exists():
        data_path = repo_root / "debug_flow_data.safetensors"

    rust_dump = load_file(data_path)

    # Get mel from flow output or generate one
    # Use the reference flow output as mel input to HiFT
    if "flow_output" in rust_dump:
        mel = rust_dump["flow_output"].to(device)
    elif "ref_flow_output" in rust_dump:
        mel = rust_dump["ref_flow_output"].to(device)
    else:
        # Generate mel using flow
        print("Flow output not found in debug data, running flow inference...")
        with open(model_dir / "cosyvoice3.yaml") as f:
            full_configs = load_hyperpyyaml(f, overrides={"llm": None, "hift": None})
        flow = full_configs["flow"]
        flow_weights = load_file(model_dir / "flow.safetensors")
        flow.load_state_dict(flow_weights, strict=False)
        flow.eval()
        flow.to(device)

        mu = rust_dump["mu"].to(device)
        mask = rust_dump["mask"].to(device)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        spks = rust_dump["spks"].to(device)
        cond = rust_dump["cond"].to(device)
        x_init = rust_dump["x_init"].to(device)

        with torch.inference_mode():
            mel = flow.decoder.forward(mu, mask, x_init, spks, cond, 10, 0.7)

    print(f"Mel input shape: {mel.shape}")
    tensor_stats(mel, "mel")

    print("\n=== Tracing HiFT decode ===")
    intermediates = {}

    with torch.inference_mode():
        # Run F0 predictor
        f0 = hift.f0_predictor(mel)
        mel_len = mel.shape[2]
        print(f"\nF0 raw shape: {f0.shape}")
        # Handle different f0 shapes
        if f0.dim() == 2:
            f0 = f0[:, :mel_len].unsqueeze(1)  # [B, T] -> [B, 1, T]
        else:
            f0 = f0[:, :, :mel_len]
        print(f"F0 shape: {f0.shape}")
        tensor_stats(f0, "f0")
        intermediates["f0"] = f0.cpu()

        # Upsample F0
        # f0 is already [B, 1, T] from the check above
        s = hift.f0_upsamp(f0).transpose(1, 2)

        # Source module
        s, _, uv = hift.m_source(s)
        s = s.transpose(1, 2)
        print(f"Source shape: {s.shape}")
        tensor_stats(s, "source")
        intermediates["source"] = s.cpu()

        # === Manual decode trace ===
        print("\n=== Decode loop trace ===")

        # STFT of source
        s_stft_real, s_stft_imag = hift._stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
        print(f"s_stft shape: {s_stft.shape}")
        tensor_stats(s_stft, "s_stft")
        intermediates["s_stft"] = s_stft.cpu()

        # Conv pre
        x = hift.conv_pre(mel)
        print(f"\nAfter conv_pre: {x.shape}")
        tensor_stats(x, "x after conv_pre")

        # Debug conv_pre weights
        if hasattr(hift.conv_pre, "weight"):
            tensor_stats(hift.conv_pre.weight, "conv_pre weight")
        elif hasattr(hift.conv_pre, "weight_v"):
            w = (
                hift.conv_pre.weight_v
                * hift.conv_pre.weight_g
                / torch.norm(hift.conv_pre.weight_v, dim=2, keepdim=True)
            )
            tensor_stats(w, "conv_pre weight (computed)")


        for i in range(hift.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, hift.lrelu_slope)
            x = hift.ups[i](x)

            if i == hift.num_upsamples - 1:
                x = hift.reflection_pad(x)

            print(f"\n=== Loop {i} ===")
            print(f"After ups[{i}]: {x.shape}")
            tensor_stats(x, f"x after ups[{i}]")

            # Source fusion
            si = hift.source_downs[i](s_stft)
            si = hift.source_resblocks[i](si)
            print(f"si shape: {si.shape}")
            tensor_stats(si, f"si[{i}]")

            # Fusion stats (matching Rust debug)
            x_mean = x.mean().item()
            si_mean = si.mean().item()
            print(f"  Fusion: x_mean={x_mean:.4f}, si_mean={si_mean:.4f}")

            # Fusion
            x = x + si
            print("After fusion:")
            tensor_stats(x, f"x after fusion[{i}]")

            # Resblocks
            xs = None
            for j in range(hift.num_kernels):
                if xs is None:
                    xs = hift.resblocks[i * hift.num_kernels + j](x)
                else:
                    xs = xs + hift.resblocks[i * hift.num_kernels + j](x)
            x = xs / hift.num_kernels

            print("After resblocks:")
            tensor_stats(x, f"x after resblocks[{i}]")

            intermediates[f"loop{i}_x"] = x.cpu()
            intermediates[f"loop{i}_si"] = si.cpu()

        # Post
        x = torch.nn.functional.leaky_relu(x)
        x = hift.conv_post(x)

        # Debug Snake Alphas
        print("\n=== Snake Alphas ===")
        # Collect all modules
        snakes = []
        for name, mod in hift.named_modules():
            if "Snake" in str(type(mod)):
                snakes.append((name, mod))

        for name, mod in snakes:
            if hasattr(mod, "alpha"):
                tensor_stats(mod.alpha, f"Snake {name} alpha")

        # Debug conv_post weights
        if hasattr(hift.conv_post, "weight_g"):
            # Weight norm applied
            w = (
                hift.conv_post.weight_v
                * hift.conv_post.weight_g
                / torch.norm(hift.conv_post.weight_v, dim=2, keepdim=True)
            )
            pass
        # Actually hift.conv_post is likely just Conv1d if remove_weight_norm was called?
        # No, hift.eval() doesn't remove weight norm automatically.
        # But we can look at .weight directly which PyTorch computes on forward if parametrized/hooked?
        # Or look at raw params.

        # Let's just print the effective weight
        if hasattr(hift.conv_post, "weight"):
            w = hift.conv_post.weight
            print(f"\nconv_post weight shape: {w.shape}")
            tensor_stats(w, "conv_post weight")
        if hasattr(hift.conv_post, "bias") and hift.conv_post.bias is not None:
            b = hift.conv_post.bias
            tensor_stats(b, "conv_post bias")

        print("\nAfter conv_post (mag_log):")
        tensor_stats(x, "conv_post output")

        n_fft_half = hift.istft_params["n_fft"] // 2 + 1
        magnitude = torch.exp(x[:, :n_fft_half, :])
        phase = torch.sin(x[:, n_fft_half:, :])

        print("\n=== ISTFT input ===")
        tensor_stats(magnitude, "magnitude")
        tensor_stats(phase, "phase")

        # ISTFT
        audio = hift._istft(magnitude, phase)
        intermediates["pre_clamp_audio"] = audio.cpu()
        audio = torch.clamp(audio, -hift.audio_limit, hift.audio_limit)

        print("\n=== Final audio ===")
        print(f"Audio shape: {audio.shape}")
        tensor_stats(audio, "audio")

        # Check DC offset
        dc_offset = audio.mean().item()
        print(f"\nDC offset: {dc_offset:.6f}")
        if abs(dc_offset) > 0.1:
            print("⚠️  Large DC offset detected!")

        # Save intermediates
        intermediates["mel"] = mel.cpu()
        intermediates["audio"] = audio.cpu()
        save_file(intermediates, repo_root / "py_hift_intermediates.safetensors")
        print("\nSaved intermediates to py_hift_intermediates.safetensors")


if __name__ == "__main__":
    main()
