"""
Test HiFT directly with prompt audio to verify HiFT works correctly.
"""

import sys

sys.path.insert(0, ".")

import torch
import torchaudio
from safetensors.torch import load_file

from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
from cosyvoice.hifigan.generator import CausalHiFTGenerator


def main():
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"

    # Load prompt audio directly and compute mel
    print("Loading prompt audio...")
    waveform, sr = torchaudio.load("asset/zero_shot_prompt.wav")

    # Resample to 24k
    if sr != 24000:
        waveform = torchaudio.functional.resample(waveform, sr, 24000)

    print(f"Audio shape: {waveform.shape}, SR: 24000")

    # Create HiFT
    f0_predictor = CausalConvRNNF0Predictor(
        num_class=1,
        in_channels=80,
        cond_channels=512,
    )

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
    print("HiFT loaded")

    # Run the HiFT's _stft to get spectrogram from audio
    with torch.no_grad():
        # HiFT._stft gives (real, imag) parts
        s_stft_real, s_stft_imag = hift._stft(waveform.squeeze(0))
        print(f"STFT real shape: {s_stft_real.shape}")

        # Combine and run through decode-like path
        # Actually, let's use the source module correctly

        # Extract mel from audio using torchaudio
        n_fft = 1920
        hop_size = 480
        n_mels = 80

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_size,
            n_mels=n_mels,
            f_min=0.0,
            f_max=None,
            center=False,
            power=1.0,
        )

        mel = mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        print(f"Mel shape: {mel.shape}")
        print(f"Mel min: {mel.min().item():.4f}, max: {mel.max().item():.4f}")

        # Run HiFT inference
        result = hift.inference(mel)
        audio = result[0] if isinstance(result, tuple) else result

        print(f"Output audio shape: {audio.shape}")
        print(f"Output audio max: {audio.abs().max().item()}")

        # Save
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        torchaudio.save("hift_roundtrip.wav", audio, 24000)
        print("Saved to hift_roundtrip.wav")


if __name__ == "__main__":
    main()
