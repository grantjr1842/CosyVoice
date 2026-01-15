import torch
import torchaudio
import os
import sys

# Add project root to path to import cosyvoice modules
sys.path.append(os.getcwd())

from cosyvoice.compat.matcha_compat import mel_spectrogram

def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav, backend="soundfile")
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

def main():
    wav_path = "asset/interstellar-tars-01-resemble-denoised.wav"
    target_sr = 24000

    print(f"Loading {wav_path}...")
    speech = load_wav(wav_path, target_sr)
    print(f"Audio shape: {speech.shape}")
    print(f"Audio stats: min={speech.min():.6f}, max={speech.max():.6f}, mean={speech.mean():.6f}")

    # Config from cosyvoice3.yaml
    n_fft = 1920
    num_mels = 80
    hop_size = 480
    win_size = 1920
    fmin = 0
    fmax = None # null in yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speech = speech.to(device)

    # Compute Mel
    mel = mel_spectrogram(
        speech,
        n_fft=n_fft,
        num_mels=num_mels,
        sampling_rate=target_sr,
        hop_size=hop_size,
        win_size=win_size,
        fmin=fmin,
        fmax=fmax,
        center=False
    )

    print(f"Mel shape: {mel.shape}")
    print(f"Mel stats: min={mel.min():.6f}, max={mel.max():.6f}, mean={mel.mean():.6f}, mean_abs={mel.abs().mean():.6f}")

    # Alignment logic simulation
    mel_len = mel.shape[2]
    # In Rust example: token_len=159.
    # aligned = min(316 / 2, 159) = 158.
    # So we slice first 158*2 = 316 frames.
    # Actually mel_len is 316. So it takes all of it.

    # But let's print sliced if it were truncated (e.g. if token len was small)

    # Rust printed conds stats log:
    # conds stats: min=-8.976562, max=4.667969, mean=-0.463873, mean_abs=0.566223

    # Conds in Rust is [prompt_feat, zeros].
    # So stats of conds will be dragged down by zeros.
    # But max and min should match prompt_feat (unless clean mel has smaller range).
    # Rust random noise min/max: -4.7 to 4.6.
    # Rust conds max: 4.66.

    # We should look at Mel stats explicitly.
    pass

if __name__ == "__main__":
    main()
