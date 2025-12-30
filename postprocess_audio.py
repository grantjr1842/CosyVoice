#!/usr/bin/env python3
"""
Audio Post-Processing for CosyVoice Output

Applies noise reduction and normalization to improve audio quality.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Normalize audio to a target dB level."""
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        target_rms = 10 ** (target_db / 20)
        gain = target_rms / rms
        audio = audio * gain
    # Clip to prevent distortion
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def reduce_noise(audio: np.ndarray, sr: int, prop_decrease: float = 0.8) -> np.ndarray:
    """Apply spectral gating noise reduction."""
    try:
        import noisereduce as nr

        # Use non-stationary noise reduction for speech
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=prop_decrease,
            stationary=False,
            n_fft=1024,
            hop_length=256,
        )
        return reduced
    except ImportError:
        print("Warning: noisereduce not installed. Skipping noise reduction.")
        return audio


def apply_high_pass_filter(
    audio: np.ndarray, sr: int, cutoff_hz: float = 80.0
) -> np.ndarray:
    """Apply high-pass filter to remove low-frequency rumble."""
    from scipy.signal import butter, filtfilt

    nyquist = sr / 2
    normalized_cutoff = cutoff_hz / nyquist
    b, a = butter(4, normalized_cutoff, btype="high")
    filtered = filtfilt(b, a, audio)
    return filtered.astype(np.float32)


def process_audio(
    input_path: str,
    output_path: str = None,
    noise_reduce: bool = True,
    normalize: bool = True,
    high_pass: bool = True,
    noise_strength: float = 0.7,
    target_db: float = -3.0,
) -> str:
    """
    Apply post-processing to audio file.

    Args:
        input_path: Path to input audio file
        output_path: Path to output file (default: adds '_enhanced' suffix)
        noise_reduce: Apply noise reduction
        normalize: Apply normalization
        high_pass: Apply high-pass filter (removes rumble below 80Hz)
        noise_strength: Noise reduction strength (0-1, higher = more reduction)
        target_db: Target normalization level in dB

    Returns:
        Path to processed audio file
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_enhanced")
    else:
        output_path = Path(output_path)

    # Load audio
    waveform, sr = torchaudio.load(str(input_path))
    audio = waveform.numpy().squeeze()

    print(f"Processing: {input_path}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio) / sr:.2f}s")

    # Apply high-pass filter to remove rumble
    if high_pass:
        print("  Applying high-pass filter (80Hz)...")
        audio = apply_high_pass_filter(audio, sr)

    # Apply noise reduction
    if noise_reduce:
        print(f"  Applying noise reduction (strength={noise_strength})...")
        audio = reduce_noise(audio, sr, prop_decrease=noise_strength)

    # Normalize
    if normalize:
        print(f"  Normalizing to {target_db}dB...")
        audio = normalize_audio(audio, target_db)

    # Save
    waveform_out = torch.from_numpy(audio).unsqueeze(0).float()
    torchaudio.save(str(output_path), waveform_out, sr)
    print(f"  Saved: {output_path}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Post-process CosyVoice audio output")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output", help="Output audio file")
    parser.add_argument(
        "--no-noise-reduce", action="store_true", help="Skip noise reduction"
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Skip normalization"
    )
    parser.add_argument(
        "--no-high-pass", action="store_true", help="Skip high-pass filter"
    )
    parser.add_argument(
        "--noise-strength",
        type=float,
        default=0.7,
        help="Noise reduction strength (0-1)",
    )
    parser.add_argument(
        "--target-db", type=float, default=-3.0, help="Target normalization level in dB"
    )

    args = parser.parse_args()

    process_audio(
        args.input,
        args.output,
        noise_reduce=not args.no_noise_reduce,
        normalize=not args.no_normalize,
        high_pass=not args.no_high_pass,
        noise_strength=args.noise_strength,
        target_db=args.target_db,
    )


if __name__ == "__main__":
    main()
