#!/usr/bin/env python3
"""
Audio Post-Processing v2 - Gentler approach without aggressive noise reduction.

Focus on:
- Upsampling to 48kHz for better playback compatibility
- Gentle EQ (presence boost, low-cut)
- Light dynamic range compression
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio


def upsample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int = 48000) -> tuple:
    """Upsample audio to higher sample rate."""
    if orig_sr == target_sr:
        return audio, orig_sr
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(audio), target_sr


def apply_eq(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply gentle EQ: low-cut at 60Hz, presence boost at 3kHz."""
    from scipy.signal import butter, sosfilt

    # Low-cut filter at 60Hz (remove rumble)
    sos_low = butter(2, 60 / (sr / 2), btype="high", output="sos")
    audio = sosfilt(sos_low, audio).astype(np.float32)

    # Gentle presence boost around 3kHz (adds clarity)
    # Using a simple shelf-like approach
    sos_high = butter(1, 2500 / (sr / 2), btype="high", output="sos")
    high_freq = sosfilt(sos_high, audio).astype(np.float32)
    audio = audio + 0.15 * high_freq  # Add 15% of high frequencies

    return audio


def soft_clip(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Apply soft clipping to prevent harsh distortion."""
    # Soft knee compression near threshold
    mask = np.abs(audio) > threshold
    audio[mask] = np.sign(audio[mask]) * (
        threshold
        + (1 - threshold) * np.tanh((np.abs(audio[mask]) - threshold) / (1 - threshold))
    )
    return audio


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize to target peak level."""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio


def process_audio_v2(
    input_path: str,
    output_path: str = None,
    upsample: bool = True,
    target_sr: int = 48000,
    eq: bool = True,
    normalize: bool = True,
) -> str:
    """
    Apply gentle post-processing to audio file.

    Args:
        input_path: Path to input audio file
        output_path: Path to output file (default: adds '_v2' suffix)
        upsample: Upsample to 48kHz
        target_sr: Target sample rate when upsampling
        eq: Apply gentle EQ
        normalize: Normalize peak level

    Returns:
        Path to processed audio file
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_v2")
    else:
        output_path = Path(output_path)

    # Load audio
    waveform, sr = torchaudio.load(str(input_path))

    print(f"Processing: {input_path}")
    print(f"  Input: {sr} Hz, {waveform.shape[1] / sr:.2f}s")

    # Upsample first (before any processing)
    if upsample:
        print(f"  Upsampling to {target_sr}Hz...")
        waveform, sr = upsample_audio(waveform, sr, target_sr)

    audio = waveform.numpy().squeeze()

    # Apply gentle EQ
    if eq:
        print("  Applying gentle EQ (low-cut + presence)...")
        audio = apply_eq(audio, sr)

    # Soft clip to prevent harsh peaks
    audio = soft_clip(audio)

    # Normalize
    if normalize:
        print("  Normalizing peak level...")
        audio = normalize_peak(audio)

    # Save
    waveform_out = torch.from_numpy(audio).unsqueeze(0).float()
    torchaudio.save(str(output_path), waveform_out, sr)
    print(f"  Output: {sr} Hz")
    print(f"  Saved: {output_path}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Post-process audio (v2 - gentle)")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output", help="Output audio file")
    parser.add_argument("--no-upsample", action="store_true", help="Skip upsampling")
    parser.add_argument("--no-eq", action="store_true", help="Skip EQ")
    parser.add_argument(
        "--no-normalize", action="store_true", help="Skip normalization"
    )
    parser.add_argument(
        "--target-sr", type=int, default=48000, help="Target sample rate"
    )

    args = parser.parse_args()

    process_audio_v2(
        args.input,
        args.output,
        upsample=not args.no_upsample,
        target_sr=args.target_sr,
        eq=not args.no_eq,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
