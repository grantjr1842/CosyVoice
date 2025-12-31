#!/usr/bin/env python3
"""
Analyze and compare audio files to find differences between Python and Rust TTS outputs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import scipy.io.wavfile as wavfile
    import scipy.signal as signal
except ImportError:
    print("Installing scipy...")
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "scipy"], check=True)
    import scipy.io.wavfile as wavfile
    import scipy.signal as signal


def load_wav(path: str) -> tuple:
    """Load a WAV file and return sample rate and normalized samples."""
    sample_rate, samples = wavfile.read(path)

    # Convert to float32, normalize
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype == np.int32:
        samples = samples.astype(np.float32) / 2147483648.0
    elif samples.dtype == np.float32:
        pass  # Already float32
    else:
        samples = samples.astype(np.float32)

    # Handle stereo -> mono
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)

    return sample_rate, samples


def compute_audio_stats(samples: np.ndarray, name: str) -> dict:
    """Compute statistics for audio samples."""
    return {
        "name": name,
        "length_samples": len(samples),
        "duration_sec": len(samples) / 24000.0,  # Assume 24kHz
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "rms": float(np.sqrt(np.mean(samples**2))),
        "peak_db": float(20 * np.log10(np.max(np.abs(samples)) + 1e-8)),
        "rms_db": float(20 * np.log10(np.sqrt(np.mean(samples**2)) + 1e-8)),
        "zero_crossings": int(np.sum(np.abs(np.diff(np.sign(samples))) > 0)),
    }


def compute_spectral_stats(samples: np.ndarray, sample_rate: int) -> dict:
    """Compute spectral statistics."""
    # Compute magnitude spectrogram
    f, t, Sxx = signal.spectrogram(samples, fs=sample_rate, nperseg=1024, noverlap=512)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # Spectral centroid (weighted mean frequency)
    spectral_centroid = np.sum(f[:, np.newaxis] * Sxx, axis=0) / (
        np.sum(Sxx, axis=0) + 1e-10
    )

    # Spectral flatness (geometric mean / arithmetic mean)
    log_Sxx = np.log(Sxx + 1e-10)
    spectral_flatness = np.exp(np.mean(log_Sxx, axis=0)) / (
        np.mean(Sxx, axis=0) + 1e-10
    )

    return {
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "spectral_flatness_mean": float(np.mean(spectral_flatness)),
        "spectral_flatness_std": float(np.std(spectral_flatness)),
        "spectrogram_mean_db": float(np.mean(Sxx_db)),
        "spectrogram_std_db": float(np.std(Sxx_db)),
        "low_freq_energy_ratio": float(
            np.sum(Sxx[: len(f) // 4, :]) / (np.sum(Sxx) + 1e-10)
        ),
        "high_freq_energy_ratio": float(
            np.sum(Sxx[3 * len(f) // 4 :, :]) / (np.sum(Sxx) + 1e-10)
        ),
    }


def compare_audio_files(file1: str, file2: str, output_path: str = None) -> dict:
    """Compare two audio files and return differences."""
    print(f"Comparing: {file1}")
    print(f"With:      {file2}")
    print()

    sr1, samples1 = load_wav(file1)
    sr2, samples2 = load_wav(file2)

    stats1 = compute_audio_stats(samples1, Path(file1).name)
    stats2 = compute_audio_stats(samples2, Path(file2).name)

    print(f"{'Metric':<30} {'File 1':<20} {'File 2':<20} {'Diff':<20}")
    print("=" * 90)

    for key in stats1:
        if key == "name":
            continue
        v1 = stats1[key]
        v2 = stats2[key]
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            diff = v2 - v1
            print(f"{key:<30} {v1:<20.6f} {v2:<20.6f} {diff:<20.6f}")
        else:
            print(f"{key:<30} {v1:<20} {v2:<20}")

    print()

    # Spectral analysis
    print("Spectral Analysis:")
    print("=" * 90)

    spec_stats1 = compute_spectral_stats(samples1, sr1)
    spec_stats2 = compute_spectral_stats(samples2, sr2)

    for key in spec_stats1:
        v1 = spec_stats1[key]
        v2 = spec_stats2[key]
        diff = v2 - v1
        print(f"{key:<30} {v1:<20.6f} {v2:<20.6f} {diff:<20.6f}")

    # Sample-level comparison (if same length, or on overlapping portion)
    print()
    print("Sample-level Analysis (first 1000 samples):")
    print("=" * 90)

    min_len = min(len(samples1), len(samples2), 1000)
    s1_head = samples1[:min_len]
    s2_head = samples2[:min_len]

    mae = np.mean(np.abs(s1_head - s2_head))
    mse = np.mean((s1_head - s2_head) ** 2)
    max_diff = np.max(np.abs(s1_head - s2_head))
    correlation = np.corrcoef(s1_head, s2_head)[0, 1]

    print(f"Mean Absolute Error:  {mae:.8f}")
    print(f"Mean Squared Error:   {mse:.8f}")
    print(f"Max Difference:       {max_diff:.8f}")
    print(f"Correlation:          {correlation:.8f}")

    # Check if audio is essentially garbage (very low correlation, high flatness, etc.)
    print()
    print("Quality Assessment:")
    print("=" * 90)

    issues1 = []
    issues2 = []

    # Check for clipping
    if np.max(np.abs(samples1)) > 0.99:
        issues1.append("Possible clipping detected")
    if np.max(np.abs(samples2)) > 0.99:
        issues2.append("Possible clipping detected")

    # Check for very low energy (near silence)
    if stats1["rms"] < 0.001:
        issues1.append(f"Very low energy (RMS={stats1['rms']:.6f}) - nearly silent")
    if stats2["rms"] < 0.001:
        issues2.append(f"Very low energy (RMS={stats2['rms']:.6f}) - nearly silent")

    # Check for very high spectral flatness (noise-like)
    if spec_stats1["spectral_flatness_mean"] > 0.5:
        issues1.append(
            f"High spectral flatness ({spec_stats1['spectral_flatness_mean']:.3f}) - noise-like"
        )
    if spec_stats2["spectral_flatness_mean"] > 0.5:
        issues2.append(
            f"High spectral flatness ({spec_stats2['spectral_flatness_mean']:.3f}) - noise-like"
        )

    # Check for unusual frequency distribution
    if spec_stats1["high_freq_energy_ratio"] > 0.3:
        issues1.append(
            f"High energy in high frequencies ({spec_stats1['high_freq_energy_ratio']:.3f})"
        )
    if spec_stats2["high_freq_energy_ratio"] > 0.3:
        issues2.append(
            f"High energy in high frequencies ({spec_stats2['high_freq_energy_ratio']:.3f})"
        )

    print(f"File 1 ({Path(file1).name}):")
    if issues1:
        for issue in issues1:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✅ No obvious quality issues detected")

    print(f"\nFile 2 ({Path(file2).name}):")
    if issues2:
        for issue in issues2:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✅ No obvious quality issues detected")

    # Build result dict
    result = {
        "file1": {
            "path": file1,
            "stats": stats1,
            "spectral": spec_stats1,
            "issues": issues1,
        },
        "file2": {
            "path": file2,
            "stats": stats2,
            "spectral": spec_stats2,
            "issues": issues2,
        },
        "comparison": {
            "mae": float(mae),
            "mse": float(mse),
            "max_diff": float(max_diff),
            "correlation": float(correlation),
        },
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved comparison to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare audio files")
    parser.add_argument("file1", help="First audio file (reference)")
    parser.add_argument("file2", help="Second audio file (to compare)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    args = parser.parse_args()

    compare_audio_files(args.file1, args.file2, args.output)


if __name__ == "__main__":
    main()
