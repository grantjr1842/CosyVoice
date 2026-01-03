import wave

import numpy as np


def analyze_wav(filename):
    with wave.open(filename, "rb") as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        data = f.readframes(nframes)
        samples = np.frombuffer(data, dtype=np.int16)

        # Audio normalization to float
        samples_f = samples.astype(np.float32) / 32768.0

        print(f"File: {filename}")
        print(f"  Channels: {nchannels}, Rate: {framerate}, Frames: {nframes}")
        print(f"  Max Amplitude: {np.max(np.abs(samples_f)):.4f}")
        print(f"  Mean: {np.mean(samples_f):.4f}")
        print(f"  Std Dev: {np.std(samples_f):.4f}")
        print(f"  Zero Crossings: {np.sum(samples_f[1:] * samples_f[:-1] < 0)}")
        print(f"  NaNs: {np.isnan(samples_f).sum()}, Infs: {np.isinf(samples_f).sum()}")

        # Check for silence vs sound
        energy = np.mean(samples_f**2)
        print(f"  Energy: {energy:.6f}")

        # Power spectrum peak
        fft = np.abs(np.fft.rfft(samples_f))
        peak_freq = np.argmax(fft) * (framerate / nframes)
        print(f"  Peak Frequency: {peak_freq:.2f} Hz")


if __name__ == "__main__":
    analyze_wav("native_output.wav")
