import wave

import numpy as np


def analyze_wav(filename):
    with wave.open(filename, "rb") as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        data = f.readframes(nframes)
        samples = np.frombuffer(data, dtype=np.int16)

        samples_f = samples.astype(np.float32) / 32768.0

        print(f"\n=== {filename} ===")
        print(
            f"  Channels: {nchannels}, Rate: {framerate}, Frames: {nframes}, Duration: {nframes / framerate:.2f}s"
        )
        print(f"  Max Amplitude: {np.max(np.abs(samples_f)):.4f}")
        print(f"  Mean: {np.mean(samples_f):.4f}")
        print(f"  Std Dev: {np.std(samples_f):.4f}")
        print(f"  Zero Crossings: {np.sum(samples_f[1:] * samples_f[:-1] < 0)}")
        print(f"  Energy: {np.mean(samples_f**2):.6f}")

        # Power spectrum peak
        fft = np.abs(np.fft.rfft(samples_f))
        peak_freq = np.argmax(fft) * (framerate / nframes)
        print(f"  Peak Frequency: {peak_freq:.2f} Hz")


if __name__ == "__main__":
    analyze_wav("python_hift_output.wav")
    analyze_wav("native_output.wav")
