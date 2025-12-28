#!/usr/bin/env python3
import sys
import time

import numpy as np
import torch

# Add third_party to path
sys.path.append("third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import AutoModel


def benchmark_tts(model_dir, test_text, prompt_text, prompt_wav, iterations=3):
    print(f"\n🚀 Starting benchmark for model: {model_dir}")

    # Measure model loading time
    start_load = time.time()
    cosyvoice = AutoModel(model_dir=model_dir)
    load_time = time.time() - start_load
    print(f"📦 Model Load Time: {load_time:.2f}s")

    metrics = {
        "ftl": [],  # First Token Latency
        "rtf": [],  # Real Time Factor
        "throughput": [],  # Audio seconds per generation second
        "total_time": [],
    }

    # Warmup
    print("🔥 Warming up (compilation might take a while)...")
    for _ in range(2):
        for _ in cosyvoice.inference_zero_shot(
            test_text, prompt_text, prompt_wav, stream=True
        ):
            pass

    print(f"🏃 Running {iterations} iterations...")
    for i in range(iterations):
        start_gen = time.time()
        first_token_time = None
        total_audio_len = 0

        for output in cosyvoice.inference_zero_shot(
            test_text, prompt_text, prompt_wav, stream=True
        ):
            if first_token_time is None:
                first_token_time = time.time() - start_gen

            audio_len = output["tts_speech"].shape[1] / cosyvoice.sample_rate
            total_audio_len += audio_len

        end_gen = time.time()
        total_gen_time = end_gen - start_gen

        rtf = total_gen_time / total_audio_len
        metrics["ftl"].append(first_token_time)
        metrics["rtf"].append(rtf)
        metrics["throughput"].append(total_audio_len / total_gen_time)
        metrics["total_time"].append(total_gen_time)

        print(
            f"Iteration {i + 1}: FTL={first_token_time:.3f}s, RTF={rtf:.3f}, Audio={total_audio_len:.2f}s"
        )

    # Calculate averages
    print("\n" + "=" * 40)
    print("📊 Benchmark Results (Averages)")
    print("=" * 40)
    print(f"First Token Latency: {np.mean(metrics['ftl']):.3f}s")
    print(f"Real-Time Factor:  {np.mean(metrics['rtf']):.3f}")
    print(f"Throughput:         {np.mean(metrics['throughput']):.2f}s/s")
    print(f"Avg Gen Time:       {np.mean(metrics['total_time']):.2f}s")
    print("=" * 40)

    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
    PROMPT_WAV = "./asset/interstellar-tars-01-resemble-denoised.wav"
    PROMPT_TEXT = "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. Nothing's changed on that."
    TEST_TEXT = "The quick brown fox jumps over the lazy dog. Performance optimization is the key to a better user experience."
    PROMPT_PREFIX = (
        "You are a helpful assistant. Please speak in English.<|endofprompt|>"
    )

    benchmark_tts(MODEL_DIR, TEST_TEXT, PROMPT_PREFIX + PROMPT_TEXT, PROMPT_WAV)
