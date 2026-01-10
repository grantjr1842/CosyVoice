#!/usr/bin/env python3
"""
Python Benchmark for CosyVoice3 Parity Comparison.
Benchmarks RTF and saves audio outputs to output/benchmark/python.
"""
import json
import time
import torch
import torchaudio
from pathlib import Path
from cosyvoice.cli.cosyvoice import AutoModel

# Configuration matches Rust benchmark defaults
MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
PROMPT_WAV = "./asset/interstellar-tars-01-resemble-denoised.wav"
PROMPT_TEXT = "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. Nothing's changed on that."
TEST_TEXT = "The quick brown fox jumps over the lazy dog. Performance optimization is key."
OUTPUT_DIR = Path("output/benchmark/python")
ITERATIONS = 3

def main():
    print(f"Initializing Python CosyVoice Engine from {MODEL_DIR}...")
    start_load = time.time()
    cosyvoice = AutoModel(model_dir=MODEL_DIR)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    prompt_prefix = "Please speak in English.<|endofprompt|>"
    full_prompt_text = prompt_prefix + PROMPT_TEXT

    print("Warming up...")
    # Warmup
    list(cosyvoice.inference_zero_shot(TEST_TEXT, full_prompt_text, PROMPT_WAV, stream=False))

    print(f"Running {ITERATIONS} iterations...")
    total_duration = 0.0
    metrics = {
        "model_dir": MODEL_DIR,
        "test_text": TEST_TEXT,
        "prompt_text": PROMPT_TEXT,
        "iterations": ITERATIONS,
        "load_time_s": load_time,
        "sample_rate": cosyvoice.sample_rate,
        "per_iteration": []
    }

    for i in range(ITERATIONS):
        print(f"Iteration {i+1} starting...")
        start = time.time()

        # Inference
        outputs = list(cosyvoice.inference_zero_shot(TEST_TEXT, full_prompt_text, PROMPT_WAV, stream=False))

        duration = time.time() - start
        total_duration += duration

        # Save output
        if outputs:
            output = outputs[0] # Assume single segment for short text
            audio = output['tts_speech'] # Tensor [1, samples]

            output_path = OUTPUT_DIR / f"iter_{i}.wav"
            torchaudio.save(str(output_path), audio, cosyvoice.sample_rate)

            # Calculate audio duration
            audio_samples = audio.shape[1]
            audio_duration = audio_samples / cosyvoice.sample_rate

            rtf = duration / audio_duration

            iter_metrics = {
                "iteration": i,
                "synthesis_time_s": duration,
                "audio_duration_s": audio_duration,
                "audio_samples": audio_samples,
                "rtf": rtf,
                "output_file": str(output_path)
            }
            metrics["per_iteration"].append(iter_metrics)

            print(f"Iteration {i+1}: {duration:.2f}s (Audio: {audio_duration:.2f}s, RTF: {rtf:.3f})")
            print(f"  Saved: {output_path}")

    avg_duration = total_duration / ITERATIONS
    avg_rtf = sum(m["rtf"] for m in metrics["per_iteration"]) / len(metrics["per_iteration"])
    metrics["avg_synthesis_time_s"] = avg_duration
    metrics["avg_rtf"] = avg_rtf
    metrics["total_duration_s"] = total_duration

    # Save metrics to JSON
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    print(f"\nAverage Synthesis Time: {avg_duration:.2f}s")
    print(f"Average RTF: {avg_rtf:.3f}")
    print("Benchmark complete.")

if __name__ == "__main__":
    main()
