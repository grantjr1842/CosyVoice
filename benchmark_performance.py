#!/usr/bin/env python3
import argparse
import json
import statistics
import time

import torch

# Note: Matcha-TTS path no longer needed - using cosyvoice.compat.matcha_compat
from cosyvoice.cli.cosyvoice import AutoModel


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CosyVoice TTS performance.")
    parser.add_argument(
        "--model-dir",
        default="pretrained_models/Fun-CosyVoice3-0.5B",
    )
    parser.add_argument(
        "--prompt-wav",
        default="./asset/interstellar-tars-01-resemble-denoised.wav",
    )
    parser.add_argument(
        "--prompt-text",
        default=(
            "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. "
            "Nothing's changed on that."
        ),
    )
    parser.add_argument(
        "--prompt-prefix",
        default="You are a helpful assistant. Please speak in English.<|endofprompt|>",
    )
    parser.add_argument(
        "--test-text",
        default=(
            "The quick brown fox jumps over the lazy dog. Performance optimization is "
            "the key to a better user experience."
        ),
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--load-trt", action="store_true")
    parser.add_argument("--load-vllm", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trt-concurrent", type=int, default=1)
    parser.add_argument("--use-rl", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _stats(values):
    if not values:
        return {"avg": None, "median": None, "min": None, "max": None}
    return {
        "avg": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def benchmark_tts(
    model_dir,
    test_text,
    prompt_text,
    prompt_wav,
    iterations=3,
    warmup=2,
    stream=True,
    load_trt=False,
    load_vllm=False,
    fp16=False,
    trt_concurrent=1,
    use_rl=False,
):
    print(f"Starting benchmark for model: {model_dir}")

    start_load = time.perf_counter()
    cosyvoice = AutoModel(
        model_dir=model_dir,
        load_trt=load_trt,
        load_vllm=load_vllm,
        fp16=fp16,
        trt_concurrent=trt_concurrent,
        use_rl=use_rl,
    )
    load_time = time.perf_counter() - start_load
    print(f"Model load time: {load_time:.2f}s")

    metrics = {
        "ftl": [],
        "rtf": [],
        "throughput": [],
        "total_time": [],
        "audio_secs": [],
    }

    print("Warming up...")
    for _ in range(warmup):
        for _ in cosyvoice.inference_zero_shot(
            test_text, prompt_text, prompt_wav, stream=stream
        ):
            pass

    print(f"Running {iterations} iterations...")
    for i in range(iterations):
        start_gen = time.perf_counter()
        first_token_time = None
        total_audio_len = 0.0

        for output in cosyvoice.inference_zero_shot(
            test_text, prompt_text, prompt_wav, stream=stream
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter() - start_gen

            audio_len = output["tts_speech"].shape[1] / cosyvoice.sample_rate
            total_audio_len += audio_len

        total_gen_time = time.perf_counter() - start_gen

        if total_audio_len > 0:
            rtf = total_gen_time / total_audio_len
            throughput = total_audio_len / total_gen_time
        else:
            rtf = 0.0
            throughput = 0.0

        metrics["ftl"].append(first_token_time)
        metrics["rtf"].append(rtf)
        metrics["throughput"].append(throughput)
        metrics["total_time"].append(total_gen_time)
        metrics["audio_secs"].append(total_audio_len)

        print(
            f"Iteration {i + 1}: FTL={first_token_time:.3f}s, "
            f"RTF={rtf:.3f}, Audio={total_audio_len:.2f}s"
        )

    summary = {
        "ftl": _stats(metrics["ftl"]),
        "rtf": _stats(metrics["rtf"]),
        "throughput": _stats(metrics["throughput"]),
        "total_time": _stats(metrics["total_time"]),
        "audio_secs": _stats(metrics["audio_secs"]),
    }

    print("\nBenchmark Results (Averages)")
    print(f"First Token Latency: {summary['ftl']['avg']:.3f}s")
    print(f"Real-Time Factor:    {summary['rtf']['avg']:.3f}")
    print(f"Throughput:          {summary['throughput']['avg']:.2f}s/s")
    print(f"Avg Gen Time:        {summary['total_time']['avg']:.2f}s")

    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    return {
        "model_dir": model_dir,
        "load_time": load_time,
        "metrics": metrics,
        "summary": summary,
    }


if __name__ == "__main__":
    args = _parse_args()
    result = benchmark_tts(
        model_dir=args.model_dir,
        test_text=args.test_text,
        prompt_text=args.prompt_prefix + args.prompt_text,
        prompt_wav=args.prompt_wav,
        iterations=args.iterations,
        warmup=args.warmup,
        stream=args.stream,
        load_trt=args.load_trt,
        load_vllm=args.load_vllm,
        fp16=args.fp16,
        trt_concurrent=args.trt_concurrent,
        use_rl=args.use_rl,
    )
    if args.json or args.output:
        output = json.dumps(result, indent=2)
        if args.json:
            print("\nJSON Result")
            print(output)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
                f.write("\n")
