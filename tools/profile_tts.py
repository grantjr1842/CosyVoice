#!/usr/bin/env python3
import argparse
import io
import json
import os
import statistics
import sys
import time
import wave
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _add_repo_to_path() -> None:
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile CosyVoice TTS in Python or via the HTTP server."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    py = sub.add_parser("python", help="Run profiling via the Python pipeline.")
    py.add_argument("--model-dir", required=True)
    py.add_argument("--text", required=True)
    py.add_argument("--prompt-audio", required=True)
    py.add_argument("--prompt-text", default="")
    py.add_argument("--instruct-text", default="")
    py.add_argument("--speed", type=float, default=1.0)
    py.add_argument("--iterations", type=int, default=3)
    py.add_argument("--warmup", type=int, default=2)
    py.add_argument("--stream", action="store_true")
    py.add_argument("--load-trt", action="store_true")
    py.add_argument("--load-vllm", action="store_true")
    py.add_argument("--fp16", action="store_true")
    py.add_argument("--trt-concurrent", type=int, default=1)
    py.add_argument("--use-rl", action="store_true")

    srv = sub.add_parser("server", help="Run profiling against the HTTP server.")
    srv.add_argument("--url", required=True)
    srv.add_argument("--text", required=True)
    srv.add_argument("--prompt-audio", required=True)
    srv.add_argument("--prompt-text", default="")
    srv.add_argument("--speaker", default="")
    srv.add_argument("--speed", type=float, default=1.0)
    srv.add_argument("--iterations", type=int, default=3)
    srv.add_argument("--warmup", type=int, default=1)
    srv.add_argument("--timeout", type=float, default=120.0)

    parser.add_argument("--output", default="")
    return parser.parse_args()


def _duration_from_wav(wav_bytes: bytes, headers) -> float:
    header_val = headers.get("x-audio-duration")
    if header_val:
        try:
            return float(header_val)
        except ValueError:
            pass
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    return frames / float(rate) if rate else 0.0


def _stats(values):
    if not values:
        return {"avg": None, "median": None, "min": None, "max": None}
    return {
        "avg": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _summarize(samples):
    return {
        "iterations": len(samples),
        "ftl": _stats([s["ftl"] for s in samples if s["ftl"] is not None]),
        "rtf": _stats([s["rtf"] for s in samples if s["rtf"] is not None]),
        "throughput": _stats(
            [s["throughput"] for s in samples if s["throughput"] is not None]
        ),
        "total_time": _stats([s["total_time"] for s in samples]),
        "audio_secs": _stats([s["audio_secs"] for s in samples]),
    }


def _run_python(args: argparse.Namespace) -> dict:
    _add_repo_to_path()
    from cosyvoice.cli.cosyvoice import AutoModel

    model_kwargs = {
        "model_dir": args.model_dir,
        "load_trt": args.load_trt,
        "load_vllm": args.load_vllm,
        "fp16": args.fp16,
        "trt_concurrent": args.trt_concurrent,
        "use_rl": args.use_rl,
    }

    load_start = time.perf_counter()
    model = AutoModel(**model_kwargs)
    load_time = time.perf_counter() - load_start

    if args.instruct_text:
        infer = lambda: model.inference_instruct2(
            args.text,
            args.instruct_text,
            args.prompt_audio,
            stream=args.stream,
            speed=args.speed,
        )
    else:
        infer = lambda: model.inference_zero_shot(
            args.text,
            args.prompt_text,
            args.prompt_audio,
            stream=args.stream,
            speed=args.speed,
        )

    for _ in range(args.warmup):
        for _ in infer():
            pass

    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        ftl = None
        audio_secs = 0.0
        for output in infer():
            if ftl is None:
                ftl = time.perf_counter() - start
            audio_secs += output["tts_speech"].shape[1] / model.sample_rate
        total_time = time.perf_counter() - start
        rtf = total_time / audio_secs if audio_secs > 0 else None
        throughput = audio_secs / total_time if total_time > 0 else None
        samples.append(
            {
                "ftl": ftl,
                "rtf": rtf,
                "throughput": throughput,
                "total_time": total_time,
                "audio_secs": audio_secs,
            }
        )

    return {
        "mode": "python",
        "load_time": load_time,
        "samples": samples,
        "summary": _summarize(samples),
    }


def _post_json(url: str, payload: dict, timeout: float):
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    resp = urllib_request.urlopen(req, timeout=timeout)
    ttfb = time.perf_counter() - start
    data = resp.read()
    total = time.perf_counter() - start
    return resp, data, ttfb, total


def _run_server(args: argparse.Namespace) -> dict:
    payload = {
        "text": args.text,
        "prompt_audio": args.prompt_audio,
        "prompt_text": args.prompt_text or None,
        "speaker": args.speaker or None,
        "speed": args.speed,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    for _ in range(args.warmup):
        _post_json(args.url, payload, args.timeout)

    samples = []
    for _ in range(args.iterations):
        resp, data, ttfb, total = _post_json(args.url, payload, args.timeout)
        audio_secs = _duration_from_wav(data, resp.headers)
        rtf = total / audio_secs if audio_secs > 0 else None
        throughput = audio_secs / total if total > 0 else None
        samples.append(
            {
                "ftl": ttfb,
                "rtf": rtf,
                "throughput": throughput,
                "total_time": total,
                "audio_secs": audio_secs,
            }
        )

    return {
        "mode": "server",
        "samples": samples,
        "summary": _summarize(samples),
    }


def main() -> int:
    args = _parse_args()
    try:
        if args.mode == "python":
            result = _run_python(args)
        else:
            result = _run_server(args)
    except HTTPError as exc:
        sys.stderr.write(f"HTTP error {exc.code}: {exc.read().decode('utf-8')}\n")
        return 1
    except URLError as exc:
        sys.stderr.write(f"Network error: {exc}\n")
        return 1

    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
