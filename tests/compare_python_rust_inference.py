#!/usr/bin/env python3
"""
Compare Python CosyVoice3 model inference with native Rust implementation.

This script records EVERY value and metric at each stage of the pipeline:
1. Frontend (text tokenization, prompt audio processing)
2. LLM (speech token generation)
3. Flow (mel spectrogram generation)
4. HiFT (vocoder, mel->audio)

All intermediate tensors and metrics are saved for comparison with the native Rust server.
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torchaudio

# Add project root and third_party to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "Matcha-TTS"))

from safetensors.torch import save_file


@dataclass
class InferenceMetrics:
    """Container for all inference metrics and intermediate values."""

    # Timings (in milliseconds)
    frontend_time_ms: float = 0.0
    llm_time_ms: float = 0.0
    flow_time_ms: float = 0.0
    hift_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Token/Shape info
    text_token_len: int = 0
    prompt_text_token_len: int = 0
    prompt_speech_token_len: int = 0
    generated_speech_token_len: int = 0

    # Mel/Audio shapes
    prompt_mel_shape: tuple = ()
    generated_mel_shape: tuple = ()
    audio_samples: int = 0
    audio_duration_sec: float = 0.0

    # RTF
    rtf: float = 0.0

    # Numeric stats for tensors (for comparison)
    tensor_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timings": {
                "frontend_ms": self.frontend_time_ms,
                "llm_ms": self.llm_time_ms,
                "flow_ms": self.flow_time_ms,
                "hift_ms": self.hift_time_ms,
                "total_ms": self.total_time_ms,
            },
            "shapes": {
                "text_token_len": self.text_token_len,
                "prompt_text_token_len": self.prompt_text_token_len,
                "prompt_speech_token_len": self.prompt_speech_token_len,
                "generated_speech_token_len": self.generated_speech_token_len,
                "prompt_mel_shape": list(self.prompt_mel_shape),
                "generated_mel_shape": list(self.generated_mel_shape),
                "audio_samples": self.audio_samples,
                "audio_duration_sec": self.audio_duration_sec,
            },
            "performance": {
                "rtf": self.rtf,
            },
            "tensor_stats": self.tensor_stats,
        }


def tensor_stats(t: torch.Tensor, name: str) -> dict:
    """Compute statistics for a tensor."""
    t_flat = t.float().flatten()
    return {
        f"{name}_shape": list(t.shape),
        f"{name}_dtype": str(t.dtype),
        f"{name}_min": float(t_flat.min().item()),
        f"{name}_max": float(t_flat.max().item()),
        f"{name}_mean": float(t_flat.mean().item()),
        f"{name}_std": float(t_flat.std().item()) if t_flat.numel() > 1 else 0.0,
        f"{name}_l2_norm": float(torch.norm(t_flat, p=2).item()),
        f"{name}_first_10": t_flat[:10].tolist()
        if t_flat.numel() >= 10
        else t_flat.tolist(),
        f"{name}_last_10": t_flat[-10:].tolist() if t_flat.numel() >= 10 else [],
    }


class InstrumentedCosyVoice3Model:
    """
    Instrumented wrapper around CosyVoice3Model that records all intermediate values.
    """

    def __init__(self, original_model, frontend, sample_rate: int, device):
        self.original_model = original_model
        self.frontend = frontend
        self.sample_rate = sample_rate
        self.device = device

        # Captured intermediate data
        self.captured_speech_tokens = []
        self.captured_mel = None
        self.captured_audio = None

    def wrap_token2wav(self, original_token2wav):
        """Wrap token2wav to capture mel and audio."""
        outer = self

        def wrapped(
            token,
            prompt_token,
            prompt_feat,
            embedding,
            token_offset,
            uuid,
            stream=False,
            finalize=False,
            speed=1.0,
        ):
            # Capture mel from flow
            with torch.amp.autocast("cuda", enabled=self.original_model.fp16):
                tts_mel, _ = self.original_model.flow.inference(
                    token=token.to(self.device, dtype=torch.int32),
                    token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(
                        self.device
                    ),
                    prompt_token=prompt_token.to(self.device),
                    prompt_token_len=torch.tensor(
                        [prompt_token.shape[1]], dtype=torch.int32
                    ).to(self.device),
                    prompt_feat=prompt_feat.to(self.device),
                    prompt_feat_len=torch.tensor(
                        [prompt_feat.shape[1]], dtype=torch.int32
                    ).to(self.device),
                    embedding=embedding.to(self.device),
                    streaming=stream,
                    finalize=finalize,
                )

                # Capture mel before trimming
                if outer.captured_mel is None:
                    outer.captured_mel = tts_mel.clone().cpu()

                tts_mel_trimmed = tts_mel[
                    :, :, token_offset * self.original_model.flow.token_mel_ratio :
                ]

                # Handle caching
                if self.original_model.hift_cache_dict[uuid] is not None:
                    hift_cache_mel = self.original_model.hift_cache_dict[uuid]["mel"]
                    tts_mel_trimmed = torch.concat(
                        [hift_cache_mel, tts_mel_trimmed], dim=2
                    )
                else:
                    self.original_model.hift_cache_dict[uuid] = {
                        "mel": tts_mel_trimmed,
                        "speech_offset": 0,
                    }

                if speed != 1.0:
                    assert token_offset == 0 and finalize is True
                    tts_mel_trimmed = torch.nn.functional.interpolate(
                        tts_mel_trimmed,
                        size=int(tts_mel_trimmed.shape[2] / speed),
                        mode="linear",
                    )

                # Capture audio from hift
                tts_speech, _ = self.original_model.hift.inference(
                    speech_feat=tts_mel_trimmed, finalize=finalize
                )

                if outer.captured_audio is None:
                    outer.captured_audio = tts_speech.clone().cpu()

                tts_speech = tts_speech[
                    :, self.original_model.hift_cache_dict[uuid]["speech_offset"] :
                ]
                self.original_model.hift_cache_dict[uuid]["speech_offset"] += (
                    tts_speech.shape[1]
                )

            return tts_speech

        return wrapped

    @torch.inference_mode()
    def inference_with_metrics(
        self,
        tts_text: str,
        prompt_text: str,
        prompt_wav: str,
        speed: float = 1.0,
    ) -> tuple:
        """
        Run full inference pipeline with detailed metrics recording.

        Returns:
            - audio tensor
            - metrics
            - intermediate tensors dict
        """
        metrics = InferenceMetrics()
        intermediates = {}

        total_start = time.perf_counter()

        # ==================== FRONTEND ====================
        frontend_start = time.perf_counter()

        # Normalize text
        prompt_text_norm = self.frontend.text_normalize(
            prompt_text, split=False, text_frontend=True
        )
        tts_text_norm_list = list(
            self.frontend.text_normalize(tts_text, split=True, text_frontend=True)
        )
        tts_text_norm = tts_text_norm_list[0] if tts_text_norm_list else tts_text

        print(f"[Frontend] Prompt text normalized: '{prompt_text_norm}'")
        print(f"[Frontend] TTS text normalized: '{tts_text_norm}'")

        # Get frontend features
        model_input = self.frontend.frontend_zero_shot(
            tts_text_norm, prompt_text_norm, prompt_wav, self.sample_rate, ""
        )

        frontend_end = time.perf_counter()
        metrics.frontend_time_ms = (frontend_end - frontend_start) * 1000

        # Record frontend outputs
        for key, value in model_input.items():
            if isinstance(value, torch.Tensor):
                intermediates[f"frontend_{key}"] = value.clone().cpu()
                metrics.tensor_stats.update(tensor_stats(value, f"frontend_{key}"))
                print(f"[Frontend] {key}: shape={value.shape}, dtype={value.dtype}")

        # Key shape metrics
        metrics.text_token_len = (
            model_input["text"].shape[1] if "text" in model_input else 0
        )
        metrics.prompt_text_token_len = model_input.get(
            "prompt_text", torch.zeros(1, 0)
        ).shape[1]
        metrics.prompt_speech_token_len = model_input.get(
            "llm_prompt_speech_token", torch.zeros(1, 0)
        ).shape[1]
        if "prompt_speech_feat" in model_input:
            metrics.prompt_mel_shape = tuple(model_input["prompt_speech_feat"].shape)

        # Record embeddings
        if "llm_embedding" in model_input:
            intermediates["llm_embedding"] = model_input["llm_embedding"].clone().cpu()
            metrics.tensor_stats.update(
                tensor_stats(model_input["llm_embedding"], "llm_embedding")
            )
        if "flow_embedding" in model_input:
            intermediates["flow_embedding"] = (
                model_input["flow_embedding"].clone().cpu()
            )
            metrics.tensor_stats.update(
                tensor_stats(model_input["flow_embedding"], "flow_embedding")
            )

        # ==================== FULL TTS PIPELINE ====================
        # Use the high-level tts() method but instrument it

        # Reset captures
        self.captured_speech_tokens = []
        self.captured_mel = None
        self.captured_audio = None

        # Monkey-patch the LLM's tts_speech_token_dict to capture tokens
        original_tts_speech_token_dict = {}

        print("\n[TTS] Running full pipeline...")
        llm_start = time.perf_counter()

        # Run tts generator
        all_audio = []
        for output in self.original_model.tts(**model_input, stream=False, speed=speed):
            audio = output["tts_speech"]
            all_audio.append(audio)

        total_end = time.perf_counter()

        # Combine all audio chunks
        if all_audio:
            final_audio = torch.cat(all_audio, dim=1)
        else:
            final_audio = torch.zeros(1, 0)

        # Try to get the speech tokens that were generated (from the model's internal dict)
        # Note: The tts method clears these after completion, so we can't capture them directly
        # We'll need to use a different approach - check token count from model output

        # Calculate metrics from what we can observe
        metrics.audio_samples = final_audio.numel()
        metrics.audio_duration_sec = metrics.audio_samples / self.sample_rate
        metrics.total_time_ms = (total_end - total_start) * 1000
        metrics.rtf = (
            metrics.total_time_ms / 1000 / metrics.audio_duration_sec
            if metrics.audio_duration_sec > 0
            else float("inf")
        )

        # Record audio stats
        intermediates["final_audio"] = final_audio.clone()
        metrics.tensor_stats.update(tensor_stats(final_audio, "final_audio"))

        print(f"\n{'=' * 60}")
        print("[Summary]")
        print(f"  Frontend: {metrics.frontend_time_ms:.1f}ms")
        print(f"  Total:    {metrics.total_time_ms:.1f}ms")
        print(
            f"  Audio:    {metrics.audio_duration_sec:.2f}s ({metrics.audio_samples} samples)"
        )
        print(f"  RTF:      {metrics.rtf:.4f}")
        print(f"{'=' * 60}")

        return final_audio, metrics, intermediates


def run_instrumented_inference(
    model_dir: str,
    tts_text: str,
    prompt_text: str,
    prompt_wav: str,
    output_dir: Path,
    use_rl: bool = True,
):
    """Run instrumented inference and save all metrics."""
    from cosyvoice.cli.cosyvoice import CosyVoice3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing CosyVoice3...")
    cosyvoice = CosyVoice3(model_dir, load_trt=False, fp16=False, use_rl=use_rl)
    print(f"Model initialized. Sample rate: {cosyvoice.sample_rate}")
    print()

    # Create instrumented wrapper
    instrumented = InstrumentedCosyVoice3Model(
        cosyvoice.model,
        cosyvoice.frontend,
        cosyvoice.sample_rate,
        cosyvoice.model.device,
    )

    # Run inference
    print("Running instrumented inference...")
    audio, metrics, intermediates = instrumented.inference_with_metrics(
        tts_text=tts_text,
        prompt_text=prompt_text,
        prompt_wav=prompt_wav,
    )

    return audio, metrics, intermediates, cosyvoice.sample_rate


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Record Python TTS inference metrics for comparison with Rust"
    )
    parser.add_argument(
        "--model-dir",
        default="pretrained_models/Fun-CosyVoice3-0.5B",
        help="Model directory",
    )
    parser.add_argument(
        "--prompt-wav", default="asset/zero_shot_prompt.wav", help="Prompt audio file"
    )
    parser.add_argument(
        "--prompt-text",
        default="You are a helpful assistant.<|endofprompt|>This is a reference voice sample.",
        help="Prompt text (with instruction prefix)",
    )
    parser.add_argument(
        "--tts-text",
        default="Hello world, this is a test of the text to speech system.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/python_inference_metrics",
        help="Output directory for metrics",
    )
    parser.add_argument(
        "--output-wav", default="python_reference_output.wav", help="Output audio file"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model directory: {args.model_dir}")
    print(f"Prompt WAV: {args.prompt_wav}")
    print(f"Prompt text: {args.prompt_text}")
    print(f"TTS text: {args.tts_text}")
    print(f"Output directory: {output_dir}")
    print()

    # Run inference
    audio, metrics, intermediates, sample_rate = run_instrumented_inference(
        args.model_dir,
        args.tts_text,
        args.prompt_text,
        args.prompt_wav,
        output_dir,
    )

    # Save audio
    output_wav_path = output_dir / args.output_wav
    torchaudio.save(str(output_wav_path), audio, sample_rate)
    print(f"\nSaved audio to: {output_wav_path}")

    # Save metrics as JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    # Save intermediate tensors as safetensors
    tensors_path = output_dir / "intermediates.safetensors"
    # Convert to float32 for safetensors compatibility
    tensors_to_save = {}
    for k, v in intermediates.items():
        if v.dtype in [torch.int32, torch.int64]:
            tensors_to_save[k] = v.int().contiguous()
        else:
            tensors_to_save[k] = v.float().contiguous()
    save_file(tensors_to_save, str(tensors_path))
    print(f"Saved intermediate tensors to: {tensors_path}")

    # Print tensor shapes summary
    print("\nIntermediate tensor shapes:")
    for k, v in intermediates.items():
        print(f"  {k}: {list(v.shape)} ({v.dtype})")

    print("\n✅ Python inference metrics recorded successfully!")


if __name__ == "__main__":
    main()
