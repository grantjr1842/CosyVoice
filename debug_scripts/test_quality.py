#!/usr/bin/env python3
"""Test different inference modes to find best audio quality."""

import sys

sys.path.append("third_party/Matcha-TTS")

from pathlib import Path

import torchaudio

from cosyvoice.cli.cosyvoice import AutoModel

OUTPUT_DIR = Path("output")
DEFAULT_PROMPT_WAV = "./asset/interstellar-tars-01-resemble-denoised.wav"
MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"


def test_quality():
    """Test inference_instruct2 mode with explicit quality instructions."""
    print("=" * 60)
    print("Audio Quality Test - Using inference_instruct2 mode")
    print("=" * 60)

    cosyvoice = AutoModel(model_dir=MODEL_DIR)
    print(f"✅ Model loaded. Sample rate: {cosyvoice.sample_rate} Hz")

    tts_text = "Hello! I am an AI voice assistant powered by Fun-CosyVoice3. How may I help you today?"

    # Use instruct mode with explicit quality instructions
    instruct_text = "Please speak in English with a clear, natural, and warm voice."

    print(f"\n🎤 Reference voice: {DEFAULT_PROMPT_WAV}")
    print(f'📝 Instruction: "{instruct_text}"')
    print(f'🔊 Synthesizing: "{tts_text[:50]}..."')

    OUTPUT_DIR.mkdir(exist_ok=True)

    for i, output in enumerate(
        cosyvoice.inference_instruct2(
            tts_text, instruct_text, DEFAULT_PROMPT_WAV, stream=False
        )
    ):
        output_path = OUTPUT_DIR / f"quality_test_instruct2_{i}.wav"
        torchaudio.save(str(output_path), output["tts_speech"], cosyvoice.sample_rate)
        print(f"   💾 Saved: {output_path}")

    print("\n✨ Quality test complete!")
    print("\nCompare:")
    print("  - output/voice_clone_0_0.wav (inference_zero_shot)")
    print("  - output/quality_test_instruct2_0.wav (inference_instruct2)")


if __name__ == "__main__":
    test_quality()
