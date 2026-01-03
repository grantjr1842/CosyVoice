#!/usr/bin/env python3
"""Test with the default zero_shot_prompt.wav reference audio."""


# Note: Matcha-TTS path no longer needed - using cosyvoice.compat.matcha_compat

from pathlib import Path

import torchaudio

from cosyvoice.cli.cosyvoice import AutoModel

OUTPUT_DIR = Path("output")
# Try the default zero_shot_prompt.wav
DEFAULT_PROMPT_WAV = "./asset/zero_shot_prompt.wav"
DEFAULT_PROMPT_TEXT = ""  # Will use the audio without explicit transcription
MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"


def test_default_prompt():
    """Test with the model's default zero_shot prompt audio."""
    print("=" * 60)
    print("Testing with default zero_shot_prompt.wav")
    print("=" * 60)

    cosyvoice = AutoModel(model_dir=MODEL_DIR)
    print(f"✅ Model loaded. Sample rate: {cosyvoice.sample_rate} Hz")

    tts_text = "Hello! I am an AI voice assistant powered by Fun-CosyVoice3. How may I help you today?"

    # Use explicit English prompt
    prompt_prefix = "Please speak in English.<|endofprompt|>"
    prompt_text = prompt_prefix + "This is a sample prompt text for testing."

    print(f"\n🎤 Reference voice: {DEFAULT_PROMPT_WAV}")
    print(f'🔊 Synthesizing: "{tts_text[:50]}..."')

    OUTPUT_DIR.mkdir(exist_ok=True)

    for i, output in enumerate(
        cosyvoice.inference_zero_shot(
            tts_text, prompt_text, DEFAULT_PROMPT_WAV, stream=False
        )
    ):
        output_path = OUTPUT_DIR / f"test_default_prompt_{i}.wav"
        torchaudio.save(str(output_path), output["tts_speech"], cosyvoice.sample_rate)
        print(f"   💾 Saved: {output_path}")

    print("\n✨ Test complete!")


if __name__ == "__main__":
    test_default_prompt()
