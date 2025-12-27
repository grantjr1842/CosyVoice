#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fun-CosyVoice3 Example - Voice Cloning Demo

This example demonstrates zero-shot voice cloning using the Fun-CosyVoice3-0.5B-2512 model.
The default reference voice is the interstellar-tars voice clip.
"""

import sys

sys.path.append("third_party/Matcha-TTS")

import torchaudio

from cosyvoice.cli.cosyvoice import AutoModel

# =============================================================================
# Default Voice Cloning Configuration
# =============================================================================

# Reference voice clip for voice cloning
DEFAULT_PROMPT_WAV = "./asset/interstellar-tars-01-resemble-denoised.wav"

# Transcription of the reference voice clip
DEFAULT_PROMPT_TEXT = "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. Nothing's changed on that."

# Model directory (will auto-download if not present)
DEFAULT_MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"


def voice_cloning_example():
    """
    Zero-shot voice cloning example.

    Uses the default reference voice to synthesize new text.
    """
    print("=" * 60)
    print("Fun-CosyVoice3 Voice Cloning Example")
    print("=" * 60)

    # Initialize model
    print(f"\n📦 Loading model from: {DEFAULT_MODEL_DIR}")
    cosyvoice = AutoModel(model_dir=DEFAULT_MODEL_DIR)
    print(f"✅ Model loaded. Sample rate: {cosyvoice.sample_rate} Hz")

    # Example texts to synthesize
    texts = [
        "Hello! I am an AI voice assistant powered by Fun-CosyVoice3. How may I help you today?",
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    ]

    # Prompt prefix for better quality (recommended for CosyVoice3)
    prompt_prefix = "You are a helpful assistant.<|endofprompt|>"
    full_prompt_text = prompt_prefix + DEFAULT_PROMPT_TEXT

    print(f"\n🎤 Reference voice: {DEFAULT_PROMPT_WAV}")
    print(f'📝 Reference transcription: "{DEFAULT_PROMPT_TEXT}"')

    for idx, tts_text in enumerate(texts):
        print(f'\n🔊 Synthesizing [{idx + 1}/{len(texts)}]: "{tts_text[:50]}..."')

        for i, output in enumerate(
            cosyvoice.inference_zero_shot(
                tts_text, full_prompt_text, DEFAULT_PROMPT_WAV, stream=False
            )
        ):
            output_path = f"output_voice_clone_{idx}_{i}.wav"
            torchaudio.save(output_path, output["tts_speech"], cosyvoice.sample_rate)
            print(f"   💾 Saved: {output_path}")

    print("\n✨ Voice cloning complete!")


def cross_lingual_example():
    """
    Cross-lingual synthesis example.

    Synthesizes text in a different language than the reference voice.
    """
    print("\n" + "=" * 60)
    print("Fun-CosyVoice3 Cross-Lingual Example")
    print("=" * 60)

    cosyvoice = AutoModel(model_dir=DEFAULT_MODEL_DIR)

    # Chinese text with English reference voice
    tts_text = "You are a helpful assistant.<|endofprompt|>你好，我是一个人工智能语音助手。很高兴为您服务！"

    print(f"\n🎤 Reference voice: {DEFAULT_PROMPT_WAV}")
    print("📝 Synthesizing Chinese text with English reference voice")

    for i, output in enumerate(
        cosyvoice.inference_cross_lingual(tts_text, DEFAULT_PROMPT_WAV, stream=False)
    ):
        output_path = f"output_cross_lingual_{i}.wav"
        torchaudio.save(output_path, output["tts_speech"], cosyvoice.sample_rate)
        print(f"   💾 Saved: {output_path}")

    print("\n✨ Cross-lingual synthesis complete!")


def instruct_example():
    """
    Instruction-controlled synthesis example.

    Uses natural language instructions to control speech style.
    """
    print("\n" + "=" * 60)
    print("Fun-CosyVoice3 Instruction Example")
    print("=" * 60)

    cosyvoice = AutoModel(model_dir=DEFAULT_MODEL_DIR)

    # Text with style instruction
    tts_text = "Today is a beautiful day. The sun is shining and birds are singing."
    instruct_text = (
        "You are a helpful assistant. Please speak slowly and calmly.<|endofprompt|>"
    )

    print(f"\n🎤 Reference voice: {DEFAULT_PROMPT_WAV}")
    print(f'📝 Instruction: "{instruct_text}"')
    print(f'📝 Text: "{tts_text}"')

    for i, output in enumerate(
        cosyvoice.inference_instruct2(
            tts_text, instruct_text, DEFAULT_PROMPT_WAV, stream=False
        )
    ):
        output_path = f"output_instruct_{i}.wav"
        torchaudio.save(output_path, output["tts_speech"], cosyvoice.sample_rate)
        print(f"   💾 Saved: {output_path}")

    print("\n✨ Instruction-controlled synthesis complete!")


def main():
    """Run the voice cloning example by default."""
    voice_cloning_example()

    # Uncomment to run additional examples:
    # cross_lingual_example()
    # instruct_example()


if __name__ == "__main__":
    main()
