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


def cosyvoice3_example():
    """CosyVoice3 Usage, check https://funaudiollm.github.io/cosyvoice3/ for more details"""
    cosyvoice = AutoModel(model_dir="pretrained_models/Fun-CosyVoice3-0.5B")
    # zero_shot usage
    for i, j in enumerate(
        cosyvoice.inference_zero_shot(
            "Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
            "You are a helpful assistant.<|endofprompt|>I hope you can do better than me in the future.",
            "./asset/zero_shot_prompt.wav",
            stream=False,
        )
    ):
        torchaudio.save(
            "zero_shot_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
        )

    # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L280
    for i, j in enumerate(
        cosyvoice.inference_cross_lingual(
            "You are a helpful assistant.<|endofprompt|>[breath]Because that generation of people[breath]are used to living in the countryside,[breath]neighbors are very active,[breath]um, very familiar.[breath]",
            "./asset/zero_shot_prompt.wav",
            stream=False,
        )
    ):
        torchaudio.save(
            "fine_grained_control_{}.wav".format(i),
            j["tts_speech"],
            cosyvoice.sample_rate,
        )

    # instruct usage, for supported control, check cosyvoice/utils/common.py#L28
    for i, j in enumerate(
        cosyvoice.inference_instruct2(
            "It's rare, usually only during National Day or Mid-Autumn Festival.",
            "You are a helpful assistant. Please use a sad tone.<|endofprompt|>",
            "./asset/zero_shot_prompt.wav",
            stream=False,
        )
    ):
        torchaudio.save(
            "instruct_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
        )
    for i, j in enumerate(
        cosyvoice.inference_instruct2(
            "Received a birthday gift from a friend from afar, that unexpected surprise and deep blessing filled my heart with sweet happiness, smiling like a flower blooming.",
            "You are a helpful assistant. Please speak as fast as possible.<|endofprompt|>",
            "./asset/zero_shot_prompt.wav",
            stream=False,
        )
    ):
        torchaudio.save(
            "instruct_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
        )

    # hotfix usage
    for i, j in enumerate(
        cosyvoice.inference_zero_shot(
            "Executives also praised the report via phone, SMS, WeChat, etc.",
            "You are a helpful assistant.<|endofprompt|>I hope you can do better than me in the future.",
            "./asset/zero_shot_prompt.wav",
            stream=False,
        )
    ):
        torchaudio.save(
            "hotfix_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
        )


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


def cosyvoice3_example():
    """CosyVoice3 Usage, check https://funaudiollm.github.io/cosyvoice3/ for more details"""
    cosyvoice = AutoModel(model_dir="pretrained_models/Fun-CosyVoice3-0.5B")
    # zero_shot usage
    for i, j in enumerate(
        cosyvoice.inference_zero_shot(
            "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
            "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
            "./asset/zero_shot_prompt.wav",
            stream=False,
        )
    ):
        torchaudio.save(
            "zero_shot_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
        )

    # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L280
    for i, j in enumerate(
        cosyvoice.inference_cross_lingual(
            "You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]",
            "./asset/zero_shot_prompt.wav",
            stream=False,
        )
    ):
        torchaudio.save(
            "fine_grained_control_{}.wav".format(i),
            j["tts_speech"],
            cosyvoice.sample_rate,
        )

    # instruct usage, for supported control, check cosyvoice/utils/common.py#L28
    for i, j in enumerate(
        cosyvoice.inference_instruct2(
            "好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。",
            "You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
            "./asset/zero_shot_prompt.wav",
            stream=False,
        )
    ):
        torchaudio.save(
            "instruct_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
        )
    for i, j in enumerate(
        cosyvoice.inference_instruct2(
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
            "You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>",
            "./asset/zero_shot_prompt.wav",
            stream=False,
        )
    ):
        torchaudio.save(
            "instruct_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
        )

    # hotfix usage
    for i, j in enumerate(
        cosyvoice.inference_zero_shot(
            "高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。",
            "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
            "./asset/zero_shot_prompt.wav",
            stream=False,
        )
    ):
        torchaudio.save(
            "hotfix_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
        )


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
    # voice_cloning_example()
    cosyvoice3_example()

    # Uncomment to run additional examples:
    # instruct_example()


if __name__ == "__main__":
    main()
