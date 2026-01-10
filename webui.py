# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
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
Fun-CosyVoice3 Web UI

A Gradio-based web interface for Fun-CosyVoice3-0.5B-2512.
Supports zero-shot voice cloning, cross-lingual synthesis, and instruction-controlled TTS.
"""

import argparse
import os
import random
import sys

import gradio as gr
import numpy as np
import torchaudio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import logging

# =============================================================================
# Inference Modes for Fun-CosyVoice3
# =============================================================================

inference_mode_list = ["3s Zero-Shot", "Cross-Lingual", "Instruct"]
instruct_dict = {
    "3s Zero-Shot": "1. Select prompt audio file or record it. Note: Max 30s. If both provided, file takes precedence.\n2. Enter prompt text.\n3. Click Generate Audio button.",
    "Cross-Lingual": "1. Select prompt audio file or record it. Note: Max 30s. If both provided, file takes precedence.\n2. Click Generate Audio button.",
    "Instruct": "1. Select prompt audio file.\n2. Enter instruct text (control language, emotion, speed, etc.).\n3. Click Generate Audio button.",
}
stream_mode_list = [("No", False), ("Yes", True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {"__type__": "update", "value": seed}


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(
    tts_text,
    mode_checkbox_group,
    prompt_text,
    prompt_wav_upload,
    prompt_wav_record,
    instruct_text,
    seed,
    stream,
    speed,
):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None


    # Validation for instruction mode
    if mode_checkbox_group == "Instruct":
        if instruct_text == "":
            gr.Warning("You are using Instruct Mode, please enter instruct text")
            yield (cosyvoice.sample_rate, default_data)
            return
        if prompt_wav is None:
            gr.Warning("Please provide prompt audio")
            yield (cosyvoice.sample_rate, default_data)
            return

    # Validation for cross-lingual mode
    if mode_checkbox_group == "Cross-Lingual":
        if instruct_text != "":
            gr.Info("You are using Cross-Lingual Mode, instruct text will be ignored")
        if prompt_wav is None:
            gr.Warning("You are using Cross-Lingual Mode, please provide prompt audio")
            yield (cosyvoice.sample_rate, default_data)
            return
        gr.Info(
            "You are using Cross-Lingual Mode, please ensure synth text and prompt text are different languages"
        )

    # Validation for zero-shot mode
    if mode_checkbox_group == "3s Zero-Shot":
        if prompt_wav is None:
            gr.Warning("Prompt audio is empty, did you forget to upload it?")
            yield (cosyvoice.sample_rate, default_data)
            return
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning(
                "Prompt audio sample rate {} is lower than {}".format(
                    torchaudio.info(prompt_wav).sample_rate, prompt_sr
                )
            )
            yield (cosyvoice.sample_rate, default_data)
            return
        if prompt_text == "":
            gr.Warning("Prompt text is empty, did you forget to enter it?")
            yield (cosyvoice.sample_rate, default_data)
            return
        if instruct_text != "":
            gr.Info("You are using 3s Zero-Shot Mode, instruct text will be ignored!")

    # Run inference based on mode
    if mode_checkbox_group == "3s Zero-Shot":
        logging.info("get zero_shot inference request")
        set_all_random_seed(seed)
        # Add prompt prefix for better quality
        full_prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text
        for i in cosyvoice.inference_zero_shot(
            tts_text, full_prompt_text, prompt_wav, stream=stream, speed=speed
        ):
            yield (cosyvoice.sample_rate, i["tts_speech"].numpy().flatten())
    elif mode_checkbox_group == "Cross-Lingual":
        logging.info("get cross_lingual inference request")
        set_all_random_seed(seed)
        # Add prompt prefix for better quality
        full_tts_text = "You are a helpful assistant.<|endofprompt|>" + tts_text
        for i in cosyvoice.inference_cross_lingual(
            full_tts_text, prompt_wav, stream=stream, speed=speed
        ):
            yield (cosyvoice.sample_rate, i["tts_speech"].numpy().flatten())
    else:  # Instruct Mode
        logging.info("get instruct inference request")
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct2(
            tts_text, instruct_text, prompt_wav, stream=stream, speed=speed
        ):
            yield (cosyvoice.sample_rate, i["tts_speech"].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("""
        ### Fun-CosyVoice3 Speech Synthesis

        🎤 **Fun-CosyVoice3-0.5B-2512** - Latest Generation Speech Synthesis Model

        [GitHub](https://github.com/FunAudioLLM/CosyVoice) |
        [ModelScope](https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) |
        [HuggingFace](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) |
        [Paper](https://arxiv.org/pdf/2505.17589)
        """)
        gr.Markdown(
            "#### Please enter text to synthesize, choose inference mode, and follow the steps below"
        )

        tts_text = gr.Textbox(
            label="Input Text",
            lines=1,
            value="I am a new generative speech model launched by the Tongyi Lab Speech Team, providing comfortable and natural speech synthesis capabilities.",
        )
        with gr.Row():
            mode_checkbox_group = gr.Radio(
                choices=inference_mode_list,
                label="Inference Mode",
                value=inference_mode_list[0],
            )
            instruction_text = gr.Text(
                label="Instructions",
                value=instruct_dict[inference_mode_list[0]],
                scale=0.5,
            )
            stream = gr.Radio(
                choices=stream_mode_list,
                label="Stream Inference",
                value=stream_mode_list[0][1],
            )
            speed = gr.Number(
                value=1,
                label="Speed Adjustment (Non-streaming only)",
                minimum=0.5,
                maximum=2.0,
                step=0.1,
            )
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001f3b2")
                seed = gr.Number(value=0, label="Random Seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(
                sources="upload",
                type="filepath",
                label="Select prompt audio file (Sample rate >= 16kHz)",
            )
            prompt_wav_record = gr.Audio(
                sources="microphone", type="filepath", label="Record prompt audio"
            )
        prompt_text = gr.Textbox(
            label="Input Prompt Text",
            lines=1,
            placeholder="Please enter prompt text (must match audio content). Automatic recognition not supported yet...",
            value="",
        )
        instruct_text = gr.Textbox(
            label="Input Instruct Text",
            lines=1,
            placeholder="Example: Please speak this in Cantonese <|endofprompt|>",
            value="",
        )

        generate_button = gr.Button("Generate Audio")

        audio_output = gr.Audio(label="Generated Audio", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(
            generate_audio,
            inputs=[
                tts_text,
                mode_checkbox_group,
                prompt_text,
                prompt_wav_upload,
                prompt_wav_record,
                instruct_text,
                seed,
                stream,
                speed,
            ],
            outputs=[audio_output],
        )
        mode_checkbox_group.change(
            fn=change_instruction,
            inputs=[mode_checkbox_group],
            outputs=[instruction_text],
        )
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Fun-CosyVoice3-0.5B",
        help="local path or modelscope repo id",
    )
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = [""]
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
