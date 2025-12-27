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

inference_mode_list = ["3s极速复刻", "跨语种复刻", "自然语言控制"]
instruct_dict = {
    "3s极速复刻": "1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮",
    "跨语种复刻": "1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮",
    "自然语言控制": "1. 选择prompt音频文件\n2. 输入instruct文本（控制语言、情感、语速等）\n3. 点击生成音频按钮",
}
stream_mode_list = [("否", False), ("是", True)]
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
    if mode_checkbox_group == "自然语言控制":
        if instruct_text == "":
            gr.Warning("您正在使用自然语言控制模式, 请输入instruct文本")
            yield (cosyvoice.sample_rate, default_data)
            return
        if prompt_wav is None:
            gr.Warning("请提供prompt音频")
            yield (cosyvoice.sample_rate, default_data)
            return

    # Validation for cross-lingual mode
    if mode_checkbox_group == "跨语种复刻":
        if instruct_text != "":
            gr.Info("您正在使用跨语种复刻模式, instruct文本会被忽略")
        if prompt_wav is None:
            gr.Warning("您正在使用跨语种复刻模式, 请提供prompt音频")
            yield (cosyvoice.sample_rate, default_data)
            return
        gr.Info("您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言")

    # Validation for zero-shot mode
    if mode_checkbox_group == "3s极速复刻":
        if prompt_wav is None:
            gr.Warning("prompt音频为空，您是否忘记输入prompt音频？")
            yield (cosyvoice.sample_rate, default_data)
            return
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning(
                "prompt音频采样率{}低于{}".format(
                    torchaudio.info(prompt_wav).sample_rate, prompt_sr
                )
            )
            yield (cosyvoice.sample_rate, default_data)
            return
        if prompt_text == "":
            gr.Warning("prompt文本为空，您是否忘记输入prompt文本？")
            yield (cosyvoice.sample_rate, default_data)
            return
        if instruct_text != "":
            gr.Info("您正在使用3s极速复刻模式，instruct文本会被忽略！")

    # Run inference based on mode
    if mode_checkbox_group == "3s极速复刻":
        logging.info("get zero_shot inference request")
        set_all_random_seed(seed)
        # Add prompt prefix for better quality
        full_prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text
        for i in cosyvoice.inference_zero_shot(
            tts_text, full_prompt_text, prompt_wav, stream=stream, speed=speed
        ):
            yield (cosyvoice.sample_rate, i["tts_speech"].numpy().flatten())
    elif mode_checkbox_group == "跨语种复刻":
        logging.info("get cross_lingual inference request")
        set_all_random_seed(seed)
        # Add prompt prefix for better quality
        full_tts_text = "You are a helpful assistant.<|endofprompt|>" + tts_text
        for i in cosyvoice.inference_cross_lingual(
            full_tts_text, prompt_wav, stream=stream, speed=speed
        ):
            yield (cosyvoice.sample_rate, i["tts_speech"].numpy().flatten())
    else:  # 自然语言控制
        logging.info("get instruct inference request")
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct2(
            tts_text, instruct_text, prompt_wav, stream=stream, speed=speed
        ):
            yield (cosyvoice.sample_rate, i["tts_speech"].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("""
        ### Fun-CosyVoice3 语音合成

        🎤 **Fun-CosyVoice3-0.5B-2512** - 最新一代语音合成模型

        [GitHub](https://github.com/FunAudioLLM/CosyVoice) |
        [ModelScope](https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) |
        [HuggingFace](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) |
        [论文](https://arxiv.org/pdf/2505.17589)
        """)
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        tts_text = gr.Textbox(
            label="输入合成文本",
            lines=1,
            value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。",
        )
        with gr.Row():
            mode_checkbox_group = gr.Radio(
                choices=inference_mode_list,
                label="选择推理模式",
                value=inference_mode_list[0],
            )
            instruction_text = gr.Text(
                label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5
            )
            stream = gr.Radio(
                choices=stream_mode_list,
                label="是否流式推理",
                value=stream_mode_list[0][1],
            )
            speed = gr.Number(
                value=1,
                label="速度调节(仅支持非流式推理)",
                minimum=0.5,
                maximum=2.0,
                step=0.1,
            )
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001f3b2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(
                sources="upload",
                type="filepath",
                label="选择prompt音频文件，注意采样率不低于16khz",
            )
            prompt_wav_record = gr.Audio(
                sources="microphone", type="filepath", label="录制prompt音频文件"
            )
        prompt_text = gr.Textbox(
            label="输入prompt文本",
            lines=1,
            placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...",
            value="",
        )
        instruct_text = gr.Textbox(
            label="输入instruct文本（自然语言控制模式）",
            lines=1,
            placeholder="例如：请用广东话说这句话<|endofprompt|>",
            value="",
        )

        generate_button = gr.Button("生成音频")

        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)

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
