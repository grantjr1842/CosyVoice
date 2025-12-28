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
CosyVoice3 - Text-to-Speech with Fun-CosyVoice3-0.5B-2512

This module provides the CosyVoice3 class and AutoModel function for TTS inference.
Only Fun-CosyVoice3-0.5B-2512 is supported. Legacy models (CosyVoice v1/v2) have been removed.
"""

import os
import time
from typing import Generator

import torch
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from tqdm import tqdm

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoice3Model
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.gpu_optimizer import GpuOptimizer


class CosyVoice3:
    """
    CosyVoice3 - Main interface for Fun-CosyVoice3-0.5B-2512 TTS.

    Supports:
    - Zero-shot voice cloning (inference_zero_shot)
    - Cross-lingual synthesis (inference_cross_lingual)
    - Instruction-controlled synthesis (inference_instruct2)

    Example:
        >>> from cosyvoice.cli.cosyvoice import CosyVoice3, AutoModel
        >>> cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
        >>> for output in cosyvoice.inference_zero_shot(
        ...     'Hello world!',
        ...     'You are a helpful assistant.<|endofprompt|>Reference text.',
        ...     './asset/reference.wav'
        ... ):
        ...     torchaudio.save('output.wav', output['tts_speech'], cosyvoice.sample_rate)
    """

    def __init__(
        self, model_dir, load_trt=False, load_vllm=False, fp16=None, trt_concurrent=1
    ):
        """
        Initialize CosyVoice3.

        Args:
            model_dir: Path to model directory or ModelScope model ID
            load_trt: Load TensorRT engine for accelerated inference
            load_vllm: Load vLLM for accelerated LLM inference
            fp16: Use FP16 precision. If None, it will be auto-detected based on GPU capabilities.
            trt_concurrent: Number of concurrent TRT contexts
        """
        if fp16 is None:
            optimizer = GpuOptimizer()

            # MatMul Precision for Ampere+
            mm_precision = optimizer.suggest_matmul_precision()
            if mm_precision != "default":
                torch.set_float32_matmul_precision(mm_precision)
                logging.info(
                    f"Setting torch.set_float32_matmul_precision('{mm_precision}') for performance."
                )

            params = optimizer.suggest_parameters()
            fp16 = params.get("fp16", False)
            logging.info(f"Auto-configured parameters: fp16={fp16}")

        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = "{}/cosyvoice3.yaml".format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError(
                "{} not found! Only Fun-CosyVoice3-0.5B-2512 is supported.".format(
                    hyper_yaml_path
                )
            )
        with open(hyper_yaml_path, "r") as f:
            configs = load_hyperpyyaml(
                f,
                overrides={
                    "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")
                },
            )
        self.frontend = CosyVoiceFrontEnd(
            configs["get_tokenizer"],
            configs["feat_extractor"],
            "{}/campplus.onnx".format(model_dir),
            "{}/speech_tokenizer_v3.onnx".format(model_dir),
            "{}/spk2info.pt".format(model_dir),
            configs["allowed_special"],
        )
        self.sample_rate = configs["sample_rate"]
        if torch.cuda.is_available() is False and (load_trt is True or fp16 is True):
            load_trt, fp16 = False, False
            logging.warning("no cuda device, set load_trt/fp16 to False")
        self.model = CosyVoice3Model(
            configs["llm"], configs["flow"], configs["hift"], fp16
        )
        self.model.load(
            "{}/llm.pt".format(model_dir),
            "{}/flow.pt".format(model_dir),
            "{}/hift.pt".format(model_dir),
        )
        if load_vllm:
            self.model.load_vllm("{}/vllm".format(model_dir))
        if load_trt:
            if self.fp16 is True:
                logging.warning(
                    "DiT tensorRT fp16 engine have some performance issue, use at caution!"
                )
            self.model.load_trt(
                "{}/flow.decoder.estimator.{}.mygpu.plan".format(
                    model_dir, "fp16" if self.fp16 is True else "fp32"
                ),
                "{}/flow.decoder.estimator.fp32.onnx".format(model_dir),
                trt_concurrent,
                self.fp16,
            )
        del configs

    def list_available_spks(self):
        """List available speaker IDs."""
        spks = list(self.frontend.spk2info.keys())
        return spks

    def add_zero_shot_spk(self, prompt_text, prompt_wav, zero_shot_spk_id):
        """
        Add a zero-shot speaker for reuse.

        Args:
            prompt_text: Transcription of the prompt audio
            prompt_wav: Path to prompt audio file
            zero_shot_spk_id: Unique ID to assign to this speaker

        Returns:
            bool: True if successful
        """
        assert zero_shot_spk_id != "", "do not use empty zero_shot_spk_id"
        model_input = self.frontend.frontend_zero_shot(
            "", prompt_text, prompt_wav, self.sample_rate, ""
        )
        del model_input["text"]
        del model_input["text_len"]
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True

    def save_spkinfo(self):
        """Save speaker info to disk."""
        torch.save(self.frontend.spk2info, "{}/spk2info.pt".format(self.model_dir))

    def inference_zero_shot(
        self,
        tts_text,
        prompt_text,
        prompt_wav,
        zero_shot_spk_id="",
        stream=False,
        speed=1.0,
        text_frontend=True,
    ):
        """
        Zero-shot voice cloning inference.

        Args:
            tts_text: Text to synthesize
            prompt_text: Transcription of prompt audio (include instruction prefix for better results)
            prompt_wav: Path to prompt audio file (reference voice)
            zero_shot_spk_id: Optional saved speaker ID to use instead of prompt_wav
            stream: Enable streaming output
            speed: Playback speed adjustment
            text_frontend: Apply text normalization

        Yields:
            dict: Contains 'tts_speech' tensor
        """
        prompt_text = self.frontend.text_normalize(
            prompt_text, split=False, text_frontend=text_frontend
        )
        for i in tqdm(
            self.frontend.text_normalize(
                tts_text, split=True, text_frontend=text_frontend
            )
        ):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning(
                    "synthesis text {} too short than prompt text {}, this may lead to bad performance".format(
                        i, prompt_text
                    )
                )
            model_input = self.frontend.frontend_zero_shot(
                i, prompt_text, prompt_wav, self.sample_rate, zero_shot_spk_id
            )
            start_time = time.time()
            logging.info("synthesis text {}".format(i))
            for model_output in self.model.tts(
                **model_input, stream=stream, speed=speed
            ):
                speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
                inference_time = time.time() - start_time
                if speech_len > 0:
                    rtf = inference_time / speech_len
                    logging.info(
                        "yield speech len {:.2f}s, inference time {:.3f}s, rtf {:.3f}".format(
                            speech_len, inference_time, rtf
                        )
                    )
                else:
                    logging.warning("yield speech len is 0, skipping rtf calculation")
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(
        self,
        tts_text,
        prompt_wav,
        zero_shot_spk_id="",
        stream=False,
        speed=1.0,
        text_frontend=True,
    ):
        """
        Cross-lingual synthesis inference.

        Args:
            tts_text: Text to synthesize (can be different language from prompt)
            prompt_wav: Path to prompt audio file
            zero_shot_spk_id: Optional saved speaker ID
            stream: Enable streaming output
            speed: Playback speed adjustment
            text_frontend: Apply text normalization

        Yields:
            dict: Contains 'tts_speech' tensor
        """
        for i in tqdm(
            self.frontend.text_normalize(
                tts_text, split=True, text_frontend=text_frontend
            )
        ):
            model_input = self.frontend.frontend_cross_lingual(
                i, prompt_wav, self.sample_rate, zero_shot_spk_id
            )
            start_time = time.time()
            logging.info("synthesis text {}".format(i))
            for model_output in self.model.tts(
                **model_input, stream=stream, speed=speed
            ):
                speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
                logging.info(
                    "yield speech len {}, rtf {}".format(
                        speech_len, (time.time() - start_time) / speech_len
                    )
                )
                yield model_output
                start_time = time.time()

    def inference_instruct2(
        self,
        tts_text,
        instruct_text,
        prompt_wav,
        zero_shot_spk_id="",
        stream=False,
        speed=1.0,
        text_frontend=True,
    ):
        """
        Instruction-controlled synthesis inference.

        Args:
            tts_text: Text to synthesize
            instruct_text: Instruction for voice style (e.g., language, emotion, speed)
            prompt_wav: Path to prompt audio file
            zero_shot_spk_id: Optional saved speaker ID
            stream: Enable streaming output
            speed: Playback speed adjustment
            text_frontend: Apply text normalization

        Yields:
            dict: Contains 'tts_speech' tensor
        """
        for i in tqdm(
            self.frontend.text_normalize(
                tts_text, split=True, text_frontend=text_frontend
            )
        ):
            model_input = self.frontend.frontend_instruct2(
                i, instruct_text, prompt_wav, self.sample_rate, zero_shot_spk_id
            )
            start_time = time.time()
            logging.info("synthesis text {}".format(i))
            for model_output in self.model.tts(
                **model_input, stream=stream, speed=speed
            ):
                speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
                logging.info(
                    "yield speech len {}, rtf {}".format(
                        speech_len, (time.time() - start_time) / speech_len
                    )
                )
                yield model_output
                start_time = time.time()


def AutoModel(**kwargs):
    """
    Automatically create a CosyVoice3 model.

    This function downloads the model if necessary and returns a CosyVoice3 instance.
    Only Fun-CosyVoice3-0.5B-2512 is supported.

    Args:
        model_dir: Path to model directory or ModelScope model ID
        load_trt: Load TensorRT engine
        load_vllm: Load vLLM
        fp16: Use FP16 precision
        trt_concurrent: Number of concurrent TRT contexts

    Returns:
        CosyVoice3: Initialized model instance

    Raises:
        TypeError: If model type is not Fun-CosyVoice3

    Example:
        >>> cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
    """
    if not os.path.exists(kwargs["model_dir"]):
        kwargs["model_dir"] = snapshot_download(kwargs["model_dir"])

    # Auto-enable TensorRT if ONNX model exists
    onnx_path = os.path.join(kwargs["model_dir"], "flow.decoder.estimator.fp32.onnx")
    if "load_trt" not in kwargs and os.path.exists(onnx_path):
        try:
            import tensorrt

            logging.info(
                "Detected ONNX model at {}. Enabling TensorRT loading.".format(
                    onnx_path
                )
            )
            kwargs["load_trt"] = True
        except ImportError:
            logging.warning(
                "Detected ONNX model but tensorrt module not found. Skipping TensorRT."
            )
            kwargs["load_trt"] = False

    if os.path.exists("{}/cosyvoice3.yaml".format(kwargs["model_dir"])):
        return CosyVoice3(**kwargs)
    else:
        raise TypeError(
            "Only Fun-CosyVoice3-0.5B-2512 is supported. "
            "Please download the model with: "
            "from modelscope import snapshot_download; "
            "snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')"
        )
