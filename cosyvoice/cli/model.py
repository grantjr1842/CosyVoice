# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
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
CosyVoice3Model - The only supported model class for Fun-CosyVoice3-0.5B-2512.

This module provides the CosyVoice3Model class for text-to-speech inference.
Legacy CosyVoice v1 and v2 models are no longer supported.
"""

import os
import threading
import time
import uuid
from contextlib import nullcontext
from typing import Generator

import numpy as np
import torch
from torch.nn import functional as F

from cosyvoice.utils.common import TrtContextWrapper
from cosyvoice.utils.file_utils import (
    convert_onnx_to_trt,
    export_cosyvoice2_vllm,
    logging,
)


class CosyVoice3Model:
    """
    CosyVoice3 Model for Fun-CosyVoice3-0.5B-2512.

    This is the only supported model class. Supports:
    - Zero-shot voice cloning
    - Cross-lingual synthesis
    - Instruction-controlled synthesis
    - Streaming and non-streaming inference
    """

    def __init__(
        self,
        llm: torch.nn.Module,
        flow: torch.nn.Module,
        hift: torch.nn.Module,
        fp16: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        # NOTE must matching training static_chunk_size
        self.token_hop_len = 25
        # rtf and decoding related
        self.llm_context = (
            torch.cuda.stream(torch.cuda.Stream(self.device))
            if torch.cuda.is_available()
            else nullcontext()
        )
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

    def load(self, llm_model, flow_model, hift_model):
        """Load model weights from disk."""
        self.llm.load_state_dict(
            torch.load(llm_model, map_location=self.device, weights_only=True),
            strict=True,
        )
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(
            torch.load(flow_model, map_location=self.device, weights_only=True),
            strict=True,
        )
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {
            k.replace("generator.", ""): v
            for k, v in torch.load(
                hift_model, map_location=self.device, weights_only=True
            ).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

        # Apply torch.compile for performance
        if hasattr(torch, "compile"):
            from cosyvoice.utils.gpu_optimizer import GpuOptimizer

            optimizer = GpuOptimizer()
            compile_mode = optimizer.suggest_compile_mode()

            if compile_mode:
                logging.info(
                    f"Applying torch.compile to models with mode='{compile_mode}'..."
                )
                self.llm = torch.compile(self.llm, mode=compile_mode)
                self.flow = torch.compile(self.flow, mode=compile_mode)
                self.hift = torch.compile(self.hift, mode=compile_mode)
            else:
                logging.info("Skipping torch.compile (no compatible GPU detected).")

    def load_vllm(self, model_dir):
        """Load vLLM for accelerated inference."""
        export_cosyvoice2_vllm(self.llm, model_dir, self.device)
        from vllm import EngineArgs, LLMEngine

        engine_args = EngineArgs(
            model=model_dir,
            skip_tokenizer_init=True,
            enable_prompt_embeds=True,
            gpu_memory_utilization=0.2,
        )
        self.llm.vllm = LLMEngine.from_engine_args(engine_args)
        self.llm.lock = threading.Lock()
        del self.llm.llm.model.model.layers

    def load_trt(
        self,
        flow_decoder_estimator_model,
        flow_decoder_onnx_model,
        trt_concurrent,
        fp16,
    ):
        """Load TensorRT engine for accelerated flow decoder."""
        assert torch.cuda.is_available(), "tensorrt only supports gpu!"
        if (
            not os.path.exists(flow_decoder_estimator_model)
            or os.path.getsize(flow_decoder_estimator_model) == 0
        ):
            convert_onnx_to_trt(
                flow_decoder_estimator_model,
                self.get_trt_kwargs(),
                flow_decoder_onnx_model,
                fp16,
            )
        del self.flow.decoder.estimator
        import tensorrt as trt

        with open(flow_decoder_estimator_model, "rb") as f:
            estimator_engine = trt.Runtime(
                trt.Logger(trt.Logger.INFO)
            ).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, "failed to load trt {}".format(
            flow_decoder_estimator_model
        )
        self.flow.decoder.estimator = TrtContextWrapper(
            estimator_engine, trt_concurrent=trt_concurrent, device=self.device
        )

    def get_trt_kwargs(self):
        """Get TensorRT optimization parameters."""
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {
            "min_shape": min_shape,
            "opt_shape": opt_shape,
            "max_shape": max_shape,
            "input_names": input_names,
        }

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        """LLM inference job for generating speech tokens."""
        prompt_text = prompt_text.to(self.device)
        prompt_text_len = torch.tensor(
            [prompt_text.shape[1]], dtype=torch.int32, device=self.device
        )
        prompt_speech_token = llm_prompt_speech_token.to(self.device)
        prompt_speech_token_len = torch.tensor(
            [prompt_speech_token.shape[1]], dtype=torch.int32, device=self.device
        )
        embedding = llm_embedding.to(self.device)
        autocast_enabled = (
            self.fp16 is True
            and self.device.type == "cuda"
            and hasattr(self.llm, "vllm") is False
        )
        with (
            torch.inference_mode(),
            self.llm_context,
            torch.amp.autocast("cuda", enabled=autocast_enabled),
        ):
            if isinstance(text, Generator):
                assert not hasattr(self.llm, "vllm"), (
                    "streaming input text does not support vllm!"
                )
                for i in self.llm.inference_bistream(
                    text=text,
                    prompt_text=prompt_text,
                    prompt_text_len=prompt_text_len,
                    prompt_speech_token=prompt_speech_token,
                    prompt_speech_token_len=prompt_speech_token_len,
                    embedding=embedding,
                ):
                    self.tts_speech_token_dict[uuid].append(i)
            else:
                text = text.to(self.device)
                text_len = torch.tensor(
                    [text.shape[1]], dtype=torch.int32, device=self.device
                )
                for i in self.llm.inference(
                    text=text,
                    text_len=text_len,
                    prompt_text=prompt_text,
                    prompt_text_len=prompt_text_len,
                    prompt_speech_token=prompt_speech_token,
                    prompt_speech_token_len=prompt_speech_token_len,
                    embedding=embedding,
                    uuid=uuid,
                ):
                    self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def vc_job(self, source_speech_token, uuid):
        """Voice conversion job."""
        self.tts_speech_token_dict[uuid] = source_speech_token.flatten().tolist()
        self.llm_end_dict[uuid] = True

    def token2wav(
        self,
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
        """Convert speech tokens to waveform."""
        if token.device != self.device or token.dtype != torch.int32:
            token = token.to(self.device, dtype=torch.int32)
        if prompt_token.device != self.device or prompt_token.dtype != torch.int32:
            prompt_token = prompt_token.to(self.device, dtype=torch.int32)
        if prompt_feat.device != self.device:
            prompt_feat = prompt_feat.to(self.device)
        if embedding.device != self.device:
            embedding = embedding.to(self.device)
        token_len = torch.tensor([token.shape[1]], dtype=torch.int32, device=self.device)
        prompt_token_len = torch.tensor(
            [prompt_token.shape[1]], dtype=torch.int32, device=self.device
        )
        prompt_feat_len = torch.tensor(
            [prompt_feat.shape[1]], dtype=torch.int32, device=self.device
        )
        autocast_enabled = self.fp16 is True and self.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=autocast_enabled):
            # timer for flow
            tts_mel, _ = self.flow.inference(
                token=token,
                token_len=token_len,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
                streaming=stream,
                finalize=finalize,
            )

            tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio :]
            # append mel cache
            if self.hift_cache_dict[uuid] is not None:
                hift_cache_mel = self.hift_cache_dict[uuid]["mel"]
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
            else:
                self.hift_cache_dict[uuid] = {"mel": tts_mel, "speech_offset": 0}
            if speed != 1.0:
                assert token_offset == 0 and finalize is True, (
                    "speed change only support non-stream inference mode"
                )
                tts_mel = F.interpolate(
                    tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear"
                )

            # timer for hift
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)

            tts_speech = tts_speech[:, self.hift_cache_dict[uuid]["speech_offset"] :]
            self.hift_cache_dict[uuid]["speech_offset"] += tts_speech.shape[1]
        return tts_speech

    @torch.inference_mode()
    def tts(
        self,
        text=torch.zeros(1, 0, dtype=torch.int32),
        flow_embedding=torch.zeros(0, 192),
        llm_embedding=torch.zeros(0, 192),
        prompt_text=torch.zeros(1, 0, dtype=torch.int32),
        llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat=torch.zeros(1, 0, 80),
        source_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        stream=False,
        speed=1.0,
        **kwargs,
    ):
        """
        Main TTS inference method.

        Args:
            text: Input text tokens
            flow_embedding: Flow model embedding
            llm_embedding: LLM embedding
            prompt_text: Prompt text tokens for zero-shot
            llm_prompt_speech_token: LLM prompt speech tokens
            flow_prompt_speech_token: Flow prompt speech tokens
            prompt_speech_feat: Prompt speech features
            source_speech_token: Source speech tokens for voice conversion
            stream: Enable streaming inference
            speed: Speed adjustment (non-streaming only)

        Yields:
            dict: Contains 'tts_speech' tensor with generated audio
        """
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = (
                [],
                False,
            )
            self.hift_cache_dict[this_uuid] = None
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(
                target=self.llm_job,
                args=(
                    text,
                    prompt_text,
                    llm_prompt_speech_token,
                    llm_embedding,
                    this_uuid,
                ),
            )
        else:
            p = threading.Thread(
                target=self.vc_job, args=(source_speech_token, this_uuid)
            )
        p.start()
        if stream is True:
            token_offset = 0
            prompt_token_pad = int(
                np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len)
                * self.token_hop_len
                - flow_prompt_speech_token.shape[1]
            )
            while True:
                # Reduced sleep time for lower latency
                time.sleep(0.01)
                this_token_hop_len = (
                    self.token_hop_len + prompt_token_pad
                    if token_offset == 0
                    else self.token_hop_len
                )
                if (
                    len(self.tts_speech_token_dict[this_uuid]) - token_offset
                    >= this_token_hop_len + self.flow.pre_lookahead_len
                ):
                    this_tts_speech_token = torch.tensor(
                        self.tts_speech_token_dict[this_uuid][
                            : token_offset
                            + this_token_hop_len
                            + self.flow.pre_lookahead_len
                        ]
                    ).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(
                        token=this_tts_speech_token,
                        prompt_token=flow_prompt_speech_token,
                        prompt_feat=prompt_speech_feat,
                        embedding=flow_embedding,
                        token_offset=token_offset,
                        uuid=this_uuid,
                        stream=stream,
                        finalize=False,
                    )
                    token_offset += this_token_hop_len
                    yield {"tts_speech": this_tts_speech.cpu()}
                if (
                    self.llm_end_dict[this_uuid] is True
                    and len(self.tts_speech_token_dict[this_uuid]) - token_offset
                    < this_token_hop_len + self.flow.pre_lookahead_len
                ):
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(
                self.tts_speech_token_dict[this_uuid],
                dtype=torch.int32,
            ).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                token_offset=token_offset,
                uuid=this_uuid,
                finalize=True,
            )
            yield {"tts_speech": this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(
                self.tts_speech_token_dict[this_uuid],
                dtype=torch.int32,
            ).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                token_offset=0,
                uuid=this_uuid,
                finalize=True,
                speed=speed,
            )
            yield {"tts_speech": this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
        if (
            torch.cuda.is_available()
            and os.environ.get("COSYVOICE_CLEAR_CACHE", "0") == "1"
        ):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
