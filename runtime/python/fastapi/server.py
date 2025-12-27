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
Fun-CosyVoice3 FastAPI Server

Provides TTS inference via REST API for Fun-CosyVoice3-0.5B-2512.
"""

import argparse
import logging
import os
import sys

logging.getLogger("matplotlib").setLevel(logging.WARNING)
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../../..".format(ROOT_DIR))
sys.path.append("{}/../../../third_party/Matcha-TTS".format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav

app = FastAPI(
    title="Fun-CosyVoice3 TTS API",
    description="Text-to-Speech API powered by Fun-CosyVoice3-0.5B-2512",
    version="3.0.0",
)

# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()
):
    """
    Zero-shot voice cloning inference.

    Args:
        tts_text: Text to synthesize
        prompt_text: Transcription of the prompt audio
        prompt_wav: Reference audio file for voice cloning
    """
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    # Add prompt prefix for CosyVoice3
    full_prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text
    model_output = cosyvoice.inference_zero_shot(
        tts_text, full_prompt_text, prompt_speech_16k
    )
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(
    tts_text: str = Form(), prompt_wav: UploadFile = File()
):
    """
    Cross-lingual synthesis inference.

    Args:
        tts_text: Text to synthesize (can be different language from prompt)
        prompt_wav: Reference audio file
    """
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    # Add prompt prefix for CosyVoice3
    full_tts_text = "You are a helpful assistant.<|endofprompt|>" + tts_text
    model_output = cosyvoice.inference_cross_lingual(full_tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()
):
    """
    Instruction-controlled synthesis inference.

    Args:
        tts_text: Text to synthesize
        instruct_text: Natural language instruction for voice style
        prompt_wav: Reference audio file
    """
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(
        tts_text, instruct_text, prompt_speech_16k
    )
    return StreamingResponse(generate_data(model_output))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": "Fun-CosyVoice3-0.5B-2512"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Fun-CosyVoice3-0.5B",
        help="local path or modelscope repo id",
    )
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
