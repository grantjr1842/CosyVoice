![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=Fun-CosyVoice3🤠&text2=Zero-Shot%20Voice%20Cloning%20💖%20Large%20Language%20Model&width=800&height=210)

## 👉🏻 Fun-CosyVoice3 👈🏻

**Fun-CosyVoice 3.0**: [Demos](https://funaudiollm.github.io/cosyvoice3/) | [Paper](https://arxiv.org/pdf/2505.17589) | [ModelScope](https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | [HuggingFace](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | [CV3-Eval](https://github.com/FunAudioLLM/CV3-Eval)

> **Note**: This repository only supports **Fun-CosyVoice3-0.5B-2512**. Legacy models (CosyVoice v1/v2) have been removed.

## Highlight 🔥

**Fun-CosyVoice 3.0** is an advanced text-to-speech (TTS) system based on large language models (LLM), surpassing its predecessor (CosyVoice 2.0) in content consistency, speaker similarity, and prosody naturalness. It is designed for zero-shot multilingual speech synthesis in the wild.

### Key Features
- **Language Coverage**: Covers 9 common languages (Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian), 18+ Chinese dialects/accents
- **Content Consistency & Naturalness**: Achieves state-of-the-art performance in content consistency, speaker similarity, and prosody naturalness
- **Pronunciation Inpainting**: Supports pronunciation inpainting of Chinese Pinyin and English CMU phonemes
- **Text Normalization**: Supports reading of numbers, special symbols and various text formats without a traditional frontend module
- **Bi-Streaming**: Support both text-in streaming and audio-out streaming, latency as low as 150ms
- **Instruct Support**: Supports various instructions such as languages, dialects, emotions, speed, volume, etc.

## Evaluation

| Model | Open-Source | Model Size | test-zh<br>CER (%) ↓ | test-zh<br>Speaker Similarity (%) ↑ | test-en<br>WER (%) ↓ | test-en<br>Speaker Similarity (%) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Human | - | - | 1.26 | 75.5 | 2.14 | 73.4 |
| Seed-TTS | ❌ | - | 1.12 | 79.6 | 2.25 | 76.2 |
| **Fun-CosyVoice3-0.5B-2512** | ✅ | 0.5B | 1.21 | 78.0 | 2.24 | 71.8 |
| Fun-CosyVoice3-0.5B-2512_RL | ✅ | 0.5B | 0.81 | 77.4 | 1.68 | 69.5 |

## Install

### Quick Start with Pixi (Recommended)

[Pixi](https://pixi.sh/) is a fast, cross-platform package manager that handles both Conda and PyPI dependencies.

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Clone the repository
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice

# Install dependencies
pixi install

# Download the model
pixi run download-model

# Run the example
pixi run example

# Start the web UI
pixi run webui
```

### Alternative: Conda Installation

If you prefer Conda, you can still use the traditional installation method:

```bash
# Clone the repo
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive

# Create and activate conda environment
conda create -n cosyvoice -y python=3.12
conda activate cosyvoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# If you encounter sox compatibility issues
# Ubuntu
sudo apt-get install sox libsox-dev
# CentOS
sudo yum install sox sox-devel
```

### Model Download

Download the pretrained Fun-CosyVoice3-0.5B model:

```python
# Using ModelScope SDK
from modelscope import snapshot_download

snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')

snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')

# Or using HuggingFace (for overseas users)
from huggingface_hub import snapshot_download

snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')

snapshot_download('FunAudioLLM/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

```bash
# Note: ttsfrd is currently only available for Python 3.8 and 3.10.
# It is NOT compatible with Python 3.12.
# If you require ttsfrd, please use a Python 3.10 environment.
```

## Troubleshooting

### Known Issues

- **Garbled Audio Output**: If you experience garbled or unintelligible audio (especially for English), ensure you are using `transformers==4.51.3`. Newer versions (e.g., 4.54+) effectively break the multilingual capabilities of this model. This repository's `pyproject.toml` pins this specific version automatically to prevent this issue.

- **Language Confusion**: For zero-shot voice cloning, it is recommended to add an explicit instruction to the prompt text, e.g., `"You are a helpful assistant. Please speak in English.<|endofprompt|>"`.

### Audio Quality Optimizations

This repository includes several optimizations for improved audio quality:

- **RL-trained LLM**: Loads `llm.rl.pt` by default for reduced mispronunciations and improved clarity
- **Tuned sampling**: `top_p=0.7` for more consistent output
- **Optimized vocoder**: `nsf_voiced_threshold=5` for better voicing detection

## Basic Usage

### Voice Cloning Example

```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

# Load the model
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

# Zero-shot voice cloning
prompt_wav = './asset/interstellar-tars-01-resemble-denoised.wav'
prompt_text = "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. Nothing's changed on that."

for i, output in enumerate(cosyvoice.inference_zero_shot(
    'Hello! I am an AI voice assistant. How may I help you today?',
    'You are a helpful assistant.<|endofprompt|>' + prompt_text,
    prompt_wav,
    stream=False
)):
    torchaudio.save(f'output_{i}.wav', output['tts_speech'], cosyvoice.sample_rate)
```

### Run Example Script

```bash
python example.py
```

### Start Web UI

```bash
python webui.py --port 8000
```

Or with pixi:

```bash
pixi run webui
```

## Inference Modes

Fun-CosyVoice3 supports three inference modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Zero-Shot (3s极速复刻)** | Clone voice from a short audio clip | Voice cloning with reference audio |
| **Cross-Lingual (跨语种复刻)** | Synthesize text in a different language than the prompt | Multilingual synthesis |
| **Instruction (自然语言控制)** | Control voice style with natural language instructions | Fine-grained control over speech |

## API Servers

### FastAPI Server

```bash
cd runtime/python/fastapi
python server.py --port 50000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B
```

Endpoints:
- `POST /inference_zero_shot` - Zero-shot voice cloning
- `POST /inference_cross_lingual` - Cross-lingual synthesis
- `POST /inference_instruct2` - Instruction-controlled synthesis
- `GET /health` - Health check

### gRPC Server

```bash
cd runtime/python/grpc
python server.py --port 50000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B
```

### Rust Server (High Performance)

The Rust server offers high-performance inference and can be run natively.

#### Build

```bash
cd rust
cargo build --release
```

> **Note**: The build configuration in `rust/.cargo/config.toml` automatically points PyO3 to the correct Python environment (pixi or system). You do not need to wrap the build command.

#### Run

```bash
# Run directly from project root to load .env configuration
./rust/target/release/cosyvoice-server
```

The server will automatically configure its environment (including `LD_LIBRARY_PATH` and Python setup) on startup.

### Docker Deployment

```bash
cd runtime/python
docker build -t cosyvoice:v3.0 .

# FastAPI server
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v3.0 \
    /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && \
    python3 server.py --port 50000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B && \
    sleep infinity"
```

## Advanced Usage

For advanced users, training and inference scripts are provided in `examples/libritts/cosyvoice/run.sh`.

## Discussion & Communication

You can directly discuss on [GitHub Issues](https://github.com/FunAudioLLM/CosyVoice/issues).

You can also scan the QR code to join our official DingDing chat group.

<img src="./asset/dingding.png" width="250px">

## Acknowledge

1. We borrowed a lot of code from [FunASR](https://github.com/modelscope/FunASR).
2. We borrowed a lot of code from [FunCodec](https://github.com/modelscope/FunCodec).
3. We borrowed a lot of code from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
4. We borrowed a lot of code from [AcademiCodec](https://github.com/yangdongchao/AcademiCodec).
5. We borrowed a lot of code from [WeNet](https://github.com/wenet-e2e/wenet).

## Citations

```bibtex
@article{du2025cosyvoice,
  title={CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author={Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and Yu, Fan and Zhao, Tianyu and Wang, Hao and Lv, Xiang and Wang, Hui and Shi, Xian and An, Keyu and others},
  journal={arXiv preprint arXiv:2505.17589},
  year={2025}
}

@article{du2024cosyvoice,
  title={Cosyvoice 2: Scalable streaming speech synthesis with large language models},
  author={Du, Zhihao and Wang, Yuxuan and Chen, Qian and Shi, Xian and Lv, Xiang and Zhao, Tianyu and Gao, Zhifu and Yang, Yexin and Gao, Changfeng and Wang, Hui and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}

@article{du2024cosyvoice,
  title={Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens},
  author={Du, Zhihao and Chen, Qian and Zhang, Shiliang and Hu, Kai and Lu, Heng and Yang, Yexin and Hu, Hangrui and Zheng, Siqi and Gu, Yue and Ma, Ziyang and others},
  journal={arXiv preprint arXiv:2407.05407},
  year={2024}
}
```

## Disclaimer

The content provided above is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet. If any content infringes on your rights, please contact us to request its removal.
