#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio
from safetensors.torch import save_file

from cosyvoice.cli.cosyvoice import AutoModel


def main():
    parser = argparse.ArgumentParser(description="Dump Python frontend artifacts for parity.")
    parser.add_argument(
        "--model-dir",
        default="pretrained_models/Fun-CosyVoice3-0.5B",
        help="Path to CosyVoice3 model directory.",
    )
    parser.add_argument(
        "--prompt-wav",
        default="asset/interstellar-tars-01-resemble-denoised.wav",
        help="Path to prompt WAV file.",
    )
    parser.add_argument(
        "--output",
        default="frontend_artifacts.safetensors",
        help="Output safetensors path.",
    )
    args = parser.parse_args()

    print("Initializing CosyVoice to dump frontend artifacts...")
    cosyvoice = AutoModel(model_dir=args.model_dir)

    wav_path = args.prompt_wav
    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found.")
        sys.exit(1)

    print(f"Processing {wav_path}...")

    # 1. Speech Tokens (expects 16k audio internally)
    speech_token, _speech_token_len = cosyvoice.frontend._extract_speech_token(wav_path)
    print(f"Speech Tokens Shape: {speech_token.shape}")
    print(f"Speech Tokens (First 20): {speech_token[0, :20]}")

    # 2. Speaker Embedding
    embedding = cosyvoice.frontend._extract_spk_embedding(wav_path)
    print(f"Speaker Embedding Shape: {embedding.shape}")

    # 3. Speech Feat (mel 24k) - used for Flow prompt
    speech_feat, _speech_feat_len = cosyvoice.frontend._extract_speech_feat(wav_path)
    print(f"Speech Feat Shape: {speech_feat.shape}")

    # 4. Whisper Log Mel (16k) - for Speech Tokenizer
    speech_16k = torchaudio.load(wav_path)[0]
    if speech_16k.size(0) > 1:
        speech_16k = speech_16k.mean(dim=0, keepdim=True)
    if speech_16k.shape[1] / 16000 > 30:
        print("Warning: Audio too long (>30s)")

    from cosyvoice.utils.file_utils import load_wav
    import whisper

    speech = load_wav(wav_path, 16000)
    whisper_mel = whisper.log_mel_spectrogram(speech, n_mels=128)
    print(f"Whisper Log Mel Shape: {whisper_mel.shape}")

    # Save to safetensors
    tensors = {
        "speech_tokens": speech_token.cpu(),
        "speaker_embedding": embedding.cpu(),
        "speech_feat": speech_feat.cpu(),
        "whisper_mel": whisper_mel.cpu(),
    }

    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, save_path)
    print(f"Saved artifacts to {save_path}")


if __name__ == "__main__":
    main()
