
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel
import sys
import os
from safetensors.torch import save_file

def main():
    print("Initializing CosyVoice to dump frontend artifacts...")
    # Use the same model as native_example
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"
    cosyvoice = AutoModel(model_dir=model_dir)

    wav_path = "asset/interstellar-tars-01-resemble-denoised.wav"
    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found.")
        sys.exit(1)

    print(f"Processing {wav_path}...")

    # 1. Speech Tokens
    # Note: frontend._extract_speech_token expects 16k audio
    speech_token, speech_token_len = cosyvoice.frontend._extract_speech_token(wav_path)
    print(f"Speech Tokens Shape: {speech_token.shape}")
    print(f"Speech Tokens (First 20): {speech_token[0, :20]}")

    # 2. Speaker Embedding
    embedding = cosyvoice.frontend._extract_spk_embedding(wav_path)
    print(f"Speaker Embedding Shape: {embedding.shape}")

    # 3. Speech Feat (mel 24k) - used for Flow prompt
    speech_feat, speech_feat_len = cosyvoice.frontend._extract_speech_feat(wav_path)
    print(f"Speech Feat Shape: {speech_feat.shape}")

    # 4. Whisper Log Mel (16k) - for Speech Tokenizer
    speech_16k = torchaudio.load(wav_path)[0]
    if speech_16k.size(0) > 1:
        speech_16k = speech_16k.mean(dim=0, keepdim=True)
    if speech_16k.shape[1] / 16000 > 30:
        print("Warning: Audio too long (>30s)")

    resampler = torchaudio.transforms.Resample(orig_freq=cosyvoice.sample_rate, new_freq=16000)
    # Note: cosyvoice.sample_rate is 24000. load_wav might have loaded original sr?
    # cosyvoice.frontend uses load_wav(path, 16000) which does resampling.
    # Let's use internal method to be exact.
    # But internal method doesn't return mel.

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
        "whisper_mel": whisper_mel.cpu()
    }

    save_path = "frontend_artifacts.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, save_path)
    print(f"Saved artifacts to {save_path}")

if __name__ == "__main__":
    main()
