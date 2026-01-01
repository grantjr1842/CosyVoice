import os

import numpy as np
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper
from safetensors.torch import save_file

# Configuration based on finding
# Flow: 24k, n_fft=1920, hop=480, win=1920, 80 mels, center=False
FLOW_SR = 24000
FLOW_N_FFT = 1920
FLOW_HOP_LENGTH = 480
FLOW_WIN_LENGTH = 1920
FLOW_N_MELS = 80


def get_flow_feat_extractor():
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=FLOW_SR,
        n_fft=FLOW_N_FFT,
        win_length=FLOW_WIN_LENGTH,
        hop_length=FLOW_HOP_LENGTH,
        n_mels=FLOW_N_MELS,
        f_min=0.0,
        f_max=None,
        center=False,
        power=1.0,  # Magnitude
    )


def main():
    device = torch.device("cpu")
    wav_path = "asset/zero_shot_prompt.wav"
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"

    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found")
        return

    print(f"Loading {wav_path}...")
    waveform, sr = torchaudio.load(wav_path)

    # 1. Speech Tokenizer Features (16k, 128 mels) -> Tokens via ONNX
    print("Generating Speech Tokens (via ONNX)...")
    speech_16k = torchaudio.functional.resample(waveform, sr, 16000)

    # Init ORT session for speech tokenizer
    speech_tokenizer_path = os.path.join(model_dir, "speech_tokenizer_v3.onnx")
    if not os.path.exists(speech_tokenizer_path):
        print(f"Error: {speech_tokenizer_path} not found. Skipping ONNX gen.")
        return

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    sess_options.intra_op_num_threads = 1
    speech_session = onnxruntime.InferenceSession(
        speech_tokenizer_path, sess_options, providers=["CPUExecutionProvider"]
    )

    feat = whisper.log_mel_spectrogram(speech_16k.squeeze(0), n_mels=128)
    # Input 0: feats [1, 128, T] ? No, check signature
    # check cli/frontend.py:
    # self.speech_tokenizer_session.run(None, {inputs[0].name: feat... numpy(), inputs[1].name: length})
    # feat shape from whisper is [80, T] usually? No, n_mels=128. [128, T].
    # But frontend uses: feat.detach().cpu().numpy().
    # Note: whisper.log_mel_spectrogram returns [n_mels, T] (padded to 30s) or just [n_mels, T]?
    # It pads to 30s usually unless we handle it.
    # frontend.py line 144: feat = whisper.log_mel_spectrogram(speech, n_mels=128)
    # speech should be padded?
    # fronted.py load_wav does NOT pad blindly.

    feat_np = feat.unsqueeze(0).detach().cpu().numpy()  # [1, 128, T]
    feat_len = np.array([feat.shape[1]], dtype=np.int32)

    speech_tokens_out = (
        speech_session.run(
            None,
            {
                speech_session.get_inputs()[0].name: feat_np,
                speech_session.get_inputs()[1].name: feat_len,
            },
        )[0]
        .flatten()
        .tolist()
    )

    # Convert to tensor [1, T_tokens]
    speech_tokens_tensor = torch.tensor(
        [speech_tokens_out], dtype=torch.long
    )  # indices are usually long/int
    print(f"Speech Tokens Shape: {speech_tokens_tensor.shape}")

    # Also keep features for compatibility/reference?
    speech_token_feat = feat.unsqueeze(0)  # [1, 128, T]
    print(f"Speech Tokenizer Feat Shape: {speech_token_feat.shape}")

    # 2. Speaker Embedding (via ONNX)
    print("Generating Speaker Embedding (via ONNX)...")
    campplus_path = os.path.join(model_dir, "campplus.onnx")
    campplus_session = onnxruntime.InferenceSession(
        campplus_path, sess_options, providers=["CPUExecutionProvider"]
    )

    # Kaldi fbank
    fbank = kaldi.fbank(speech_16k, num_mel_bins=80, dither=0.0, sample_frequency=16000)
    fbank = fbank - fbank.mean(dim=0, keepdim=True)  # [T, 80]

    # Input to campplus: [1, T, 80]
    fbank_input = fbank.unsqueeze(0).detach().cpu().numpy()

    embedding_out = (
        campplus_session.run(
            None, {campplus_session.get_inputs()[0].name: fbank_input}
        )[0]
        .flatten()
        .tolist()
    )

    speaker_embedding_tensor = torch.tensor(
        [embedding_out], dtype=torch.float32
    )  # [1, 192]
    print(f"Speaker Embedding Shape: {speaker_embedding_tensor.shape}")

    # Also keep fbank for ref
    speaker_feat = fbank.unsqueeze(0)
    print(f"Speaker Feat Shape: {speaker_feat.shape}")

    # 3. Flow Features (24k, 80 mels)
    print("Generating Flow Features (24k, 80 mels)...")
    speech_24k = torchaudio.functional.resample(waveform, sr, FLOW_SR)

    # Try importing matcha, fallback to torchaudio
    try:
        from matcha.utils.audio import mel_spectrogram

        print("Using matcha.utils.audio.mel_spectrogram")
        flow_feat = mel_spectrogram(
            speech_24k,
            FLOW_N_FFT,
            FLOW_N_MELS,
            FLOW_SR,
            FLOW_HOP_LENGTH,
            FLOW_WIN_LENGTH,
            0,
            None,
            center=False,
        )
    except ImportError:
        print("matcha not found, falling back to torchaudio logic (mimicking matcha)")
        stft = torchaudio.transforms.Spectrogram(
            n_fft=FLOW_N_FFT,
            win_length=FLOW_WIN_LENGTH,
            hop_length=FLOW_HOP_LENGTH,
            center=False,
            power=1.0,  # Magnitude
        )
        mel_scale = torchaudio.transforms.MelScale(
            n_mels=FLOW_N_MELS,
            sample_rate=FLOW_SR,
            n_stft=FLOW_N_FFT // 2 + 1,
            f_min=0.0,
            f_max=None,
        )

        mag = stft(speech_24k)
        mel = mel_scale(mag)
        flow_feat = torch.log(torch.clamp(mel, min=1e-5))

    print(f"Flow Feat Shape: {flow_feat.shape}")

    # 4. Text Tokens
    print("Generating Text Tokens...")
    try:
        from transformers import AutoTokenizer

        # Adjust path if needed.
        # Can we find tokenizer.json in model_dir?
        # Fun-CosyVoice3-0.5B usually has tokenizer files.
        # But maybe inside "llm" folder or root?
        # If not, let's look for known tokenizer path or fallback to dummy.
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        text = "Hello world | This is a native engine test."
        text_ids = tokenizer.encode(text, return_tensors="pt")
        print(f"Text: '{text}' -> Tokens: {text_ids}")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}. Using dummy tokens.")
        # Dummy tokens need to be within vocab range (e.g. < 151936)
        text_ids = torch.tensor([[100, 101, 102]], dtype=torch.long)

    # 5. Noise Tensors for Parity Testing
    print("Generating Noise Tensors for Parity Testing...")
    torch.manual_seed(42)  # Fixed seed for reproducibility

    # Flow noise: shape matches mu tensor [1, mel_dim, mel_len]
    flow_mel_len = flow_feat.shape[2]
    flow_noise = torch.randn(1, FLOW_N_MELS, flow_mel_len)
    print(f"Flow Noise Shape: {flow_noise.shape}")

    # HiFT phase injection for SineGen: [batch, harmonic_num, audio_len]
    # harmonic_num = 8 (from config), audio_len = mel_len * 480 (upsample scale)
    harmonic_num = 8
    audio_len = flow_mel_len * 480
    hift_phase = (torch.rand(1, harmonic_num, audio_len) * 2 * np.pi) - np.pi
    print(f"HiFT Phase Shape: {hift_phase.shape}")

    # Save all to safetensors
    tensors = {
        "speech_token_feat_16k": speech_token_feat.contiguous(),  # [1, 128, T]
        "speaker_feat_16k": speaker_feat.contiguous(),  # [1, T, 80]
        "flow_feat_24k": flow_feat.contiguous(),  # [1, 80, T_24k]
        "text_ids": text_ids.int().contiguous(),  # [1, L]
        # New additions:
        "speech_tokens": speech_tokens_tensor.int().contiguous(),  # [1, T_tokens]
        "speaker_embedding": speaker_embedding_tensor.contiguous(),  # [1, 192]
        # Parity testing noise tensors:
        "flow_noise": flow_noise.contiguous(),  # [1, 80, mel_len]
        "hift_phase": hift_phase.float().contiguous(),  # [1, 8, audio_len]
    }

    save_file(tensors, "tests/test_artifacts.safetensors")
    print("Saved to tests/test_artifacts.safetensors")


if __name__ == "__main__":
    main()
