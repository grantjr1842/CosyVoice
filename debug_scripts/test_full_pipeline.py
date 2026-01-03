"""
Test HiFT with mel generated from the full CosyVoice pipeline (using Flow output).
"""

import sys

sys.path.insert(0, ".")

import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice3


def main():
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"

    # Load full CosyVoice model
    print("Loading CosyVoice model...")
    cosyvoice = CosyVoice3(model_dir)
    print("Model loaded!")

    # Test with simple text
    text = "你好世界"  # Hello World in Chinese

    print(f"Synthesizing: {text}")

    # Use instruct mode with prompt audio
    prompt_audio = "asset/zero_shot_prompt.wav"
    waveform, sr = torchaudio.load(prompt_audio)

    # Get the full output
    for i, result in enumerate(cosyvoice.inference_sft(text, "中文女", stream=False)):
        audio = result["tts_speech"]
        print(f"Chunk {i}: shape={audio.shape}")

        # Save the first chunk
        if i == 0:
            torchaudio.save("cosyvoice_full_output.wav", audio.unsqueeze(0), 24000)
            print("Saved to cosyvoice_full_output.wav")

            # Also check mel intermediate
            print(f"Audio max: {audio.abs().max().item()}")
            print(f"Audio mean: {audio.mean().item()}")
            print(f"Audio std: {audio.std().item()}")

        break


if __name__ == "__main__":
    main()
