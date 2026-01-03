#!/usr/bin/env python3
"""
HiFT Vocoder Parameter Tuning Experiment

Tests different NSF (Neural Source Filter) parameter combinations
to find optimal settings for audio quality.
"""


# Note: Matcha-TTS path no longer needed - using cosyvoice.compat.matcha_compat

import os
from pathlib import Path

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoice3Model


def run_experiment():
    """Run A/B tests with different vocoder parameters."""

    MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
    OUTPUT_DIR = Path("output/vocoder_experiments")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference audio and text
    PROMPT_WAV = "./asset/interstellar-tars-01-resemble-denoised.wav"
    PROMPT_TEXT = "Please speak in English.<|endofprompt|>Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. Nothing's changed on that."
    TTS_TEXT = "Hello! I am testing different vocoder parameters to find the best audio quality."

    # Parameter combinations to test
    # Format: (nsf_alpha, nsf_sigma, nsf_voiced_threshold)
    PARAM_COMBOS = [
        # Baseline
        (0.1, 0.003, 10),
        # Less noise
        (0.1, 0.001, 10),
        (0.1, 0.002, 10),
        # More harmonics
        (0.15, 0.003, 10),
        (0.2, 0.003, 10),
        # Combined: more harmonics, less noise
        (0.15, 0.001, 10),
        (0.15, 0.002, 10),
        # Different voicing threshold
        (0.1, 0.003, 5),
        (0.1, 0.003, 15),
    ]

    print("=" * 60)
    print("HiFT Vocoder Parameter Tuning Experiment")
    print("=" * 60)

    # Load base config
    hyper_yaml_path = f"{MODEL_DIR}/cosyvoice3.yaml"
    with open(hyper_yaml_path, "r") as f:
        base_configs = load_hyperpyyaml(
            f,
            overrides={
                "qwen_pretrain_path": os.path.join(MODEL_DIR, "CosyVoice-BlankEN")
            },
        )

    # Setup frontend (shared across experiments)
    frontend = CosyVoiceFrontEnd(
        base_configs["get_tokenizer"],
        base_configs["feat_extractor"],
        f"{MODEL_DIR}/campplus.onnx",
        f"{MODEL_DIR}/speech_tokenizer_v3.onnx",
        f"{MODEL_DIR}/spk2info.pt",
        base_configs["allowed_special"],
    )
    sample_rate = base_configs["sample_rate"]

    # Normalize texts
    prompt_text_norm = frontend.text_normalize(
        PROMPT_TEXT, split=False, text_frontend=True
    )
    tts_texts = frontend.text_normalize(TTS_TEXT, split=True, text_frontend=True)

    print(f"\n📝 TTS Text: {TTS_TEXT}")
    print(f"🎤 Prompt: {PROMPT_WAV}")
    print(f"\n🧪 Testing {len(PARAM_COMBOS)} parameter combinations...\n")

    results = []

    for idx, (alpha, sigma, threshold) in enumerate(PARAM_COMBOS):
        print(
            f"\n[{idx + 1}/{len(PARAM_COMBOS)}] Testing: nsf_alpha={alpha}, nsf_sigma={sigma}, threshold={threshold}"
        )

        # Reload config with modified HiFT parameters
        with open(hyper_yaml_path, "r") as f:
            configs = load_hyperpyyaml(
                f,
                overrides={
                    "qwen_pretrain_path": os.path.join(MODEL_DIR, "CosyVoice-BlankEN"),
                },
            )

        # Modify HiFT source module parameters
        hift = configs["hift"]
        hift.m_source.sine_amp = alpha
        hift.m_source.noise_std = sigma
        hift.m_source.voiced_threshold = threshold

        # Create model with modified HiFT
        model = CosyVoice3Model(configs["llm"], configs["flow"], hift, fp16=True)

        # Load RL weights
        llm_path = (
            f"{MODEL_DIR}/llm.rl.pt"
            if os.path.exists(f"{MODEL_DIR}/llm.rl.pt")
            else f"{MODEL_DIR}/llm.pt"
        )
        model.load(llm_path, f"{MODEL_DIR}/flow.pt", f"{MODEL_DIR}/hift.pt")

        # Generate audio
        for tts_text in tts_texts:
            model_input = frontend.frontend_zero_shot(
                tts_text, prompt_text_norm, PROMPT_WAV, sample_rate, ""
            )

            for output in model.tts(**model_input, stream=False, speed=1.0):
                output_path = (
                    OUTPUT_DIR
                    / f"exp_{idx:02d}_alpha{alpha}_sigma{sigma}_thresh{threshold}.wav"
                )
                torchaudio.save(str(output_path), output["tts_speech"], sample_rate)

                duration = output["tts_speech"].shape[1] / sample_rate
                print(f"   💾 Saved: {output_path.name} ({duration:.2f}s)")

                results.append(
                    {
                        "file": output_path.name,
                        "alpha": alpha,
                        "sigma": sigma,
                        "threshold": threshold,
                        "duration": duration,
                    }
                )

        # Clear GPU memory
        del model, configs
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print("\n🎧 Listen to each file and compare quality:")
    for r in results:
        print(
            f"   - {r['file']}: alpha={r['alpha']}, sigma={r['sigma']}, thresh={r['threshold']}"
        )

    print("\n💡 Key things to listen for:")
    print("   - Breathiness (lower with higher alpha)")
    print("   - Noise/hiss (lower with lower sigma)")
    print("   - Voicing artifacts (affected by threshold)")


if __name__ == "__main__":
    run_experiment()
