#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd, env=None, cwd=None):
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Python/Rust HiFT parity checks and compare intermediates."
    )
    parser.add_argument("--skip-python", action="store_true", help="Skip Python parity run.")
    parser.add_argument("--skip-rust", action="store_true", help="Skip Rust parity run.")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio comparison.")
    parser.add_argument(
        "--artifact-path",
        help="Override ARTIFACT_PATH for both Python and Rust runs.",
    )
    parser.add_argument(
        "--use-f0-override",
        action="store_true",
        help="Use python f0_output as override in Rust run.",
    )
    parser.add_argument(
        "--f0-override-path",
        help="Path to safetensors containing f0_output for Rust override.",
    )
    parser.add_argument(
        "--use-stft-override",
        action="store_true",
        help="Use python s_stft_real/imag as override in Rust run.",
    )
    parser.add_argument(
        "--stft-override-path",
        help="Path to safetensors containing s_stft_real/imag for Rust override.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="MAE tolerance for compare_hift_intermediates.py",
    )
    parser.add_argument(
        "--rust-features",
        help="Comma-separated Cargo features to enable for the Rust run.",
    )
    parser.add_argument(
        "--use-f0-libtorch",
        action="store_true",
        help="Enable torchscript f0 predictor in Rust (requires f0-libtorch feature).",
    )
    parser.add_argument(
        "--f0-torchscript-path",
        help="Path to torchscript f0 predictor file.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python_parity = repo_root / "tests" / "test_hift_parity.py"
    rust_cmd = ["cargo", "run", "-p", "cosyvoice-native-server", "--bin", "test_native"]
    compare_script = repo_root / "tools" / "compare_hift_intermediates.py"
    audio_compare = repo_root / "tests" / "analyze_audio.py"

    if not args.skip_python:
        py_env = os.environ.copy()
        if args.artifact_path:
            py_env["ARTIFACT_PATH"] = str(Path(args.artifact_path).resolve())
        if args.skip_audio or args.skip_rust:
            py_env["SKIP_AUDIO_COMPARE"] = "1"
        run([sys.executable, str(python_parity)], env=py_env)

    if not args.skip_rust:
        env = os.environ.copy()
        env.setdefault("SAVE_HIFT_DEBUG", "1")
        env.setdefault("TEST_HIFT_ONLY", "1")
        env.setdefault(
            "COSYVOICE_MODEL_DIR",
            str(repo_root / "pretrained_models" / "Fun-CosyVoice3-0.5B"),
        )
        env.setdefault(
            "ARTIFACT_PATH",
            str(repo_root / "tests" / "test_artifacts.safetensors"),
        )
        if args.artifact_path:
            env["ARTIFACT_PATH"] = str(Path(args.artifact_path).resolve())
        seed_path = repo_root / "outputs" / "debug" / "python_intermediates.safetensors"
        if seed_path.exists():
            env.setdefault("HIFT_PARITY_SEEDS_PATH", str(seed_path.resolve()))

        override_path = None
        if args.f0_override_path:
            override_path = Path(args.f0_override_path)
        elif args.use_f0_override:
            override_path = repo_root / "outputs" / "debug" / "python_intermediates.safetensors"
        if override_path is not None:
            if override_path.exists():
                env.setdefault("HIFT_F0_OVERRIDE_PATH", str(override_path.resolve()))
            else:
                print(f"F0 override path not found: {override_path}")

        stft_override_path = None
        if args.stft_override_path:
            stft_override_path = Path(args.stft_override_path)
        elif args.use_stft_override:
            stft_override_path = repo_root / "outputs" / "debug" / "python_intermediates.safetensors"
        if stft_override_path is not None:
            if stft_override_path.exists():
                env.setdefault(
                    "HIFT_S_STFT_OVERRIDE_PATH", str(stft_override_path.resolve())
                )
            else:
                print(f"s_stft override path not found: {stft_override_path}")
        if args.use_f0_libtorch or args.f0_torchscript_path:
            ts_path = (
                Path(args.f0_torchscript_path)
                if args.f0_torchscript_path
                else repo_root
                / "pretrained_models"
                / "Fun-CosyVoice3-0.5B"
                / "f0_predictor.ts"
            )
            if ts_path.exists():
                env.setdefault("HIFT_F0_TORCHSCRIPT_PATH", str(ts_path.resolve()))
            else:
                print(f"f0 torchscript path not found: {ts_path}")
        if args.rust_features or args.use_f0_libtorch:
            features = args.rust_features or "f0-libtorch"
            rust_cmd = rust_cmd + ["--features", features]
        run(rust_cmd, env=env, cwd=repo_root / "rust")

    run([sys.executable, str(compare_script), "--tolerance", str(args.tolerance)])

    if args.skip_audio:
        return

    py_wav = repo_root / "outputs" / "audio" / "hift_parity" / "python_hift_output.wav"
    rust_wav = repo_root / "rust" / "outputs" / "audio" / "native_hift_output.wav"
    if not rust_wav.exists():
        rust_wav = repo_root / "outputs" / "audio" / "native_hift_output.wav"
    missing = [str(path) for path in (py_wav, rust_wav) if not path.exists()]
    if missing:
        print(f"Skipping audio comparison; missing files: {', '.join(missing)}")
        return

    run([sys.executable, str(audio_compare), str(py_wav), str(rust_wav)])


if __name__ == "__main__":
    main()
