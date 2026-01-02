#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd, env=None):
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Python/Rust HiFT parity checks and compare intermediates."
    )
    parser.add_argument("--skip-python", action="store_true", help="Skip Python parity run.")
    parser.add_argument("--skip-rust", action="store_true", help="Skip Rust parity run.")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio comparison.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="MAE tolerance for compare_hift_intermediates.py",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python_parity = repo_root / "tests" / "test_hift_parity.py"
    rust_cmd = ["cargo", "run", "-p", "cosyvoice-native-server", "--bin", "test_native"]
    compare_script = repo_root / "tools" / "compare_hift_intermediates.py"
    audio_compare = repo_root / "tests" / "analyze_audio.py"

    if not args.skip_python:
        run([sys.executable, str(python_parity)])

    if not args.skip_rust:
        env = os.environ.copy()
        env.setdefault("SAVE_HIFT_DEBUG", "1")
        run(rust_cmd, env=env)

    run([sys.executable, str(compare_script), "--tolerance", str(args.tolerance)])

    if args.skip_audio:
        return

    py_wav = repo_root / "outputs" / "audio" / "hift_parity" / "python_hift_output.wav"
    rust_wav = repo_root / "outputs" / "audio" / "native_hift_output.wav"
    missing = [str(path) for path in (py_wav, rust_wav) if not path.exists()]
    if missing:
        print(f"Skipping audio comparison; missing files: {', '.join(missing)}")
        return

    run([sys.executable, str(audio_compare), str(py_wav), str(rust_wav)])


if __name__ == "__main__":
    main()
