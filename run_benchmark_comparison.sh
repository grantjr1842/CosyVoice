#!/bin/bash
set -e

export BENCHMARK_ITERATIONS=3
export COSYVOICE_MODEL_DIR="pretrained_models/Fun-CosyVoice3-0.5B"

echo "=== Running Python Benchmark (example.py) ==="
pixi run python example.py > benchmark_python.log 2>&1
echo "Python benchmark complete. Log: benchmark_python.log"

echo "=== Running Rust Benchmark (native_example) ==="
# Ensure release build for fair comparison
pixi run cargo build --release --bin native_example --manifest-path rust/native-server/Cargo.toml
pixi run cargo run --release --bin native_example --manifest-path rust/native-server/Cargo.toml > benchmark_rust.log 2>&1
echo "Rust benchmark complete. Log: benchmark_rust.log"

echo "=== Benchmark Complete ==="
