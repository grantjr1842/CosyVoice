#!/bin/bash
set -e

# Compile Rust first
echo "Building Rust binary..."
pixi run cargo build --manifest-path rust/Cargo.toml --release --bin native_example

# Benchmark Configuration
ITERATIONS=5
export BENCHMARK_ITERATIONS=$ITERATIONS

P1="Hello world."
P2="The quick brown fox jumps over the lazy dog. This is a medium length sentence to test the model."
P3="The concept of general relativity, proposed by Albert Einstein, suggests that gravity is not a force but a curvature of spacetime caused by mass and energy. This theory revolutionized our understanding of the universe, leading to predictions of black holes and gravitational waves."

run_benchmark() {
    PROMPT_NAME=$1
    PROMPT_TEXT=$2

    echo "========================================"
    echo "Benchmark: $PROMPT_NAME"
    echo "Text: $PROMPT_TEXT"
    export COSYVOICE_TEXT="$PROMPT_TEXT"

    echo "----------------------------------------"
    echo "Running PYTHON..."
    pixi run python example.py | grep "BENCH_RESULT"

    echo "----------------------------------------"
    echo "Running RUST..."
    ./rust/target/release/native_example | grep "BENCH_RESULT"
    echo "========================================"
}

run_benchmark "Short" "$P1"
run_benchmark "Medium" "$P2"
run_benchmark "Long" "$P3"
