#!/bin/bash
set -e

# Compile Rust first
echo "Building Rust binary..."
pixi run cargo build --manifest-path rust/Cargo.toml --release --bin native_example

P1="Hello world."
P2="The quick brown fox jumps over the lazy dog. This is a medium length sentence to test the model."
P3="The concept of general relativity, proposed by Albert Einstein, suggests that gravity is not a force but a curvature of spacetime caused by mass and energy. This theory revolutionized our understanding of the universe, leading to predictions of black holes and gravitational waves."

echo "========================================"
echo "Benchmark 1: Short Prompt"
echo "Text: $P1"
echo "----------------------------------------"
echo "PYTHON:"
export COSYVOICE_TEXT="$P1"
/usr/bin/time -f "Real: %e s, User: %U s, Sys: %S s" pixi run python example.py
echo "----------------------------------------"
echo "RUST:"
/usr/bin/time -f "Real: %e s, User: %U s, Sys: %S s" ./rust/target/release/native_example
echo "========================================"

echo "Benchmark 2: Medium Prompt"
echo "Text: $P2"
echo "----------------------------------------"
echo "PYTHON:"
export COSYVOICE_TEXT="$P2"
/usr/bin/time -f "Real: %e s, User: %U s, Sys: %S s" pixi run python example.py
echo "----------------------------------------"
echo "RUST:"
/usr/bin/time -f "Real: %e s, User: %U s, Sys: %S s" ./rust/target/release/native_example
echo "========================================"

echo "Benchmark 3: Long Prompt"
echo "Text: $P3"
echo "----------------------------------------"
echo "PYTHON:"
export COSYVOICE_TEXT="$P3"
/usr/bin/time -f "Real: %e s, User: %U s, Sys: %S s" pixi run python example.py
echo "----------------------------------------"
echo "RUST:"
/usr/bin/time -f "Real: %e s, User: %U s, Sys: %S s" ./rust/target/release/native_example
echo "========================================"
