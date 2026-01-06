# Native LLM Parity Guide (CosyVoice3)

This guide summarizes the key parity decisions and debugging workflow for matching
Python (PyTorch) and Rust (native) LLM behavior in CosyVoice3.

## Scope

- Focus: speech token generation (LLM sampling) parity for `example.py` and
  `rust/native-server/src/bin/native_example.rs`.
- Non-goals: Flow/HiFT parity and audio DSP fidelity.

## Key Decisions

1) Speaker embedding usage (LLM input)
- Python `CosyVoice3LM` inference does not use speaker embeddings.
- RL LLM weights (`llm.rl.pt` / `llm.rl.safetensors`) do not include
  `spk_embed_affine_layer.*` parameters.
- Rust should skip speaker embedding injection in the LLM input to match Python.

2) Sampling configuration
- The CosyVoice3 YAML sets `top_p=0.8`, `top_k=25`, `win_size=10`, `tau_r=0.1`.
- Rust should default to `top_p=0.8` and use the same RAS logic.

3) Prompt prefix
- Prompt prefix used for parity runs:
  "You are a helpful assistant. Please speak in English.<|endofprompt|>"
- Applied in both `example.py` and `native_example.rs`.

## References

- Python sampling config: `pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml`
- Python inference path: `cosyvoice/llm/llm.py`
- Rust LLM: `rust/native-server/src/cosyvoice_llm.rs`
- Example scripts: `example.py`, `rust/native-server/src/bin/native_example.rs`

## Debug Workflow

### 1) Run Python example with token logging

Set `COSYVOICE_DEBUG_TOKENS=1` to emit LLM token counts.

Command:
```
COSYVOICE_DEBUG_TOKENS=1 pixi run python example.py 2>&1 | tee output/py_token_log.txt
```

Expected log hints:
- Prompt token length
- Per-segment token lengths
- `LLM generated N speech tokens`

### 2) Run native example with token logging

Command:
```
cargo run --manifest-path rust/native-server/Cargo.toml --bin native_example --release --features cuda 2>&1 | tee output/native_token_log.txt
```

Expected log hints:
- Prompt token length
- Per-segment token lengths
- `LLM: stop token reached at step ...`
- `Generated N speech tokens`

### 3) Compare durations and RMS

Command:
```
for f in output/voice_clone_0_0.wav output/voice_clone_1_0.wav \
         output/native_voice_clone_0_0.wav output/native_voice_clone_1_0.wav; do
  echo "$f"; soxi -D "$f";
done

for f in output/voice_clone_0_0.wav output/voice_clone_1_0.wav \
         output/native_voice_clone_0_0.wav output/native_voice_clone_1_0.wav; do
  echo "$f"; sox "$f" -n stat 2>&1 | rg "RMS|Maximum amplitude";
done
```

## Recent Observations (example runs)

- Python tokens: 214 / 175 (segments 1 / 2)
- Native tokens: 147 / 195 (segments 1 / 2)
- Python durations: 8.56s / 7.00s
- Native durations: 5.88s / 7.80s

These deltas suggest remaining sampling or tokenization differences.

## Next Investigation Steps

1) Add deterministic RNG seeding in Rust to reduce sampling variance.
2) Inject Python prompt speech tokens into the native example to isolate LLM.
3) Log and compare the first N generated token IDs for identical inputs.

