# ADR-001: Align Native LLM Sampling With Python Defaults

## Status
Accepted

## Context

The Rust native LLM path produced speech token lengths and audio durations that
were out of parity with the Python CosyVoice3 example. The Python configuration
comes from `pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml`, which sets
RAS sampling parameters and uses an LLM inference path that does not condition
on speaker embeddings.

## Decision

1) Match the Python sampling defaults by setting `top_p=0.8` and using RAS
sampling with `top_k=25`, `win_size=10`, and `tau_r=0.1`.

2) Skip speaker embedding injection in the native LLM input to mirror the Python
`CosyVoice3LM` inference path and because `llm.rl.safetensors` does not provide
`spk_embed_affine_layer.*`.

3) Standardize prompt prefix usage and token-length logging for parity runs.

## Consequences

- Native sampling is closer to Python defaults and easier to compare.
- Speaker conditioning in the LLM path is disabled in Rust for parity purposes.
- Additional logging improves observability during parity debugging.

## Alternatives Considered

### Keep speaker embeddings in LLM
- Pros: explicit speaker conditioning.
- Cons: mismatched with Python inference behavior and missing weights.

### Use top_p=0.7
- Pros: simpler alignment with earlier Rust defaults.
- Cons: diverges from YAML configuration used by Python.

## Implementation

- `rust/native-server/src/cosyvoice_llm.rs`: set default `sampling_top_p=0.8` and
  skip speaker embedding injection in the LLM input.
- `example.py` and `rust/native-server/src/bin/native_example.rs`: aligned prompt
  prefix and token length logging.
- `cosyvoice/cli/model.py`: optional token count logging via
  `COSYVOICE_DEBUG_TOKENS=1`.

## References

- `pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml`
- `cosyvoice/llm/llm.py`
- `rust/native-server/src/cosyvoice_llm.rs`
- `docs/native_llm_parity_guide.md`
