# Research Log: CosyVoice Project Modernization

## Session Date: 2025-12-27

### User Request Summary
1. Configure project to work with pixi
2. Remove all references to models except Fun-CosyVoice3-0.5B-2512
3. Scope requirements for voice cloning feature with specific reference voice
4. Update documentation

---

## Repository Audit

### Current Project Structure
- Main directory: `/home/grant/github/CosyVoice-1`
- Package manager: Currently uses Conda + pip with `requirements.txt`
- No existing `pixi.toml` or `pyproject.toml` for Pixi configuration
- License: Apache 2.0

### Model Landscape

| Model | Version | Status | Notes |
|-------|---------|--------|-------|
| CosyVoice-300M | v1.0 | **To Remove** | Legacy base model |
| CosyVoice-300M-SFT | v1.0 | **To Remove** | SFT fine-tuned variant |
| CosyVoice-300M-Instruct | v1.0 | **To Remove** | Instruction-tuned variant |
| CosyVoice2-0.5B | v2.0 | **To Remove** | Previous generation |
| **Fun-CosyVoice3-0.5B-2512** | v3.0 | **Keep** | Latest model, target |

### Files with Model References

#### Python Files
- `example.py`: Lines 10, 17, 29, 39, 74 - Multiple model references
- `webui.py`: Lines 58-71, 127-129, 170 - Model-specific logic
- `cosyvoice/cli/cosyvoice.py`: Lines 27-136 - `CosyVoice` class for v1, lines 139-186 - `CosyVoice2` class
- `cosyvoice/bin/export_jit.py`: Line 34 - Default model path
- `cosyvoice/bin/export_onnx.py`: Line 48 - Default model path
- `vllm_example.py`: Line 15 - CosyVoice2 reference

#### Runtime/Server Files
- `runtime/python/grpc/server.py`: Line 93 - Default model
- `runtime/python/fastapi/server.py`: Line 91 - Default model
- `runtime/triton_trtllm/` - Multiple files referencing CosyVoice2

#### Documentation
- `README.md`: Lines 5-133 - Model download instructions, evaluation tables

### Voice Clip Analysis
Target reference voice: `asset/interstellar-tars-01-resemble-denoised.wav`
Transcription: "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. Nothing's changed on that."

---

## Pixi Configuration Research

### Key Findings
- Pixi is a Rust-based package manager supporting both Conda and PyPI
- Configuration via `pixi.toml` or `pyproject.toml`
- Lock files (`pixi.lock`) ensure reproducibility
- Task runner built-in

### Required Configuration Elements
1. Python version: 3.10+
2. Channels: conda-forge, pytorch
3. Platforms: linux-64 (primary)
4. System dependencies: CUDA, sox

### Migration Strategy
Convert from `requirements.txt` to `pyproject.toml` with:
- `[project]` section for metadata
- `[tool.pixi.workspace]` for channels/platforms
- `[tool.pixi.tasks]` for common commands
- `[tool.pixi.dependencies]` for Conda packages

---

## Scope: Voice Cloning Feature

### CosyVoice3 Inference Methods
From `cosyvoice/cli/cosyvoice.py` (CosyVoice3 class):

1. **`inference_zero_shot()`** - Zero-shot voice cloning ✓ (Target)
2. **`inference_cross_lingual()`** - Cross-lingual synthesis
3. **`inference_instruct2()`** - Instruction-controlled synthesis

### Minimum Requirements
1. Download Fun-CosyVoice3-0.5B-2512 model to `pretrained_models/Fun-CosyVoice3-0.5B`
2. Load with `AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')`
3. Call `inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False)`

### Default Voice Cloning Setup
```python
prompt_wav = './asset/interstellar-tars-01-resemble-denoised.wav'
prompt_text = "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. Nothing's changed on that."
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing workflows | Medium | Keep backward-compat `AutoModel` |
| GRPO training scripts broken | Low | Users likely don't need old model training |
| Pixi not installed | High | Provide installation instructions |

---

## Next Steps
1. Create implementation plan
2. Create Master GitHub Issue
3. Create sub-task issues
4. Execute changes
