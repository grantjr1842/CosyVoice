# Upstream Synchronization Summary

## Overview
This document summarizes the synchronization of the local `FunAudioLLM/CosyVoice` repository with the upstream `FunAudioLLM/CosyVoice` repository.

**Date:** 2026-01-10
**Upstream Remote:** `https://github.com/FunAudioLLM/CosyVoice`
**Local Branch:** `sync-upstream` (merged from `feat/140-fix-prompt-leakage`)

## Changes Merged
The upstream `main` branch was merged into the local repository. Key updates include:
- **Model Architecture**: Introduction of `CosyVoice2Model` and refactoring of `CosyVoice3Model` to inherit from it.
- **Inference Support**: Enhanced support for `CosyVoice2` and `CosyVoice3` models, including specific prompt handling (e.g., `<|endofprompt|>`).
- **Dependencies**: Updates to `requirements.txt` and potentially other configuration files.
- **Documentation**: Updates to `README.md` including new evaluation tables and vLLM usage instructions.

## Conflict Resolutions
Several merge conflicts were encountered and resolved. The general strategy was to **prioritize local parity fixes and CosyVoice3 support** while incorporating valid upstream architectural improvements.

### 1. `README.md`
- **Conflict**: Upstream added comprehensive evaluation tables and vLLM usage. Local had simplified structure referencing only CosyVoice3.
- **Resolution**: **Merged**. Accepted upstream's detailed tables and vLLM verification but retained local notes about Audio Quality parity fixes.

### 2. `cosyvoice/llm/llm.py`
- **Conflict**: Upstream added `instruct_token` support in `prepare_lm_input_target`.
- **Resolution**: **Accepted Upstream**. The upstream changes enable instruction-following capabilities (CosyVoice3 Instruct), which is a desired feature.

### 3. `cosyvoice/cli/frontend.py`
- **Conflict**: Minor dictionary unpacking syntax in `frontend_zero_shot`.
- **Resolution**: **Accepted Upstream**. Used the upstream syntax `{**self.spk2info...}`.

### 4. `cosyvoice/cli/cosyvoice.py`
- **Conflict**: Upstream added validation for `<|endofprompt|>` token in CosyVoice3.
- **Resolution**: **Accepted Upstream**. This check is critical for CosyVoice3 model performance. A syntax error caused by a duplicate function signature during merge was manually fixed.

### 5. `cosyvoice/cli/model.py`
- **Conflict**: Major refactoring. Upstream introduced `CosyVoice2Model` base class.
- **Resolution**: **Accepted Upstream**. We adopted the new class structure (`CosyVoice3Model(CosyVoice2Model)`) to stay aligned with the upstream architecture.

### 6. `example.py`
- **Conflict**: Upstream validation logic vs Local clean example.
- **Resolution**: **Kept Local (HEAD)**. The local version provides a cleaner, focused example for `Fun-CosyVoice3`, avoiding legacy model references found in the upstream example.

### 7. `webui.py`
- **Conflict**: Upstream default model was `CosyVoice2`. Upstream validation messages were in Chinese.
- **Resolution**: **Kept Local (HEAD)**. Retained `Fun-CosyVoice3-0.5B` as the default and kept English validation messages.

### 8. `runtime/python/fastapi/server.py` & `grpc/server.py`
- **Conflict**: Quote styles and default model arguments.
- **Resolution**: **Kept Local (HEAD)**. Preserved consistent styling and `Fun-CosyVoice3` defaults.

## Verification
- **Python Example**: `pixi run example` executed successfully, verifying that the merged code runs and generates audio with `Fun-CosyVoice3`.
- **Rust Build**: `cargo build --release` passed, confirming that Rust backend compatibility is maintained.

## Next Steps
- Verify rigorous parity checks (LLM, Flow, HiFT) on the merged codebase if further development occurs.
- Push the synchronized branch to the remote repository.
