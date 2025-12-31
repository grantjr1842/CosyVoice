# Matcha-TTS to Native Rust Conversion Research Log

**Date**: 2025-12-31
**Objective**: Convert Matcha-TTS dependency to 100% native Rust implementation

## Executive Summary

CosyVoice currently depends on the **Matcha-TTS** third-party Python library for key components in its TTS pipeline. This document analyzes the complete dependency graph and creates a parity mapping for the native Rust conversion.

---

## 1. Matcha-TTS Dependency Analysis

### 1.1 Import Locations in CosyVoice

| File | Import Path | Components Used |
|------|-------------|-----------------|
| `cosyvoice/flow/decoder.py` | `matcha.models.components.decoder` | `SinusoidalPosEmb`, `Block1D`, `ResnetBlock1D`, `Downsample1D`, `TimestepEmbedding`, `Upsample1D` |
| `cosyvoice/flow/decoder.py` | `matcha.models.components.transformer` | `BasicTransformerBlock` |
| `cosyvoice/flow/flow_matching.py` | `matcha.models.components.flow_matching` | `BASECFM` |
| `cosyvoice/hifigan/hifigan.py` | `matcha.hifigan.models` | `feature_loss`, `generator_loss`, `discriminator_loss` |
| `tests/generate_test_artifacts.py` | `matcha.utils.audio` | `mel_spectrogram` |
| `cosyvoice3.yaml` | `matcha.utils.audio` | `mel_spectrogram` |
| `cosyvoice3.yaml` | `matcha.hifigan.models` | `MultiPeriodDiscriminator` |

### 1.2 Runtime PYTHONPATH Requirements

- `third_party/Matcha-TTS` is added to PYTHONPATH in:
  - `docker/Dockerfile`
  - `examples/*/path.sh`
  - `rust/server/src/tts.rs` (line 47: hard-coded path for PyO3)

---

## 2. Matcha-TTS Component Inventory

### 2.1 `matcha/models/components/decoder.py` (444 lines)

| Python Class/Function | Purpose | Parameters |
|----------------------|---------|------------|
| `SinusoidalPosEmb` | Sinusoidal position embedding for timesteps | `dim` |
| `Block1D` | Conv1d + GroupNorm + Mish block | `dim`, `dim_out`, `groups=8` |
| `ResnetBlock1D` | Residual block with time embedding | `dim`, `dim_out`, `time_emb_dim`, `groups=8` |
| `Downsample1D` | Conv1d stride=2 downsampling | `dim` |
| `TimestepEmbedding` | MLP for timestep conditioning | `in_channels`, `time_embed_dim`, `act_fn="silu"` |
| `Upsample1D` | ConvTranspose1d upsampling | `channels`, `use_conv_transpose=True` |
| `ConformerWrapper` | Optional Conformer blocks | (Not used by CosyVoice3) |
| `Decoder` | Full U-Net decoder with transformer blocks | Complex config |

### 2.2 `matcha/models/components/flow_matching.py` (133 lines)

| Python Class/Function | Purpose | Parameters |
|----------------------|---------|------------|
| `BASECFM` | Base class for Conditional Flow Matching | `n_feats`, `cfm_params`, `n_spks`, `spk_emb_dim` |
| `BASECFM.forward` | ODE-based mel generation | `mu`, `mask`, `n_timesteps`, `temperature` |
| `BASECFM.solve_euler` | Fixed Euler ODE solver | `x`, `t_span`, `mu`, `mask`, `spks`, `cond` |
| `BASECFM.compute_loss` | CFM training loss | `x1`, `mask`, `mu`, `spks`, `cond` |
| `CFM` | CFM with Decoder estimator | Extends BASECFM |

### 2.3 `matcha/models/components/transformer.py` (317 lines)

| Python Class/Function | Purpose | Parameters |
|----------------------|---------|------------|
| `SnakeBeta` | Snake activation with trainable alpha/beta | `in_features`, `out_features`, `alpha=1.0` |
| `FeedForward` | Transformer FFN with activation | `dim`, `mult=4`, `dropout`, `activation_fn` |
| `BasicTransformerBlock` | Full transformer block with norm | `dim`, `num_attention_heads`, `attention_head_dim`, `dropout` |

### 2.4 `matcha/hifigan/models.py` (369 lines)

| Python Class/Function | Purpose | Parameters |
|----------------------|---------|------------|
| `ResBlock1` | HiFiGAN residual block type 1 | `channels`, `kernel_size`, `dilation` |
| `ResBlock2` | HiFiGAN residual block type 2 | `channels`, `kernel_size`, `dilation` |
| `Generator` | HiFiGAN generator | `h` (config object) |
| `DiscriminatorP` | Period discriminator | `period`, `kernel_size` |
| `MultiPeriodDiscriminator` | Multi-period discriminator | - |
| `DiscriminatorS` | Scale discriminator | - |
| `MultiScaleDiscriminator` | Multi-scale discriminator | - |
| `feature_loss` | Feature matching loss | `fmap_r`, `fmap_g` |
| `discriminator_loss` | Discriminator loss | `disc_real_outputs`, `disc_generated_outputs` |
| `generator_loss` | Generator adversarial loss | `disc_outputs` |

### 2.5 `matcha/utils/audio.py` (83 lines)

| Python Class/Function | Purpose | Parameters |
|----------------------|---------|------------|
| `load_wav` | Load audio file | `full_path` |
| `dynamic_range_compression` | Log compress | `x`, `C=1`, `clip_val=1e-5` |
| `dynamic_range_decompression` | Exp decompress | `x`, `C=1` |
| `mel_spectrogram` | Compute mel spectrogram | `y`, `n_fft`, `num_mels`, `sampling_rate`, `hop_size`, `win_size`, `fmin`, `fmax` |
| `spectral_normalize_torch` | Log compress torch | `magnitudes` |

---

## 3. Existing Rust Implementation Status

### 3.1 `rust/server/src/flow.rs` (799 lines) ✅ MOSTLY COMPLETE

| Rust Struct | Python Equivalent | Status |
|-------------|-------------------|--------|
| `FlowConfig` | Config object | ✅ Complete |
| `TimestepEmbedding` | `matcha.TimestepEmbedding` | ✅ Complete |
| `sinusoidal_embedding()` | `SinusoidalPosEmb` | ✅ Complete |
| `AdaLayerNormZero` | Custom AdaLN | ✅ Complete |
| `AdaLayerNormZeroFinal` | Custom AdaLN Final | ✅ Complete |
| `DiTBlock` | CosyVoice DiT block | ✅ Complete |
| `FeedForward` | `matcha.FeedForward` | ✅ Complete |
| `Attention` | Multi-head attention | ✅ Complete |
| `apply_rotary_pos_emb()` | RoPE application | ✅ Complete |
| `RotaryEmbedding` | RoPE generation | ✅ Complete |
| `mish()` | Mish activation | ✅ Complete |
| `CausalConvPositionEmbedding` | Causal conv pos embed | ✅ Complete |
| `InputEmbedding` | Input projection | ✅ Complete |
| `DiT` | Full DiT estimator | ✅ Complete |
| `ConditionalCFM` | `matcha.BASECFM` ODE solver | ✅ Complete |

### 3.2 `rust/server/src/hift.rs` (957 lines) ✅ MOSTLY COMPLETE

| Rust Struct | Python Equivalent | Status |
|-------------|-------------------|--------|
| `Snake` | Snake activation | ✅ Complete |
| `SineGen` | Sine generator for NSF | ✅ Complete |
| `SourceModuleHnNSF` | NSF source module | ✅ Complete |
| `ResBlock` | HiFiGAN ResBlock | ✅ Complete |
| `F0Predictor` | F0 predictor | ✅ Complete |
| `HiFTGenerator` | Full HiFT vocoder | ✅ Complete |
| `HiFTConfig` | HiFT configuration | ✅ Complete |

### 3.3 Missing/Partial Components

| Component | Location | Status |
|-----------|----------|--------|
| `mel_spectrogram` | `matcha.utils.audio` | ❌ Not in Rust (uses torchaudio/librosa) |
| `Block1D` | `matcha.decoder` | ⚠️ Used only in Python decoder, not needed for DiT |
| `ResnetBlock1D` | `matcha.decoder` | ⚠️ Used only in Python decoder, not needed for DiT |
| `Downsample1D` | `matcha.decoder` | ⚠️ Used only in Python decoder, not needed for DiT |
| `Upsample1D` | `matcha.decoder` | ⚠️ Used only in Python decoder, not needed for DiT |
| `BasicTransformerBlock` | `matcha.transformer` | ⚠️ CosyVoice uses custom DiT, not this |
| `SnakeBeta` | `matcha.transformer` | ⚠️ Different from Snake in hift.rs |
| `MultiPeriodDiscriminator` | `matcha.hifigan` | ❌ Training only, not needed for inference |
| `feature_loss` | `matcha.hifigan` | ❌ Training only |
| `generator_loss` | `matcha.hifigan` | ❌ Training only |
| `discriminator_loss` | `matcha.hifigan` | ❌ Training only |

---

## 4. Parity Value Mapping

### 4.1 Flow Matching Constants

| Parameter | Matcha Python | Rust flow.rs | Notes |
|-----------|---------------|--------------|-------|
| `sigma_min` | 1e-4 | `sigma: f64` | Passed to `ConditionalCFM::new` |
| `t_scheduler` | "cosine" | Hardcoded in `forward()` | Line 722-724 |
| `inference_cfg_rate` | 0.7 | `cfg_rate = 0.7` | Hardcoded line 766 |
| `n_timesteps` | 10 (default) | Passed as arg | - |

### 4.2 Audio Processing Constants

| Parameter | Matcha Python | Value | Used In |
|-----------|---------------|-------|---------|
| `n_fft` | `matcha.utils.audio` | 1024 | mel_spectrogram |
| `num_mels` | `matcha.utils.audio` | 80 | mel_spectrogram |
| `sampling_rate` | `matcha.utils.audio` | 24000 | mel_spectrogram |
| `hop_size` | `matcha.utils.audio` | 256 | mel_spectrogram |
| `win_size` | `matcha.utils.audio` | 1024 | mel_spectrogram |
| `fmin` | `matcha.utils.audio` | 0 | mel_spectrogram |
| `fmax` | `matcha.utils.audio` | 8000 | mel_spectrogram |

### 4.3 HiFT/HiFiGAN Constants

| Parameter | Matcha Python | Rust Value | Notes |
|-----------|---------------|------------|-------|
| `upsample_rates` | [8, 8] | `[8, 8]` | HiFTConfig |
| `upsample_kernel_sizes` | [16, 16] | `[16, 16]` | HiFTConfig |
| `resblock_kernel_sizes` | [3, 7, 11] | `[3, 7, 11]` | HiFTConfig |
| `resblock_dilation_sizes` | [[1,3,5], [1,3,5], [1,3,5]] | `[[1,3,5], [1,3,5], [1,3,5]]` | HiFTConfig |
| `upsample_initial_channel` | 512 | `512` | HiFTConfig |
| `harmonic_num` | 8 | `8` | SineGen |
| `sine_amp` | 0.1 | `0.1` | SineGen |
| `noise_std` | 0.003 | `0.003` | SineGen |
| `sampling_rate` | 24000 | `24000` | SineGen |

---

## 5. Required Changes for Full Native Rust

### 5.1 Inference Pipeline (REQUIRED)

1. **Remove PYTHONPATH injection** in `rust/server/src/tts.rs` line 47
2. **Implement native `mel_spectrogram`** in Rust using `rustfft` or audio crate
   - Or keep using pre-computed mel from Python/ONNX frontend

### 5.2 Training Components (NOT REQUIRED for inference)

These are only needed if training will be done in Rust:
- `MultiPeriodDiscriminator`
- `MultiScaleDiscriminator`
- `feature_loss`, `generator_loss`, `discriminator_loss`
- `Block1D`, `ResnetBlock1D`, `Downsample1D`, `Upsample1D` (U-Net decoder)
- `BasicTransformerBlock` (Matcha's version)

### 5.3 Utility Functions (OPTIONAL)

These can remain Python or be ported:
- `dynamic_range_compression/decompression`
- `spectral_normalize_torch`
- `load_wav`

---

## 6. Verification Checklist

- [ ] `sinusoidal_embedding()` parity test
- [ ] `TimestepEmbedding` parity test
- [ ] `AdaLayerNormZero` parity test
- [ ] `DiTBlock` parity test
- [ ] `DiT.forward()` end-to-end parity
- [ ] `ConditionalCFM.forward()` ODE solver parity
- [ ] `Snake` activation parity
- [ ] `SineGen.forward()` parity
- [ ] `HiFTGenerator.forward()` parity
- [ ] Full pipeline: Flow → HiFT → Audio

---

## 7. Conclusion

The Rust implementation is **~95% complete** for inference. The remaining work is:

1. **Eliminate the Matcha-TTS PYTHONPATH requirement** in `tts.rs`
2. **Optionally** implement native `mel_spectrogram` (or accept pre-computed input)
3. **Verify parity** for all components against Python reference

No training components need to be ported since inference is the target.
