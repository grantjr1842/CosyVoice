# Audio Quality Research Log

## Date: 2025-12-29

## Objective
Improve the audio quality of CosyVoice3 TTS output.

## Research Findings

### Key Audio Quality Factors in CosyVoice

Based on web research and codebase analysis:

1. **Input Reference Audio Quality**
   - Clear reference audio (3-10 seconds) is crucial
   - Single speaker, minimal background noise
   - Natural speaking pace

2. **Sampling Parameters (LLM)**
   - `top_p` (nucleus sampling): controls diversity (default: 0.8)
   - `top_k`: limits token choices (default: 25)
   - `tau_r`: repetition penalty (default: 0.1)
   - Higher quality often comes from lower sampling temperature

3. **Flow Matching (CFM/DiT) Parameters**
   - `inference_cfg_rate`: Classifier-free guidance strength (default: 0.7)
   - `sigma_min`: minimum noise level
   - Higher CFG rate = more adherence to content but potentially less natural

4. **HiFi-GAN Vocoder**
   - `audio_limit`: max amplitude (default: 0.99)
   - Upsample rates and kernel sizes affect quality
   - `nsf_alpha` and `nsf_sigma` control harmonic generation

5. **Post-Processing Opportunities**
   - Noise reduction
   - Audio level normalization
   - Sample rate upscaling (24kHz → 48kHz)

## Potential Improvements

### Quick Wins
1. **Improve reference audio quality** - Ensure prompt audio is clean and well-recorded
2. **Tune sampling parameters** - Lower `top_p` for more consistent output
3. **Enable supersampling** - Output at higher sample rates

### Advanced
1. **Enable torch.compile optimizations** - Already implemented
2. **Use 32-bit precision for vocoder** - May improve quality at cost of speed
3. **Post-process with audio enhancement** - External tool integration
