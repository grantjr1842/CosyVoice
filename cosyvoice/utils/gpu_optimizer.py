# Copyright (c) 2025 Startime (authors: Grant)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch


class GpuOptimizer:
    def __init__(self):
        self.device_count = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        self.device_name = (
            torch.cuda.get_device_name(0) if self.device_count > 0 else "CPU"
        )
        self.vram_gb = 0
        self.compute_capability = (0, 0)

        if self.device_count > 0:
            try:
                # Get VRAM in GB
                self.vram_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                # Get compute capability
                self.compute_capability = torch.cuda.get_device_capability(0)
            except Exception as e:
                logging.warning(f"Failed to get GPU properties: {e}")

    def suggest_parameters(self):
        """
        Suggests optimal parameters based on GPU hardware.
        Returns:
            dict: Dictionary containing suggested parameters (e.g., {'fp16': True})
        """
        params = {"fp16": False}

        if self.device_count == 0:
            logging.info("No GPU detected. Using default CPU parameters.")
            return params

        logging.info(
            f"Detected GPU: {self.device_name} | VRAM: {self.vram_gb:.2f}GB | Compute Capability: {self.compute_capability}"
        )

        # Logic for FP16
        # Enable FP16 if:
        # 1. Compute Capability >= 7.0 (Volta and newer have good Tensor Cores/FP16 support)
        # 2. VRAM is limited (< 8GB) but device is reasonably modern (>= 6.0 Pascal) to save memory

        major, minor = self.compute_capability

        if major >= 7:
            logging.info(
                "GPU Compute Capability >= 7.0. Suggesting FP16=True for performance."
            )
            params["fp16"] = True
        elif major == 6 and self.vram_gb < 8:
            logging.info(
                "GPU VRAM < 8GB on Pascal architecture. Suggesting FP16=True for memory efficiency."
            )
            params["fp16"] = True
        else:
            logging.info(
                "GPU does not meet criteria for auto-enabling FP16. Defaulting to False."
            )

        return params

    def suggest_compile_mode(self):
        """
        Suggests the best torch.compile mode.
        """
        if self.device_count == 0:
            return None

        # specific check for volta/turing/ampere vs older
        # reduce-overhead consumes more memory but is faster for small batches
        # max-autotune is very slow to compile

        if self.vram_gb >= 16:
            return "reduce-overhead"
        elif self.vram_gb >= 7:
            # On 8GB cards (which often report slightly less than 8), reduce-overhead might OOM but worth trying
            return "reduce-overhead"
        else:
            # Low VRAM, prefer default or no compilation if very tight
            return "default"

    def suggest_quantization(self):
        """
        Suggests quantization configuration.
        Returns:
            dict: Kwargs for QuantizationConfig (or None if no quantization suggested)
        """
        if self.device_count == 0:
            return None

        # If we have very high VRAM (e.g. A100 80GB), maybe we don't need quantization for 0.5B model
        # But for most consumer GPUs, 4-bit or 8-bit is good for speed/memory.
        # The model is small (0.5B), so maybe 8-bit is safer for quality, or even FP16 is fine.
        # But user asked for optimizations.

        # 0.5B model in FP16 is ~1GB.
        # Actually, for such a small model, quantization overhead might outweigh benefits on high-end cards.
        # But on low-end cards, it saves memory.

        # Strategy:
        # < 4GB VRAM: 4-bit
        # < 8GB VRAM: 8-bit
        # >= 8GB: FP16 (usually faster than quantized for small models due to overhead, unless memory bound)

        # However, for 'optimizations', let's offer 4-bit if user wants max speed/min memory?
        # Actually, 4-bit/8-bit compute is often slower than FP16 on adequate hardware.
        # BUT, if we are memory bandwidth bound, it helps.

        # Let's be aggressive with memory saving if VRAM is low.
        if self.vram_gb < 6:
            logging.info(
                f"VRAM ({self.vram_gb:.2f}GB) < 6GB. Suggesting 4-bit quantization."
            )
            return {"load_in_4bit": True}
        elif self.vram_gb < 12:
            logging.info(
                f"VRAM ({self.vram_gb:.2f}GB) < 12GB. Suggesting 8-bit quantization."
            )
            return {"load_in_8bit": True}

        return None
