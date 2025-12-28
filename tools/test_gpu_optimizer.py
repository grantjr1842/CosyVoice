import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cosyvoice.utils.gpu_optimizer import GpuOptimizer


class TestGpuOptimizer(unittest.TestCase):
    @patch("cosyvoice.utils.gpu_optimizer.torch.cuda")
    def test_no_gpu(self, mock_cuda):
        mock_cuda.is_available.return_value = False
        mock_cuda.device_count.return_value = 0

        optimizer = GpuOptimizer()
        params = optimizer.suggest_parameters()

        self.assertFalse(params["fp16"])

    @patch("cosyvoice.utils.gpu_optimizer.torch.cuda")
    def test_modern_gpu_high_vram(self, mock_cuda):
        # Simulate verify modern GPU (e.g., A100/3090)
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 1
        mock_cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3090"

        mock_props = MagicMock()
        mock_props.total_memory = 24 * (1024**3)  # 24 GB
        mock_cuda.get_device_properties.return_value = mock_props

        mock_cuda.get_device_capability.return_value = (8, 6)  # Ampere

        optimizer = GpuOptimizer()
        params = optimizer.suggest_parameters()

        # Should be True because capability >= 7.0
        self.assertTrue(params["fp16"])

    @patch("cosyvoice.utils.gpu_optimizer.torch.cuda")
    def test_pascal_low_vram(self, mock_cuda):
        # Simulate Pascal GPU with low VRAM (e.g., GTX 1060 6GB)
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 1
        mock_cuda.get_device_name.return_value = "NVIDIA GeForce GTX 1060"

        mock_props = MagicMock()
        mock_props.total_memory = 6 * (1024**3)  # 6 GB
        mock_cuda.get_device_properties.return_value = mock_props

        mock_cuda.get_device_capability.return_value = (6, 1)  # Pascal

        optimizer = GpuOptimizer()
        params = optimizer.suggest_parameters()

        # Should be True because Pascal (6.x) and VRAM < 8GB
        self.assertTrue(params["fp16"])

    @patch("cosyvoice.utils.gpu_optimizer.torch.cuda")
    def test_old_gpu(self, mock_cuda):
        # Simulate older GPU (e.g., Maxwell)
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 1
        mock_cuda.get_device_name.return_value = "NVIDIA GeForce GTX 980"

        mock_props = MagicMock()
        mock_props.total_memory = 4 * (1024**3)  # 4 GB
        mock_cuda.get_device_properties.return_value = mock_props

        mock_cuda.get_device_capability.return_value = (5, 2)  # Maxwell

        optimizer = GpuOptimizer()
        params = optimizer.suggest_parameters()

        # Should be False because capability < 6.0 (Maxwell doesn't do fp16 well usually or logic excludes it)
        # Verify logic: if major >= 7: True. elif major == 6 and vram < 8: True. else: False at end?
        # Initial params['fp16'] = False.
        # major=5. Conditions failed. Output False.
        self.assertFalse(params["fp16"])


if __name__ == "__main__":
    unittest.main()
