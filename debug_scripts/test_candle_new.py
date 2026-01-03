from candle_core import VarBuilder, DType, Device
import os

p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors')
print(f"Testing Candle VarBuilder from_mmaped_safetensors with: {p}")
try:
    vb = VarBuilder.from_mmaped_safetensors([p], DType.F32, Device.Cpu)
    print("SUCCESS")
except Exception as e:
    print(f"FAIL: {e}")
