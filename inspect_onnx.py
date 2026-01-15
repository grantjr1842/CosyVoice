import onnxruntime as ort
import os

model_path = "pretrained_models/Fun-CosyVoice3-0.5B/flow.decoder.estimator.fp32.onnx"

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    exit(1)

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

print("inputs:")
for i in sess.get_inputs():
    print(f"  {i.name}: {i.shape}, type={i.type}")

print("outputs:")
for o in sess.get_outputs():
    print(f"  {o.name}: {o.shape}, type={o.type}")
