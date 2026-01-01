## ModuleNotFoundError: No module named 'matcha'

**Update (2026-01)**: The `third_party/Matcha-TTS` dependency is no longer required for inference. The necessary components have been implemented in `cosyvoice/compat/matcha_compat.py`.

If you still encounter this error with older code, you have two options:

1. **Recommended**: Update your code to remove `sys.path.append("third_party/Matcha-TTS")` - it's no longer needed.
2. **Legacy**: If using the third_party module, execute `git submodule update --init --recursive` and set `export PYTHONPATH=third_party/Matcha-TTS`.

## cannot find resource.zip or cannot unzip resource.zip

Please make sure you have git-lfs installed. Execute

```sh
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl
```
