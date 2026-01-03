import json
import os

config_path = "/home/grant/github/CosyVoice-1/pretrained_models/Fun-CosyVoice3-0.5B/config.json"
blank_en_config_path = "/home/grant/github/CosyVoice-1/pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN/config.json"

with open(config_path, 'r') as f:
    config = json.load(f)

with open(blank_en_config_path, 'r') as f:
    blank_config = json.load(f)

# Merge blank_config into config, but keep existing fields if they exist and are important
# Actually, the error was specifically about vocab_size which is in blank_config
config.update(blank_config)

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Successfully updated {config_path} with fields from {blank_en_config_path}")
