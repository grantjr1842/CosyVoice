# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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
"""
Export Fun-CosyVoice3 model components to TorchScript for deployment.
"""

from __future__ import print_function

import argparse
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../..".format(ROOT_DIR))
sys.path.append("{}/../../third_party/Matcha-TTS".format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging


def get_args():
    parser = argparse.ArgumentParser(
        description="Export Fun-CosyVoice3 model for deployment"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Fun-CosyVoice3-0.5B",
        help="local path to model directory",
    )
    args = parser.parse_args()
    print(args)
    return args


def get_optimized_script(model, preserved_attrs=[]):
    script = torch.jit.script(model)
    if preserved_attrs != []:
        script = torch.jit.freeze(script, preserved_attrs=preserved_attrs)
    else:
        script = torch.jit.freeze(script)
    script = torch.jit.optimize_for_inference(script)
    return script


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s"
    )

    torch._C._jit_set_fusion_strategy([("STATIC", 1)])
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    model = AutoModel(model_dir=args.model_dir)

    # Fun-CosyVoice3 export: flow encoder only
    # Note: CosyVoice3 uses a different architecture than v1/v2

    # 1. export flow encoder
    flow_encoder = model.model.flow.encoder
    script = get_optimized_script(flow_encoder)
    script.save("{}/flow.encoder.fp32.zip".format(args.model_dir))
    script = get_optimized_script(flow_encoder.half())
    script.save("{}/flow.encoder.fp16.zip".format(args.model_dir))
    logging.info("successfully export flow_encoder")

    logging.info("JIT export complete for Fun-CosyVoice3")


if __name__ == "__main__":
    main()
