# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig, RoundingType
from model_compression_toolkit.gptq.pytorch.quantizer.ste_rounding.ste_weights_quantizer import STEWeightQuantizer

def weights_quantizer_node_builder(node: BaseNode, fw_info: FrameworkInfo, gptq_config: GradientPTQConfig):

    if gptq_config.rounding_type == RoundingType.STE:
        return STEWeightQuantizer(node, fw_info, gptq_config)
    else:
        assert False, "fdfd"

