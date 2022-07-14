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
import torch
import torch.nn as nn
from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.core.common.constants import THRESHOLD, RANGE_MAX, RANGE_MIN
from model_compression_toolkit.core.common.target_platform.op_quantization_config import QuantizationMethod

class GPTQWeightQuantizer(nn.Module):

    def __init__(self, node: BaseNode, fw_info: FrameworkInfo, gptq_config: GradientPTQConfig):
        """
        Construct a Pytorch model that constitutes as a wrapper for a Pytorch layer, built from a given graph node.
        Args:
            node: Node to build its Pytorch layer.
            fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
        """
        super().__init__()
        self.op = node.type(**node.framework_attr)

        # Save attributes
        self.weight_attrs = fw_info.get_kernel_op_attributes(node.type)

        # Save GPTQ configuration
        self.save_qconfig(node, gptq_config)

    def save_qconfig(self, node: BaseNode, gptq_config: GradientPTQConfig):
        """
        Save quantization configuration for later use of the weights quantizer
        Args:
            node: Node to read quantization configuration from.
            gptq_config: GPTQ quantization configuration
        """
        self.signed = True
        self.num_bits = node.final_weights_quantization_cfg.weights_n_bits
        self.max_delta_change = gptq_config.lsb_change_per_bit_width.get(self.num_bits)
        self.min_int = -int(self.signed) * (2 ** (self.num_bits - int(self.signed)))
        self.max_int = (2 ** (self.num_bits - int(self.signed))) - 1
        if node.final_weights_quantization_cfg.weights_quantization_method == QuantizationMethod.UNIFORM:
            # Uniform quantization
            self.min_range = node.final_weights_quantization_cfg.weights_quantization_params.get(RANGE_MIN)
            self.max_range = node.final_weights_quantization_cfg.weights_quantization_params.get(RANGE_MAX)
        else:
            # Symmetric/PowerOf2 quantization
            threshold_values = node.final_weights_quantization_cfg.weights_quantization_params.get(THRESHOLD)
            self.min_range = -threshold_values
            self.max_range = threshold_values
        self.delta_tensor = (self.max_range-self.min_range) / (2 ** self.num_bits)