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
from model_compression_toolkit.gptq.pytorch.quantizer.weights_quantizer import GPTQWeightQuantizer
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.gptq.pytorch.quantizer.quant_utils import ste_round, ste_clip

class STEWeightQuantizer(GPTQWeightQuantizer):
    """
    Class that wraps a Pytorch layer (nn.Module) with weights to be used for GPTQ training.
    """

    def __init__(self, node: BaseNode, fw_info: FrameworkInfo, gptq_config: GradientPTQConfig):
        """
        Construct a Pytorch model that constitutes as a wrapper for a Pytorch layer, built from a given graph node.
        Args:
            n: Node to build its Pytorch layer.
            fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
        """
        nn.Module.__init__(self)
        GPTQWeightQuantizer.__init__(self, node, fw_info, gptq_config)

        # loading the weights from the graph node (weights of the trained model)
        self.op.load_state_dict({k: torch.Tensor(v) for k, v in node.weights.items()}, strict=False)

        # Save trainable tensors
        self.set_trainable_params()

        # Create tensors
        self.delta_tensor = to_torch_tensor(self.delta_tensor)
        self.max_tensor_change = self.delta_tensor * self.max_delta_change

    def set_trainable_params(self):
        self.trainable_params = {}
        if len(self.weight_attrs) > 0:
            weight_attr = self.weight_attrs[0]
            self.weight_float = to_torch_tensor(torch.Tensor(getattr(self.op, weight_attr)))
            self.aux_tensor = nn.Parameter(torch.zeros_like(self.weight_float))
            self.trainable_params.update({"aux_tensor": self.aux_tensor})
            # Change layer weight from nn.Parameter to torch.Tensor
            del self.op.weight
            self.op.weight = self.weight_float

    def get_trainable_params(self):
        """
        A function to get a list trainable of trainable parameters of the quantizer for GPTQ retraining
        Returns:
            A list of trainable tensors
        """
        trainable_params = [value for value in self.trainable_params.values()]
        return trainable_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed-Forward function
        Args:
            x: input tensors to layer.
        Returns:
            torch Tensor which is the output of the wrapped layer on the given input.
        """
        # Weight fake quantization
        if len(self.weight_attrs) > 0:
            weight_quantized = self.quantize(self.weight_float, self.aux_tensor)
            self.op.weight = weight_quantized

        # Do computation
        return self.op(x)

    def quantize(self, w: torch.Tensor, v: nn.Parameter) -> torch.Tensor:
        """
        Weight fake quantizer
        Args:
            w: weights to quantize.
            v: auxiliary tensor for training the possible new delta step for the weights' quantization.
        Returns:
            quantized weights
        """
        v0 = ste_clip(v, min_val=-self.max_tensor_change, max_val=self.max_tensor_change)
        v1 = v0 / self.delta_tensor
        w0 = torch.round(w / self.delta_tensor).detach()
        w1 = w0 + v1
        w2 = ste_round(w1)
        w3 = ste_clip(w2, min_val=self.min_int, max_val=self.max_int)
        w_q = self.delta_tensor * w3
        return w_q

