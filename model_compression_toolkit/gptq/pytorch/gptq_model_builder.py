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
from typing import Tuple, Any, List
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import BaseNode
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.pytorch.back2framework.instance_builder import node_builder
from model_compression_toolkit.gptq.pytorch.quantizer.weights_quantizer_builder import weights_quantizer_node_builder



class GPTQPytorchModelBuilder(PyTorchModelBuilder):
    """
    Builder of GPTQ Pytorch models.
    """

    def __init__(self,
                 graph: common.Graph,
                 gptq_config: GradientPTQConfig,
                 append2output=None,
                 fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO,
                 return_float_outputs: bool = True):
        """

        Args:
            graph: Graph to build the model from.
            gptq_config: Configuration for GPTQ optimization.
            append2output: Nodes to append to model's output.
            fw_info: Information about the specific framework of the model that is built.
            return_float_outputs: Whether the model returns float tensors or not.
        """

        super().__init__(graph,
                         append2output,
                         fw_info,
                         return_float_outputs)
        self.gptq_config = gptq_config

    def _quantize_node_activations(self,
                                   node: BaseNode,
                                   input_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Quantize node's activation given input tensors.

        Args:
            node: Node to quantize its outputs.
            input_tensors: Input tensors of the node.

        Returns:
            Output of the node.

        """

        return node.final_activation_quantization_cfg.quantize_node_output(input_tensors)

    def build_model(self) -> Tuple[nn.Module, UserInformation]:
        """
        Build a Keras GPTQ model and return it.
        Returns: GPTQ Keras model.

        """
        model, user_info = super().build_model()

        for node in self.graph.get_topo_sorted_nodes():
            if not isinstance(node, FunctionalNode):
                if node.is_weights_quantization_enabled():
                    model.add_module(node.name, weights_quantizer_node_builder(node, fw_info, self.gptq_config))
                else:
                    model.add_module(node.name, node_builder(node))

        return model, user_info









def model_builder(graph: common.Graph,
                  gptq_config: GradientPTQConfig,
                  mode: ModelBuilderMode = ModelBuilderMode.QUANTIZED,
                  append2output=None,
                  fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO) -> Tuple[nn.Module, Any]:
    """
    Build a Pytorch model for GPTQ from a graph representing the model.
    The model is built by converting the graph nodes to Keras layers and applying them sequentially to get the model
    output tensors. The output tensors list and an input tensors list, then use to build the model.
    After the model is built in Pytorch, it is cloned to add quantization wrappers for GPTQ fine-tuning

    Args:
        graph: Graph to build its corresponding Keras model.
        gptq_config: GPTQ Configuration class.
        mode: Building mode. Read ModelBuilderMode description for more info.
        append2output: List of nodes or OutTensor objects. In float building mode,
        when the list contains nodes, all output tensors of all nodes are set as the model outputs.
        fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
        This is for passing the kernel attributes to the QuanteWrappers.

    Returns:
        A tuple of the model and an UserInformation object.
    """
    if gptq_config is None:
        Logger.exception("Building a model in GPTQ requires a GPTQ configuration as input")

    model, graph_user_info = core_model_builder(graph, mode, append2output, fw_info)

    for node in graph.get_topo_sorted_nodes():
        if not isinstance(node, FunctionalNode):
            if node.is_weights_quantization_enabled():
                model.add_module(node.name, weights_quantizer_node_builder(node, fw_info, gptq_config))
            else:
                model.add_module(node.name, node_builder(node))

    return model, graph_user_info
