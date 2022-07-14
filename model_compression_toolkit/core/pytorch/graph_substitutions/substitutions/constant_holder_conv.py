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
from torch.nn import Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d
from torch.nn.functional import conv1d, conv2d, conv_transpose1d, conv_transpose2d
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, WalkMatcher, EdgeMatcher
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.pytorch.reader.graph_builders import ConstantHolder
from model_compression_toolkit.core.pytorch.constants import IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, KERNEL, CONSTANT
from model_compression_toolkit.core.pytorch.utils import torch_tensor_to_numpy


class ConstantHolderConv(common.BaseSubstitution):
    """
    Find "permute" node to substitute new dimension argument if needed
    """

    def __init__(self):
        """
        Matches: 'ConstantHolder' followed conv layer
        """
        first_node = NodeOperationMatcher(ConstantHolder)
        second_node = NodeOperationMatcher(conv2d) | NodeOperationMatcher(conv1d) | NodeOperationMatcher(conv_transpose1d) | NodeOperationMatcher(conv_transpose2d)
        super().__init__(matcher_instance=EdgeMatcher(first_node, second_node))

    def substitute(self,
                   graph: Graph,
                   nodes: BaseNode) -> Graph:
        """
        Wrap dimension of permute with tuple if it's missing

        Args:
            graph: Graph we apply the substitution on.
            nodes: nodes that match the pattern in the substitution init.

        Returns:
            Graph after applying the substitution.
        """
        first_node = nodes[0] # constant holder node
        second_node = nodes[1] # convolution node

        # Set new
        # TODO use fw_info
        if second_node.type == conv2d:
            NewLayer = Conv2d
            out_channel_index = 1
        elif second_node.type == conv_transpose2d:
            NewLayer = ConvTranspose2d
            out_channel_index = 0
        elif second_node.type == conv1d:
            NewLayer = Conv1d
            out_channel_index = 1
        elif second_node.type == conv_transpose1d:
            NewLayer = ConvTranspose1d
            out_channel_index = 0
        else:
            return graph # skip substitution

        # Create new node of layer convolution
        weights = first_node.get_weights_by_keys(CONSTANT)
        framework_attr = second_node.framework_attr
        framework_attr.update({OUT_CHANNELS: weights.shape[1-out_channel_index]})
        framework_attr.update({IN_CHANNELS: weights.shape[out_channel_index]})
        framework_attr.update({KERNEL_SIZE: weights.shape[2:]})

        new_node = BaseNode(name=second_node.name,
                            framework_attr=framework_attr,
                            input_shape=second_node.input_shape,
                            output_shape=second_node.output_shape,
                            weights={KERNEL: weights},
                            layer_class=NewLayer,
                            has_activation=second_node.has_activation)
        graph.add_node(new_node)
        graph.remove_edge(first_node, second_node)
        graph.remove_node(first_node)
        graph.reconnect_out_edges(current_node=second_node, new_node=new_node)
        graph.reconnect_in_edges(current_node=second_node, new_node=new_node)
        graph.replace_output_node(current_node=second_node, new_node=new_node)
        graph.remove_node(second_node)

        return graph

