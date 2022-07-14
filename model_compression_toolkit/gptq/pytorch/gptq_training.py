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
from typing import Callable, List
from tqdm import tqdm
import copy
import torch
from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.gptq.common.gptq_training import GPTQTrainer
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.pytorch.constants import BIAS, KERNEL
from model_compression_toolkit.gptq.pytorch.quantizer.weights_quantizer import GPTQWeightQuantizer
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.gptq.pytorch.gptq_model_builder import GPTQPytorchModelBuilder
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, set_model, torch_tensor_to_numpy
from model_compression_toolkit.gptq.common.gptq_graph import get_compare_points
from model_compression_toolkit.gptq.pytorch.gptq_loss import multiple_tensors_mse_loss

MICRO_BIAS_UPDATE_FACTOR = 0.2  # A factor that reduce the number of iteration for correction the bias vectors.


class PytorchGPTQTrainer(GPTQTrainer):
    """
    Pytorch GPTQ training class for fine-tuning a quantized model
    """

    def __init__(self,
                 graph_float: Graph,
                 graph_quant: Graph,
                 gptq_config: GradientPTQConfig,
                 fw_impl: FrameworkImplementation,
                 fw_info: FrameworkInfo):
        """
        Build two models from a graph: A teacher network (float model) and a student network (quantized model).
        Use the dataset generator to pass images through the teacher and student networks to get intermediate
        layers outputs. Use the outputs to compute the observed loss and to back-propagate the error
        in the student network, to minimize it in the next similar steps.
        All parameters (such as number of iterations, optimizer, etc.) are in GradientPTQConfig.
        Args:
            graph_float: Graph to build a float networks from.
            graph_quant: Graph to build a quantized networks from.
            gptq_config: GradientPTQConfig with parameters about the tuning process.
            fw_impl: FrameworkImplementation object with a specific framework methods implementation.
            fw_info: Framework information
        """
        super().__init__(graph_float, graph_quant, gptq_config, fw_impl, fw_info)

    def build_gptq_model(self):
        """
        Build the GPTQ model with QuantizationWrappers
        Returns:
            Quantized graph for GPTQ fine-tuning, GPTQ graph user info
        """
        self.compare_points, _, self.compare_points_mean, self.compare_points_std = get_compare_points(self.graph_quant)
        return GPTQPytorchModelBuilder(self.graph_quant,
                                       self.gptq_config,
                                       fw_info=self.fw_info,
                                       append2output=self.compare_points)

    def train(self, representative_data_gen: Callable):
        """
          GPTQ Training using pytorch framework
          Args:
              representative_data_gen: Dataset generator to get images.
          Returns:
              Graph after GPTQ training
          """
        # ----------------------------------------------
        # Set Optimizer
        # ----------------------------------------------
        # Optimizer
        self._set_trainable_parameters()
        optimizer = self.gptq_config.optimizer
        optimizer.param_groups.clear()
        optimizer.add_param_group({'params': self.fxp_model.parameters()})

        # Set models mode
        set_model(self.float_model)
        set_model(self.fxp_model)

        def update_step(x: List[torch.Tensor]) -> torch.Tensor:
            # Forward-pass
            y_float = self.float_model(x)
            y_fxp = self.fxp_model(x)
            # Loss function
            loss_value = self.gptq_config.loss(y_fxp, y_float, [], [], [], [])
            # Back-pass
            optimizer.zero_grad()
            loss_value.backward()
            # Update parameters
            optimizer.step()
            return loss_value

        # ----------------------------------------------
        # Training loop
        # ----------------------------------------------
        self.loss_list = []
        for _ in tqdm(range(self.gptq_config.n_iter)):
            input_tensor = representative_data_gen()
            torch_tensors = to_torch_tensor(input_tensor)
            loss_value = update_step(torch_tensors)
            self.loss_list.append(loss_value.item())
            if self.gptq_config.log_function is not None:
                variables, grads = [], []
                for param in self.fxp_model.parameters():
                    if param.requires_grad:
                        variables.append(torch_tensor_to_numpy(param))
                        grads.append(torch_tensor_to_numpy(param.grad))
                self.gptq_config.log_function(self.loss_list[-1], variables, grads)
            Logger.debug(f'last loss value: {self.loss_list[-1]}')

    def update_graph(self):
        graph_quant = copy.copy(self.graph_quant)

        # Update graph after training
        for name, layer in self.fxp_model.named_modules():
            node = self.graph_quant.find_node_by_name(name)
            if len(node) > 0 and isinstance(layer, GPTQWeightQuantizer):
                node = node[0]
                weight_attrs = self.fw_info.get_kernel_op_attributes(node.type)
                for weight_attr in weight_attrs:
                    if weight_attr is not None:
                        node.set_weights_by_keys(weight_attr, self.fw_impl.to_numpy(getattr(layer.op, weight_attr)))
                if self.gptq_config.train_bias:
                    node.set_weights_by_keys(BIAS, self.fw_impl.to_numpy(getattr(layer.op, BIAS)))

        # Save losses collected during training
        graph_quant.user_info.gptq_info_dict['loss'] = self.loss_list
        return graph_quant

    def _set_trainable_parameters(self):
        """
        Get trainable parameters for GPTQ training
        Returns:
            trainable parameters
        """
        # Float and Fxp models: freeze all the parameters in the network
        for param in self.float_model.parameters():
            param.requires_grad = False
        for param in self.fxp_model.parameters():
            param.requires_grad = False

        # Fxp model: unfreeze only trainable parameters from GPTQWeightQuantizer
        for name, layer in self.fxp_model.named_modules():
            if isinstance(layer, GPTQWeightQuantizer):
                for param in layer.get_trainable_params():
                    param.requires_grad = True
                if self.gptq_config.train_bias and hasattr(layer.op, BIAS):
                    bias = getattr(layer.op, BIAS)
                    bias.requires_grad = True
