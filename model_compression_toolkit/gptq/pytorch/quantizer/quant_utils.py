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

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """
    Return the rounded values of a tensor.
    """
    return torch.round(x).detach() - x.detach() + x


def ste_clip(x: torch.Tensor, min_val=-1.0, max_val=1.0) -> torch.Tensor:
    """
    clip a variable between fixed values such that min_val<=output<=max_val
    Args:
        x: input variable
        max_val: maximum value for clipping
        min_val: minimum value for clipping (defaults to -max_val)

    Returns:
        clipped variable

    """
    return torch.clip(x, min=min_val, max=max_val).detach() - x.detach() + x