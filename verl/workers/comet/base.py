# Copied from https://raw.githubusercontent.com/fzp0424/MT-R1-Zero/refs/heads/main/verl/workers/comet/base.py

# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
The base class for reward model
"""

import torch

from abc import ABC, abstractmethod

from verl import DataProto


class BaseCOMETModel(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def compute_comet_rm(self, data: DataProto) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Computing reward given input_ids. The transformers should output a tensor with shape
           [batch_size, sequence_length], and the value at [EOS] mask should be gathered.

        Args:
            data: must contain keys "input_ids", "attention_mask" and "position_ids".
                - input_ids: [batch_size, sequence_length]
                - attention_mask: [batch_size, sequence_length]
                - position_ids: [batch_size, sequence_length]

        Returns: comet scores for the input data and metrics to log.

        """
        pass