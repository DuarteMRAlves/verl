# Inspired by https://raw.githubusercontent.com/fzp0424/MT-R1-Zero/refs/heads/main/verl/workers/comet/dp_comet.py

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
Implement a multiprocess PPOCritic
"""

import torch
import torch.distributed

from verl import DataProto
from verl.workers.comet import BaseCOMETModel

__all__ = ['DataParallelCOMET']


class DataParallelCOMET(BaseCOMETModel):

    def __init__(self, config, comet_module):
        super().__init__(config=config)
        self.comet_model = comet_module

    def compute_comet_rm(self, data: DataProto) -> tuple[torch.Tensor, dict[str, float]]:
        triplets = []
        scored_idxs = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            if data_item.non_tensor_batch["reward_model"]["style"] != "comet":
                continue

            answer_extraction = data_item.non_tensor_batch["answer_extraction"]
            if not answer_extraction.success:
                continue
            
            # Assumes the translation is the full answer.
            mt = answer_extraction.answer

            src_text = data_item.non_tensor_batch['reward_model']['src']
            tgt_text = data_item.non_tensor_batch['reward_model']['tgt']

            triplets.append({"src": src_text, "mt": mt, "ref": tgt_text})
            scored_idxs.append(i)

        reward_tensor = torch.zeros((len(data.batch['responses']),), dtype=torch.float32)
        if len(triplets) == 0:
            return reward_tensor, {}

        batch_size = self.config.micro_batch_size
        comet_output = self.comet_model.predict(triplets, batch_size=batch_size, gpus=1, progress_bar=False)
        scores = list(comet_output.scores)

        scored_idx_mask = torch.empty_like(reward_tensor, dtype=torch.bool)
        
        for score, score_idx in zip(scores, scored_idxs):
            reward_tensor[score_idx] = score
            scored_idx_mask[score_idx] = True

        metrics = {
            "comet/min": reward_tensor[scored_idx_mask].min().item(),
            "comet/max": reward_tensor[scored_idx_mask].max().item(),
            "comet/mean": reward_tensor[scored_idx_mask].mean().item(),
        }
        
        return reward_tensor, metrics
