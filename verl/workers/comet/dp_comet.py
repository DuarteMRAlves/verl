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

    def __init__(
        self,
        config,
        comet_module,
        tokenizer,
        answer_extractor,
    ):
        super().__init__(config=config)
        self.comet_model = comet_module
        self.tokenizer = tokenizer
        self.answer_extractor = answer_extractor

    def compute_comet_rm(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros((len(data.batch['responses']),), dtype=torch.float32)
        triplets = []
        scored_idxs = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            answer_extraction = self.answer_extractor(response_str)

            if not answer_extraction.success:
                continue
            
            # Assumes the translation is the full answer.
            mt = answer_extraction.answer

            src_text = data_item.non_tensor_batch['reward_model']['src']
            tgt_text = data_item.non_tensor_batch['reward_model']['tgt']

            triplets.append({"src": src_text, "mt": mt, "ref": tgt_text})
            scored_idxs.append(i)

        batch_size = self.config.micro_batch_size
        comet_output = self.comet_model.predict(triplets, batch_size=batch_size, gpus=1, progress_bar=False)
        scores = list(comet_output.scores)

        for score, score_idx in zip(scores, scored_idxs):
            reward_tensor[score_idx] = score

        return reward_tensor
