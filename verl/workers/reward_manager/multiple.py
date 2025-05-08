import torch

from collections import defaultdict
from verl import DataProto

class MultipleRewardManager:
    def __init__(self,
        tokenizer,
        num_examine: int,
        format_weight: float = 0.0,
        reward_model_weight: float = 1.0,
        comet_weight: float = 1.0,
        reward_function_weight: float = 1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_weight = format_weight
        self.reward_model_weight = reward_model_weight
        self.comet_weight = comet_weight
        self.reward_function_weight = reward_function_weight

        if (
            self.format_weight == 0 and
            self.reward_model_weight == 0 and
            self.comet_weight == 0 and
            self.reward_function_weight == 0
        ):
            raise ValueError("At least one reward weight must be non-zero.")

    def __call__(self, data: DataProto):
        weighted_scores = []

        if self.format_weight != 0 and "format_reward" in data.batch:
            weighted_scores.append(data.batch["format_reward"] * self.format_weight)
        if self.reward_model_weight != 0 and "rm_scores" in data.batch:
            weighted_scores.append(data.batch["rm_scores"] * self.reward_model_weight)
        if self.comet_weight != 0 and "comet_rm" in data.batch:
            weighted_scores.append(data.batch["comet_rm"] * self.comet_weight)
        if self.reward_function_weight != 0 and "reward_function" in data.batch:
            weighted_scores.append(data.batch["rf_scores"] * self.reward_function_weight)

        if len(weighted_scores) == 0:
            raise ValueError("No valid rewards found in the data batch.")

        if len(weighted_scores) == 1:
            # If only one reward is present, return it directly
            scores = weighted_scores[0]
        else:
            # Sum all weighted score tensors element-wise
            scores = torch.stack(weighted_scores, dim=0).sum(dim=0)

        if self.num_examine > 0:
            self._print_data_sources(data, scores)

        return scores
    

    def _print_data_sources(self, data, scores):
        datasources_counts = defaultdict(int)
        item_scores = scores.sum(dim=-1)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            data_source = data_item.non_tensor_batch['data_source']

            if datasources_counts[data_source] > self.num_examine:
                continue
            datasources_counts[data_source] += 1

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            print("[prompt]", prompt_str)
            print("[response]", response_str)
            for k, v in data_item.non_tensor_batch["reward_model"].items():
                print(f"[reward model - {k}]", v)
            if "answer_extraction" in data_item.non_tensor_batch:
                answer_extraction = data_item.non_tensor_batch['answer_extraction']
                print("[answer extraction success]", answer_extraction.success)
                print("[answer extraction thinking]", answer_extraction.thinking)
                print("[answer extraction answer]", answer_extraction.answer)
            if "format_reward" in data_item.batch:
                print("[format reward]", data_item.batch["format_reward"].sum().item())
            if "rm_scores" in data_item.batch:
                print("[rm score]", data_item.batch["rm_scores"].sum().item())
            if "comet_rm" in data_item.batch:
                print("[comet rm]", data_item.batch["comet_rm"].sum().item())
            if "rf_scores" in data_item.batch:
                print("[rf score]", data_item.batch["rf_scores"].sum().item())
            print("[final score]", item_scores[i].item(), flush=True)