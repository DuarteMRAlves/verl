import torch

from verl import DataProto
from verl.workers.reward_function.base import BaseRewardFunction

class SequentialRewardFunction(BaseRewardFunction):

    def __init__(self, config, compute_score):
        super().__init__(config)
        self.compute_score = compute_score

    def compute_scores(self, data: DataProto) -> tuple[torch.Tensor, dict[str, float]]:
        scores = []
        scored_idxs = []

        for i in range(len(data)):
            data_item = data[i]
            if data_item.non_tensor_batch["reward_model"]["style"] != "rule":
                continue

            answer_extraction = data_item.non_tensor_batch["answer_extraction"]
            if not answer_extraction.success:
                continue
            
            # Assumes the solution is the final answer.
            solution_str = answer_extraction.answer
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            score = self.compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
            scored_idxs.append(i)

        reward_tensor = torch.zeros((len(data.batch['responses']),), dtype=torch.float32)
        for score, score_idx in zip(scores, scored_idxs):
            reward_tensor[score_idx] = score

        metrics = {
            "reward_function/min": reward_tensor.min().item(),
            "reward_function/max": reward_tensor.max().item(),
            "reward_function/mean": reward_tensor.mean().item(),
        }

        return reward_tensor, metrics