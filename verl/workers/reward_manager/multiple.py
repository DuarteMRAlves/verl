import torch

from verl import DataProto

class MultipleRewardManager:
    def __init__(self,
        format_weight: float = 0.0,
        reward_model_weight: float = 1.0,
        comet_weight: float = 1.0,
    ) -> None:
        self.format_weight = format_weight
        self.reward_model_weight = reward_model_weight
        self.comet_weight = comet_weight

        if (
            self.format_weight == 0 and
            self.reward_model_weight == 0 and
            self.comet_weight == 0
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

        if len(weighted_scores) == 0:
            raise ValueError("No valid rewards found in the data batch.")

        if len(weighted_scores) == 1:
            # If only one reward is present, return it directly
            return weighted_scores[0]

        # Sum all weighted score tensors element-wise
        return torch.stack(weighted_scores, dim=0).sum(dim=0)