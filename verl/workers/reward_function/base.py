import torch

from abc import ABC, abstractmethod

from verl import DataProto


class BaseRewardFunction(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def compute_scores(data: DataProto) -> torch.Tensor:
        """Computing reward given input_ids.

        Args:
            data: must contain keys "input_ids", "attention_mask" and "position_ids".
                - input_ids: [batch_size, sequence_length]
                - attention_mask: [batch_size, sequence_length]
                - position_ids: [batch_size, sequence_length]

        Returns: scores for the input data.
        """
        pass