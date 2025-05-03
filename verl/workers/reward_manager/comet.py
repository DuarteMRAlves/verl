from verl import DataProto

class COMETRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        pass

    def __call__(self, data: DataProto):
        if not "comet_rm" in data.batch.keys():
            raise ValueError("No comet_rm key found in the batch data.")
        return data.batch["comet_rm"]