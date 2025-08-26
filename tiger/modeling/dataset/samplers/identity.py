import copy

from dataset.samplers.base import EvalSampler, TrainSampler


class IdentityTrainSampler(TrainSampler, config_name="identity"):
    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(dataset=kwargs["dataset"])

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])
        return sample


class IdentityEvalSampler(EvalSampler, config_name="identity"):
    def __init__(self, dataset):
        self._dataset = dataset

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(dataset=kwargs["dataset"])

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])
        return sample
