from dataset.samplers.base import TrainSampler, EvalSampler

import copy


class MCLSRTrainSampler(TrainSampler, config_name='mclsr'):

    def __init__(self, dataset, num_users, num_items):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids'][:-1]
        next_item = sample['item.ids'][-1]
        next_item_sequence = sample['item.ids'][1:]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'positive.ids': next_item_sequence,
            'positive.length': len(next_item_sequence),

            'labels.ids': [next_item],
            'labels.length': 1
        }


class MCLSRPredictionEvalSampler(EvalSampler, config_name='mclsr'):

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )
