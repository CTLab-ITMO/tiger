from dataset.samplers.base import TrainSampler, EvalSampler

import copy


class LastItemPredictionTrainSampler(TrainSampler, config_name='last_item_prediction'):

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
            num_items=kwargs['num_items'],
        )

    def __getitem__(self, index):
        sample = self._dataset[index]

        item_sequence = sample['item.ids'][:-1]
        last_item = sample['item.ids'][-1]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'labels.ids': [last_item],
            'labels.length': 1,
        }


class LastItemPredictionEvalSampler(EvalSampler, config_name='last_item_prediction'):

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )
