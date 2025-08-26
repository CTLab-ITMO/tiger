from dataset.samplers.base import TrainSampler, EvalSampler

import copy
import numpy as np


class MaskedItemPredictionTrainSampler(TrainSampler, config_name='masked_item_prediction'):

    def __init__(self, dataset, num_users, num_items, mask_prob=0.0):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._mask_item_idx = self._num_items + 1
        self._mask_prob = mask_prob

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            mask_prob=config.get('mask_prob', 0.0)
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids']

        masked_sequence = []
        labels = []

        for item in item_sequence:
            prob = np.random.uniform(low=0.0, high=1.0)

            if prob < self._mask_prob:
                prob /= self._mask_prob

                if prob < 0.8:
                    masked_sequence.append(self._mask_item_idx)
                elif prob < 0.9:
                    masked_sequence.append(np.random.randint(1, self._num_items + 1))
                else:
                    masked_sequence.append(item)

                labels.append(item)
            else:
                masked_sequence.append(item)
                labels.append(0)

        # Mask last item
        masked_sequence[-1] = self._mask_item_idx
        labels[-1] = item_sequence[-1]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': masked_sequence,
            'item.length': len(masked_sequence),

            'labels.ids': labels,
            'labels.length': len(labels)
        }


class MaskedItemPredictionEvalSampler(EvalSampler, config_name='masked_item_prediction'):

    def __init__(self, dataset, num_users, num_items):
        super().__init__(dataset, num_users, num_items)
        self._mask_item_idx = self._num_items + 1

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])
        item_sequence = sample['item.ids']
        labels = [item_sequence[-1]]
        sequence = item_sequence[:-1] + [self._mask_item_idx]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': sequence,
            'item.length': len(sequence),

            'labels.ids': labels,
            'labels.length': len(labels)
        }
