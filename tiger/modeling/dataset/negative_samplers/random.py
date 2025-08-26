from collections import defaultdict

from tqdm import tqdm

from dataset.negative_samplers.base import BaseNegativeSampler

import numpy as np


class RandomNegativeSampler(BaseNegativeSampler, config_name='random'):

    @classmethod
    def create_from_config(cls, _, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )

    def generate_negative_samples(self, sample, num_negatives):
        user_id = sample['user.ids'][0]
        all_items = list(range(1, self._num_items + 1))
        np.random.shuffle(all_items)

        negatives = []
        running_idx = 0
        while len(negatives) < num_negatives and running_idx < len(all_items):
            negative_idx = all_items[running_idx]
            if negative_idx not in self._seen_items[user_id]:
                negatives.append(negative_idx)
            running_idx += 1

        return negatives
