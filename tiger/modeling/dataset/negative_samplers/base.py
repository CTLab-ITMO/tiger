from collections import defaultdict, Counter
import numpy as np


class BaseNegativeSampler:

    def __init__(
            self,
            dataset,
            num_users,
            num_items
    ):
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items

        self._seen_items = defaultdict(set)
        for sample in self._dataset:
            user_id = sample['user.ids'][0]
            items = list(sample['item.ids'])
            self._seen_items[user_id].update(items)

    def generate_negative_samples(self, sample, num_negatives):
        raise NotImplementedError


class RandomNegativeSampler(BaseNegativeSampler):
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