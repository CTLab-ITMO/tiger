from collections import defaultdict

from utils import MetaParent


class BaseNegativeSampler(metaclass=MetaParent):

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
