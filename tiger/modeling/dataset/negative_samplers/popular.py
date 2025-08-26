from dataset.negative_samplers.base import BaseNegativeSampler

from collections import Counter


class PopularNegativeSampler(BaseNegativeSampler, config_name='popular'):

    def __init__(
            self,
            dataset,
            num_users,
            num_items
    ):
        super().__init__(
            dataset=dataset,
            num_users=num_users,
            num_items=num_items
        )

        self._popular_items = self._items_by_popularity()

    @classmethod
    def create_from_config(cls, _, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )

    def _items_by_popularity(self):
        popularity = Counter()

        for sample in self._dataset:
            for item_id in sample['item.ids']:
                popularity[item_id] += 1

        popular_items = sorted(popularity, key=popularity.get, reverse=True)
        return popular_items

    def generate_negative_samples(self, sample, num_negatives):
        user_id = sample['user.ids'][0]
        popularity_idx = 0
        negatives = []
        while len(negatives) < num_negatives:
            negative_idx = self._popular_items[popularity_idx]
            if negative_idx not in self._seen_items[user_id]:
                negatives.append(negative_idx)
            popularity_idx += 1

        return negatives
