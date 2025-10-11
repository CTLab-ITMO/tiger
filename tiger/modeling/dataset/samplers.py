import copy


class TrainSampler:
    def __init__(self, dataset, prediction_type):
        self._dataset = dataset
        self._prediction_type = prediction_type

        self._transforms = {
            'last_item': self._last_item_transform,
            'next_item': self._next_item_transform,
        }

    @staticmethod
    def _last_item_transform(sample):
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

    @staticmethod
    def _next_item_transform(sample):
        item_sequence = sample['item.ids'][:-1]
        next_item_sequence = sample['item.ids'][1:]
        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'labels.ids': next_item_sequence,
            'labels.length': len(next_item_sequence)
        }

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])
        return self._transforms[self._prediction_type](sample)

    def __len__(self):
        return len(self._dataset)


class EvalSampler:
    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = self._dataset[index]

        item_sequence = sample['item.ids'][:-1]
        next_item = sample['item.ids'][-1]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'labels.ids': [next_item],
            'labels.length': 1
        }
