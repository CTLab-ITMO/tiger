import json
import logging

from modeling.dataset.samplers import TrainSampler, EvalSampler

LOGGER = logging.getLogger(__name__)


class Dataset:
    def __init__(
            self,
            train_sampler,
            validation_sampler,
            test_sampler,
            num_items,
            max_sequence_length
    ):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @classmethod
    def create(cls, inter_json_path, max_sequence_length, sampler_type, is_extended=False):
        max_item_id = 0
        train_dataset, validation_dataset, test_dataset = [], [], []

        with open(inter_json_path, 'r') as f:
            user_interactions = json.load(f)

        for user_id_str, item_ids in user_interactions.items():
            user_id = int(user_id_str)

            if item_ids:
                max_item_id = max(max_item_id, max(item_ids))

            assert len(item_ids) >= 5, f'Core-5 dataset is used, user {user_id} has only {len(item_ids)} items'

            # sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (9 - valid, 10 - test)
            if is_extended:
                # sample = [1, 2, 3, 4, 5]
                # sample = [1, 2, 3, 4, 5, 6]
                # sample = [1, 2, 3, 4, 5, 6, 7]
                # sample = [1, 2, 3, 4, 5, 6, 7, 8]
                for prefix_length in range(5 - 2 + 1, len(item_ids) - 2 + 1):
                    train_dataset.append({
                        'user.ids': [user_id],
                        'user.length': 1,
                        'item.ids': item_ids[:prefix_length][-max_sequence_length:],
                        'item.length': len(item_ids[:prefix_length][-max_sequence_length:]),
                    })
            else:
                # sample = [1, 2, 3, 4, 5, 6, 7, 8]
                train_dataset.append({
                    'user.ids': [user_id],
                    'user.length': 1,
                    'item.ids': item_ids[:-2][-max_sequence_length:],
                    'item.length': len(item_ids[:-2][-max_sequence_length:])
                })

            # sample = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            validation_dataset.append({
                'user.ids': [user_id],
                'user.length': 1,
                'item.ids': item_ids[:-1][-max_sequence_length:],
                'item.length': len(item_ids[:-1][-max_sequence_length:])
            })

            # sample = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            test_dataset.append({
                'user.ids': [user_id],
                'user.length': 1,
                'item.ids': item_ids[-max_sequence_length:],
                'item.length': len(item_ids[-max_sequence_length:])
            })

        LOGGER.info(f'Train dataset size: {len(train_dataset)}')
        LOGGER.info(f'Validation dataset size: {len(validation_dataset)}')
        LOGGER.info(f'Test dataset size: {len(test_dataset)}')
        LOGGER.info(f'Max item id: {max_item_id}')

        train_sampler = TrainSampler(train_dataset, sampler_type)
        validation_sampler = EvalSampler(validation_dataset)
        test_sampler = EvalSampler(test_dataset)

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_items=max_item_id + 1,  # +1 added because our ids are 0-indexed
            max_sequence_length=max_sequence_length
        )

    def get_samplers(self):
        return self._train_sampler, self._validation_sampler, self._test_sampler

    @property
    def num_items(self):
        return self._num_items

    @property
    def max_sequence_length(self):
        return self._max_sequence_length
