import json
import logging

from .samplers import TrainSampler, EvalSampler

logger = logging.getLogger(__name__)


class ScientificDataset:
    def __init__(
            self,
            train_sampler,
            validation_sampler,
            test_sampler,
            num_users,
            num_items,
            max_sequence_length
    ):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @classmethod
    def create(cls, inter_json_path, max_sequence_length, sampler_type, mode="simple"):
        max_user_id, max_item_id = 0, 0
        train_dataset, validation_dataset, test_dataset = [], [], []

        with open(inter_json_path, 'r') as f:
            user_interactions = json.load(f)

        for user_id_str, item_ids in user_interactions.items():
            user_id = int(user_id_str)

            max_user_id = max(max_user_id, user_id)
            if item_ids:
                max_item_id = max(max_item_id, max(item_ids))

            assert len(item_ids) >= 5, f'Core-5 dataset required, user {user_id} has {len(item_ids)} items'

            if mode == "full":
                for prefix_length in range(5 - 2 + 1, len(item_ids) - 2 + 1):
                    # prefix = [1, 2, 3, 4, 5]
                    # prefix = [1, 2, 3, 4, 5, 6]
                    # prefix = [1, 2, 3, 4, 5, 6, 7]
                    # prefix = [1, 2, 3, 4, 5, 6, 7, 8]
                    train_dataset.append({
                        'user.ids': [user_id],
                        'user.length': 1,
                        'item.ids': item_ids[:prefix_length][-max_sequence_length:],
                        'item.length': len(item_ids[:prefix_length][-max_sequence_length:]),
                    })
            else:
                train_dataset.append({
                    'user.ids': [user_id],
                    'user.length': 1,
                    'item.ids': item_ids[:-2][-max_sequence_length:],
                    'item.length': len(item_ids[:-2][-max_sequence_length:])
                })
            assert len(item_ids[:-2][-max_sequence_length:]) == len(set(item_ids[:-2][-max_sequence_length:]))

            validation_dataset.append({
                'user.ids': [user_id],
                'user.length': 1,
                'item.ids': item_ids[:-1][-max_sequence_length:],
                'item.length': len(item_ids[:-1][-max_sequence_length:])
            })
            assert len(item_ids[:-1][-max_sequence_length:]) == len(set(item_ids[:-1][-max_sequence_length:]))

            test_dataset.append({
                'user.ids': [user_id],
                'user.length': 1,
                'item.ids': item_ids[-max_sequence_length:],
                'item.length': len(item_ids[-max_sequence_length:])
            })
            assert len(item_ids[-max_sequence_length:]) == len(set(item_ids[-max_sequence_length:]))

        logger.info('Train dataset size: {}'.format(len(train_dataset)))
        logger.info('Validation dataset size: {}'.format(len(validation_dataset)))
        logger.info('Test dataset size: {}'.format(len(test_dataset)))
        logger.info('Max user id: {}'.format(max_user_id))
        logger.info('Max item id: {}'.format(max_item_id))
        logger.info('Max sequence length: {}'.format(max_sequence_length))
        logger.info('Dataset sparsity: {}'.format(
            (len(train_dataset) + len(test_dataset)) / (max_user_id + 1) / (max_item_id + 1)
        ))

        train_sampler = TrainSampler(train_dataset, sampler_type)
        validation_sampler = EvalSampler(validation_dataset)
        test_sampler = EvalSampler(test_dataset)

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_users=max_user_id + 1,
            num_items=max_item_id + 1,
            max_sequence_length=max_sequence_length
        )

    def get_samplers(self):
        return self._train_sampler, self._validation_sampler, self._test_sampler

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def max_sequence_length(self):
        return self._max_sequence_length
