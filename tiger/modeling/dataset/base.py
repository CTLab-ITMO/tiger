import json
import logging
import os

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
    def create_from_config(cls, config):
        data_dir_path = os.path.join(config['path_to_data_dir'], config['name'])
        max_sequence_length = config['max_sequence_length']
        max_user_id, max_item_id = 0, 0
        train_dataset, validation_dataset, test_dataset = [], [], []

        dataset_path = os.path.join(data_dir_path, '{}.txt'.format('all_data'))
        with open(dataset_path, 'r') as f:
            data = f.readlines()

        for sample in data:
            sample = sample.strip('\n').split(' ')
            user_id = int(sample[0])
            item_ids = [int(item_id) for item_id in sample[1:]]

            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, max(item_ids))

            assert len(item_ids) >= 5

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
        logger.info('Test dataset size: {}'.format(len(test_dataset)))
        logger.info('Max user id: {}'.format(max_user_id))
        logger.info('Max item id: {}'.format(max_item_id))
        logger.info('Max sequence length: {}'.format(max_sequence_length))
        logger.info('{} dataset sparsity: {}'.format(
            config['name'], (len(train_dataset) + len(test_dataset)) / max_user_id / max_item_id
        ))

        train_sampler = TrainSampler(train_dataset, config['samplers']['type'])
        validation_sampler = EvalSampler(validation_dataset)
        test_sampler = EvalSampler(test_dataset)

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_users=max_user_id,
            num_items=max_item_id,
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


class ScientificFullDataset(ScientificDataset):
    @classmethod
    def create_from_config(cls, config, file_name="all_data"):
        data_dir_path = os.path.join(config["path_to_data_dir"], config["name"])
        max_sequence_length = config["max_sequence_length"]
        max_user_id, max_item_id = 0, 0
        train_dataset, validation_dataset, test_dataset = [], [], []

        dataset_path = os.path.join(data_dir_path, "{}.txt".format(file_name))
        with open(dataset_path, "r") as f:
            data = f.readlines()

            for sample in data:
                sample = sample.strip("\n").split(" ")
                user_id = int(sample[0])
                item_ids = [int(item_id) for item_id in sample[1:]]

                max_user_id = max(max_user_id, user_id)
                max_item_id = max(max_item_id, max(item_ids))

                assert len(item_ids) >= 5, 'Core-5 dataset required'

                # item_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                # prefix_length: 5, 6, 7, 8
                for prefix_length in range(5 - 2 + 1, len(item_ids) - 2 + 1):
                    # prefix = [1, 2, 3, 4, 5]
                    # prefix = [1, 2, 3, 4, 5, 6]
                    # prefix = [1, 2, 3, 4, 5, 6, 7]
                    # prefix = [1, 2, 3, 4, 5, 6, 7, 8]
                    prefix = item_ids[
                             :prefix_length
                             ]  # TODO no sliding window, only incrementing sequence from last 50 items

                    train_dataset.append(
                        {
                            "user.ids": [user_id],
                            "user.length": 1,
                            "item.ids": prefix[-max_sequence_length:],
                            "item.length": len(prefix[-max_sequence_length:]),
                        }
                    )

                # item_ids[:-1] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                validation_dataset.append(
                    {
                        "user.ids": [user_id],
                        "user.length": 1,
                        "item.ids": item_ids[:-1][-max_sequence_length:],
                        "item.length": len(item_ids[:-1][-max_sequence_length:]),
                    }
                )

                # item_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                test_dataset.append(
                    {
                        "user.ids": [user_id],
                        "user.length": 1,
                        "item.ids": item_ids[-max_sequence_length:],
                        "item.length": len(item_ids[-max_sequence_length:]),
                    }
                )

        logger.info("Train dataset size: {}".format(len(train_dataset)))
        logger.info("Validation dataset size: {}".format(len(validation_dataset)))
        logger.info("Test dataset size: {}".format(len(test_dataset)))
        logger.info("Max user id: {}".format(max_user_id))
        logger.info("Max item id: {}".format(max_item_id))
        logger.info("Max sequence length: {}".format(max_sequence_length))
        logger.info(
            "{} dataset sparsity: {}".format(
                config["name"],
                (len(train_dataset) + len(test_dataset)) / max_user_id / max_item_id,
            )
        )

        train_sampler = TrainSampler(train_dataset, config['samplers']['type'])
        validation_sampler = EvalSampler(validation_dataset)
        test_sampler = EvalSampler(test_dataset)

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_users=max_user_id,
            num_items=max_item_id,
            max_sequence_length=max_sequence_length,
        )


class LetterFullDataset(ScientificFullDataset):
    @classmethod
    def create_from_config(cls, config):
        user_interactions_path = os.path.join(config["letter_inter_json"])
        with open(user_interactions_path, "r") as f:
            user_interactions = json.load(f)

        dir_path = os.path.join(config["path_to_data_dir"], config["name"])
        file_name = "all_data_from_letter"
        os.makedirs(dir_path, exist_ok=True)
        dataset_path = os.path.join(dir_path, f"{file_name}.txt")

        logger.info(f"Saving data to {dataset_path}")

        # Map from LETTER format to Our format
        with open(dataset_path, "w") as f:
            for user_id, item_ids in user_interactions.items():
                items_repr = map(str, item_ids)
                f.write(f"{user_id} {' '.join(items_repr)}\n")

        dataset = ScientificFullDataset.create_from_config(config, file_name)

        return cls(
            train_sampler=dataset._train_sampler,
            validation_sampler=dataset._validation_sampler,
            test_sampler=dataset._test_sampler,
            num_users=dataset._num_users,
            num_items=dataset._num_items,
            max_sequence_length=dataset._max_sequence_length,
        )
