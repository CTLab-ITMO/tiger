import json
import re

import murmurhash
import torch


class BaseBatchProcessor:

    def __call__(self, batch):
        raise NotImplementedError

class BasicBatchProcessor(BaseBatchProcessor):

    def __call__(self, batch):
        processed_batch = {}

        for key in batch[0].keys():
            if key.endswith('.ids'):
                prefix = key.split('.')[0]
                assert '{}.length'.format(prefix) in batch[0]

                processed_batch[f'{prefix}.ids'] = []
                processed_batch[f'{prefix}.length'] = []

                for sample in batch:
                    processed_batch[f'{prefix}.ids'].extend(sample[f'{prefix}.ids'])
                    processed_batch[f'{prefix}.length'].append(sample[f'{prefix}.length'])

        for part, values in processed_batch.items():
            processed_batch[part] = torch.tensor(values, dtype=torch.long)

        return processed_batch


class LetterBatchProcessor(BasicBatchProcessor):

    def __init__(self, mapping: dict[int, list[int]], semantic_length: int):
        self._prefixes = ['item', 'labels', 'positive', 'negative']
        self._semantic_length = semantic_length
        self._mapping = mapping

        assert sorted(mapping.keys()) == list(range(len(mapping))), "Item ids must be consecutive"
        self._mapping_tensor = torch.zeros((len(mapping), semantic_length), dtype=torch.long)
        for item_id, semantic_ids in mapping.items():
            self._mapping_tensor[item_id] = torch.tensor(semantic_ids, dtype=torch.long)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        mapping_path = config["letter_index_json"]
        with open(mapping_path, "r") as f:
            mapping = json.load(f)

        semantic_length = config["semantic_length"]

        parsed = {}

        for key, semantic_ids in mapping.items():
            numbers = [int(re.search(r'\d+', item).group()) for item in semantic_ids]
            assert len(numbers) == semantic_length, "All semantic ids must have the same length"
            parsed[int(key)] = numbers

        return cls(mapping=parsed, semantic_length=semantic_length)

    def __call__(self, batch):
        processed_batch = super().__call__(batch)

        for prefix in self._prefixes:
            if f"{prefix}.ids" in processed_batch:
                ids = processed_batch[f"{prefix}.ids"]
                lengths = processed_batch[f"{prefix}.length"]
                assert ids.min() >= 0; assert ids.max() < self._mapping_tensor.size(0)
                processed_batch[f"semantic_{prefix}.ids"] = self._mapping_tensor[ids].flatten()
                processed_batch[f"semantic_{prefix}.length"] = lengths * self._semantic_length

        processed_batch['hashed_user.ids'] = torch.tensor(
            list(map(lambda x: murmurhash.hash(str(x)) % 2000, processed_batch['user.ids'].tolist())),
            dtype=torch.long
        )

        processed_batch["all_semantic_ids"] = self._mapping_tensor

        return processed_batch
