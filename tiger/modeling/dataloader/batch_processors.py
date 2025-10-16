import json

import murmurhash
import torch


class BatchProcessor:
    def __init__(self, semantic_ids=None, sem_id_len=None, user_ids_count=None):
        self._mapping = semantic_ids
        self._semantic_length = sem_id_len
        self._user_ids_count = user_ids_count

        if semantic_ids is not None:
            self._prefixes = ['item', 'labels', 'positive', 'negative']
            self._mapping_tensor = torch.tensor(semantic_ids, dtype=torch.long)

    @classmethod
    def create(cls, mapping_path, sem_id_len, user_ids_count):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)

        semantic_ids = []
        for i in range(len(mapping)):
            assert len(mapping[str(i)]) == sem_id_len, 'All semantic ids must have the same length'
            semantic_ids.append(mapping[str(i)])

        return cls(semantic_ids=semantic_ids, sem_id_len=sem_id_len, user_ids_count=user_ids_count)

    def __call__(self, batch):
        processed_batch = {}

        for key in batch[0].keys():
            if key.endswith('.ids'):
                prefix = key.split('.')[0]
                assert f'{prefix}.length' in batch[0]

                processed_batch[f'{prefix}.ids'] = []
                processed_batch[f'{prefix}.length'] = []

                for sample in batch:
                    processed_batch[f'{prefix}.ids'].extend(sample[f'{prefix}.ids'])
                    processed_batch[f'{prefix}.length'].append(sample[f'{prefix}.length'])

        for part, values in processed_batch.items():
            processed_batch[part] = torch.tensor(values, dtype=torch.long)

        if self._mapping is not None:
            for prefix in self._prefixes:
                if f'{prefix}.ids' in processed_batch:
                    ids = processed_batch[f'{prefix}.ids']
                    lengths = processed_batch[f'{prefix}.length']
                    assert ids.min() >= 0
                    assert ids.max() < self._mapping_tensor.size(0)
                    processed_batch[f'semantic_{prefix}.ids'] = self._mapping_tensor[ids].flatten()
                    processed_batch[f'semantic_{prefix}.length'] = lengths * self._semantic_length

        if self._user_ids_count is not None:
            processed_batch['hashed_user.ids'] = torch.tensor(
                list(map(lambda x: murmurhash.hash(str(x)) % self._user_ids_count,
                         processed_batch['user.ids'].tolist())),
                dtype=torch.long
            )

        return processed_batch
