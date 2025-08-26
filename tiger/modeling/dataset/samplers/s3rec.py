from dataset.samplers.base import TrainSampler, EvalSampler
from dataset.negative_samplers.base import BaseNegativeSampler

import copy
import numpy as np


class S3RecPretrainTrainSampler(TrainSampler, config_name='s3rec_pretrain'):

    def __init__(self, dataset, num_users, num_items, negative_sampler, mask_prob=0.0):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._mask_item_idx = self._num_items + 1
        self._mask_prob = mask_prob
        self._negative_sampler = negative_sampler

        self._long_sequence = []
        for sample in self._dataset:
            self._long_sequence.extend(copy.deepcopy(sample['item.ids']))

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config({'type': config['negative_sampler_type']}, **kwargs)

        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler,
            mask_prob=config.get('mask_prob', 0.0)
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids']

        if len(item_sequence) < 3:
            assert False, 'Something strange is happening'

        # Masked Item Prediction
        masked_sequence = []
        positive_sequence = []
        negative_sequence = []

        for item in item_sequence:
            prob = np.random.rand()

            if prob < self._mask_prob:
                masked_sequence.append(self._mask_item_idx)
                positive_sequence.append(item)
                negative_sequence.append(
                    self._negative_sampler.generate_negative_samples(sample, 1)[0]
                )

            else:
                masked_sequence.append(item)
                positive_sequence.append(0)
                negative_sequence.append(self._mask_item_idx)

        # Mask last item
        masked_sequence[-1] = self._mask_item_idx
        positive_sequence[-1] = item_sequence[-1]
        negative_sequence[-1] = self._negative_sampler.generate_negative_samples(sample, 1)[0]
        assert len(positive_sequence) == len(negative_sequence) == len(masked_sequence) == len(item_sequence)

        # Segment Prediction
        sample_length = np.random.randint(1, (len(item_sequence) + 1) // 2)
        start_id = np.random.randint(0, len(item_sequence) - sample_length)
        negative_start_id = np.random.randint(0, len(self._long_sequence) - sample_length)
        masked_segment_sequence = item_sequence[:start_id] + [self._mask_item_idx] * sample_length + item_sequence[start_id + sample_length:]
        positive_segment = item_sequence[start_id: start_id + sample_length]
        negative_segment = self._long_sequence[negative_start_id:negative_start_id + sample_length]
        assert len(positive_segment) == len(negative_segment) == sample_length

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': masked_sequence,
            'item.length': len(masked_sequence),

            'positive.ids': item_sequence,
            'positive.length': len(item_sequence),

            'negative.ids': negative_sequence,
            'negative.length': len(negative_sequence),

            "item_segment.ids": masked_segment_sequence,
            "item_segment.length": len(masked_segment_sequence),

            'positive_segment.ids': positive_segment,
            'positive_segment.length': len(positive_segment),

            'negative_segment.ids': negative_segment,
            'negative_segment.length': len(negative_segment)
        }


class S3RecPretrainEvalSampler(EvalSampler, config_name='s3rec_pretrain'):

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )
