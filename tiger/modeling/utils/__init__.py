import argparse
import datetime
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TensorboardWriter(SummaryWriter):
    def __init__(
            self,
            experiment_name,
            logs_dir='../tensorboard_logs',
            use_time=True
        ):
        self._experiment_name = experiment_name
        os.makedirs(logs_dir, exist_ok=True)
        super().__init__(
            log_dir=os.path.join(
                logs_dir,
                f'{experiment_name}_{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M" if use_time else "")}'
            )
        )

    def add_scalar(self, *args, **kwargs):
        super().add_scalar(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)

    return params


def create_logger(
        name,
        level=logging.DEBUG,
        format='[%(asctime)s] [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
):
    logging.basicConfig(level=level, format=format, datefmt=datefmt)
    logger = logging.getLogger(name)
    return logger


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_activation_function(name, **kwargs):
    if name == 'relu':
        return torch.nn.ReLU()
    elif name == 'gelu':
        return torch.nn.GELU()
    elif name == 'elu':
        return torch.nn.ELU(alpha=float(kwargs.get('alpha', 1.0)))
    elif name == 'leaky':
        return torch.nn.LeakyReLU(negative_slope=float(kwargs.get('negative_slope', 1e-2)))
    else:
        raise ValueError('Unknown activation function name `{}`'.format(name))


def create_masked_tensor(data, lengths, is_right_aligned=False):
    batch_size = lengths.shape[0]
    max_sequence_length = lengths.max().item()

    if len(data.shape) == 1:  # only indices
        padded_tensor = torch.zeros(
            batch_size, max_sequence_length,
            dtype=data.dtype, device=DEVICE
        )  # (batch_size, max_seq_len)
    else:
        assert len(data.shape) == 2  # embeddings
        padded_tensor = torch.zeros(
            batch_size, max_sequence_length, data.shape[-1],
            dtype=data.dtype, device=DEVICE
        )  # (batch_size, max_seq_len, emb_dim)

    mask = torch.arange(
        end=max_sequence_length,
        device=DEVICE
    )[None].tile([batch_size, 1]) < lengths[:, None]  # (batch_size, max_seq_len)

    if is_right_aligned:
        mask = torch.flip(mask, dims=[-1])
    padded_tensor[mask] = data

    return padded_tensor, mask
