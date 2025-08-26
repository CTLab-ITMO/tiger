from utils.registry import MetaParent
from utils.grid_search import Params
import utils.tensorboards

import json
import random
import logging
import argparse
import numpy as np

import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# DEVICE = torch.device('cpu')


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


def maybe_to_list(values):
    if not isinstance(values, list):
        values = [values]
    return values


def get_activation_function(name, **kwargs):
    if name == 'relu':
        return torch.nn.ReLU()
    elif name == 'gelu':
        return torch.nn.GELU()
    elif name == 'elu':
        return torch.nn.ELU(alpha=float(kwargs.get('alpha', 1.0)))
    elif name == 'leaky':
        return torch.nn.LeakyReLU(negative_slope=float(kwargs.get('negative_slope', 1e-2)))
    elif name == 'sigmoid':
        return torch.nn.Sigmoid()
    elif name == 'tanh':
        return torch.nn.Tanh()
    elif name == 'softmax':
        return torch.nn.Softmax()
    elif name == 'softplus':
        return torch.nn.Softplus(beta=int(kwargs.get('beta', 1.0)), threshold=int(kwargs.get('threshold', 20)))
    elif name == 'softmax_logit':
        return torch.nn.LogSoftmax()
    else:
        raise ValueError('Unknown activation function name `{}`'.format(name))


def dict_to_str(x, params):
    parts = []
    for k, v in x.items():
        if k in params:
            if isinstance(v, dict):
                # part = '_'.join([f'{k}-{sub_part}' for sub_part in dict_to_str(v, params[k]).split('_')])
                part = '_'.join([f'{sub_part}' for sub_part in dict_to_str(v, params[k]).split('_')])
            elif isinstance(v, tuple) or isinstance(v, list):
                sub_strings = []
                for i, sub_value in enumerate(v):
                    sub_strings.append(f'({i})_{dict_to_str(v[i], params[k][i])}')
                part = f'({"_".join(sub_strings)})'
            else:
                # part = f'{k}-{v}'
                part = f'{v}'
            parts.append(part)
        else:
            continue
    return '_'.join(parts).replace('.', '-')


def create_masked_tensor(data, lengths):
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

    padded_tensor[mask] = data

    return padded_tensor, mask
