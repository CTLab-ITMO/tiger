import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from .tensorboards import GLOBAL_TENSORBOARD_WRITER, LOGS_DIR

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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

def save_sasrec_embeds(model, output_path):
    with torch.no_grad():
        item_embeddings = model._item_embeddings.weight.data.cpu().numpy()[1:]  # (num_items, embedding_dim)
        tensor_embeddings = torch.from_numpy(item_embeddings).float()
    assert tensor_embeddings.shape == (model._num_items, model._embedding_dim)
    torch.save(tensor_embeddings, output_path)


def generate_inter_json(user_interactions_path, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    inter_dict = {}

    with open(user_interactions_path, "r", encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            user_id_str = int(parts[0])
            item_id_strs = parts[1:]
            # Преобразуем все item_id к int
            user_items = []
            for item_str in item_id_strs:
                try:
                    user_items.append(int(item_str) - 1)
                except ValueError:
                    continue

            if user_items:
                inter_dict[user_id_str - 1] = user_items

    inter_path = output_path / "all-data-inter.json"
    with open(inter_path, "w", encoding='utf-8') as f:
        json.dump(inter_dict, f, indent=4, ensure_ascii=False)

    print(f"inter.json path: {inter_path}")
    print(f"Total users count: {len(inter_dict)}")
    return inter_path