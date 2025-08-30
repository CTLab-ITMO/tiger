import argparse
import datetime
import json
import logging
import os
import random

import numpy as np
import torch


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE Training")
    parser.add_argument('--params', type=str, required=True,
                        help='Path to configuration file')
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        config = json.load(f)

    return config


def create_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_checkpoint_dir(ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir
