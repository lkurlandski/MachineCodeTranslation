"""
Configurations, such as random seeds and cpu/gpu devices.
"""

import random
import typing as tp

import numpy as np
import torch


SRC = "mal"
TGT = "ben"
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, ADDR_IDX, STR_IDX = 0, 1, 2, 3, 4, 5
UNK, PAD, BOS, EOS, ADDR, STR = "<unk>", "<pad>", "<bos>", "<eos>", "<addr>", "<str>"
SPECIAL_SYMBOLS = [UNK, PAD, BOS, EOS, ADDR, STR]


def print_config():
    print("-" * 88, f"{'-' * 40} CONFIG {'-' * 40}", "-" * 88, sep="\n")
    print(f"{seed=}")
    print(f"{device=}")
    print(f"{torch.backends.cudnn.enabled=}")


def init(device_: tp.Literal["cpu", "cuda:0"] = "cpu", seed_: int = 0, *, verbose: bool = True):
    global device
    global seed
    seed = seed_
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_)
    torch.backends.cudnn.enabled = False
    if verbose:
        print_config()


init("cpu", 0, verbose=False)
