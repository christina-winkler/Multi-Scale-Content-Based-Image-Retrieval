from os.path import exists, join
from os import listdir
import torch.utils.data as data_utils
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math
import torch
import random
import sys
import pdb
import os
import xarray as xr
import subprocess

from datasets import fair1m_dataset

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
sys.path.append("../")


def load_fair1m(args):

    train_data = fair1m_dataset.FAIR1MData(data_path=args.datadir + '/interview_datasets/FAIR1M_partial/train')
    valid_data = fair1m_dataset.FAIR1MData(data_path=args.datadir + '/interview_datasets/FAIR1M_partial/valid')
    test_data = fair1m_dataset.FAIR1MData(data_path=args.datadir + '/interview_datasets/FAIR1M_partial/test')

    train_loader = data_utils.DataLoader(train_data, args.bsz, shuffle=True,
                                         drop_last=True)
    val_loader = data_utils.DataLoader(valid_data, args.bsz, shuffle=True,
                                       drop_last=True)
    test_loader = data_utils.DataLoader(test_data, args.bsz, shuffle=False,
                                        drop_last=False)

    return train_loader, val_loader, test_loader, args


def load_resisc45(args):

    return train_loader, val_loader, test_loader, args


def load_sentinel2(args):

    return train_loader, val_loader, test_loader, args


def load_data(args):

    if args.trainset == "fair1m":
        return load_fair1m(args)

    elif args.trainset == "resisc45":
        return load_resisc45(args)

    elif args.trainset == "sentinel2":
        return load_sentinel2(args)

    else:
        raise ValueError("Dataset not available. Check for typos!")
