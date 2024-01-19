from typing import Tuple, Callable
import xarray as xr
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import glob
import pdb
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from torchvision import transforms, datasets




@dataclass
class FAIR1MData(Dataset):

    data_path: str
    transform: Callable = None

    def __post_init__(self):

        dir = Path(self.data_path)
        gentif = dir.rglob('*.tif')
        genxml = dir.rglob('*.xml')

        self.images = []
        self.labels = []

        for i in gentif:
            self.images.append(str(i))

        for i in genxml:
            self.labels.append(str(i))

        self.images = sorted(self.images)
        self.labels = sorted(self.labels)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        import pdb; pdb.set_trace()
        label = self.label[idx]
        

        return None
