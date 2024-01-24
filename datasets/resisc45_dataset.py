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
import xmltodict
import rasterio
import cv2

def get_geotiff_coordinates(geotiff_file):
    # taken from: https://www.kaggle.com/code/ollypowell/fair1m-satellite-dataset-eda
    with rasterio.open(str(geotiff_file)) as image_file:
        xmin, ymax = image_file.transform * (0, 0)  #top-left corner
        return ymax, xmin

def get_instances(xml_data):
    # taken from: https://www.kaggle.com/code/ollypowell/fair1m-satellite-dataset-eda
    try:
        items = xml_data['annotation']['objects']['object']
        if not isinstance(items, list):
            items = [items]
    except KeyError:
        items = []
    return items

@dataclass
class Resisc45(Dataset):

    data_path: str
    transform: Callable = None

    def __post_init__(self):

        dir = Path(self.data_path)
        gen_jpg = dir.rglob('*.jpg')

        self.images = []
        self.label_to_idx = {'airplane':0, 'ship':1, 'bridge':2}
        self.idx_to_label = {0: 'airplane', 1:'ship', 2:'bridge'}

        for i in gen_jpg:
            self.images.append(str(i))

        if self.transform is None:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        image = self.images[idx]

        # get label
        if 'airplane' in image:
            label = 0
        elif 'ship' in image:
            label = 1
        else:
            label = 2

        # read .jpg image from file
        img = cv2.imread(image)

        return self.transform(img)
