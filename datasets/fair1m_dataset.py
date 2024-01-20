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

def get_img_data(img_tuple):
    # taken from: https://www.kaggle.com/code/ollypowell/fair1m-satellite-dataset-eda
    filepath = img_tuple[0]
    split = img_tuple[2]
    image_id = img_tuple[3]

    data={}
    with rasterio.open(filepath) as src:
        fp = Path(filepath)
        data['filename'] = str(fp.parent.name) + '/' + str(fp.name)
        data['Img_ID'] = image_id
        data['split'] = split
        data['crs'] = int(src.crs.to_epsg())
        data['channels'] = src.count
        data['data_type'] = src.dtypes[0]
        data['ImageWidth'], data['ImageHeight'] = src.width, src.height
        data['x0'], data['y0'] = src.xy(0, 0)    # .xy(row_no, column_no)
        data['x1'], data['y1'] = src.xy(src.height, 0)
        data['x2'], data['y2'] = src.xy(src.height, src.width)
        data['x3'], data['y3'] = src.xy(0,src.width)
        data['x_cen'], data['y_cen'] = src.xy(src.height//2, src.width//2)
    return data

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

        label = self.labels[idx]
        image = self.images[idx]

        # read label information
        with open(label, 'r') as f:
            label_dict = xmltodict.parse(f.read())

        objects = label_dict['annotation'].get('objects', {}).get('object', [])
        nested_objects = get_instances(label_dict)

        # get image from geotiff
        img = rasterio.open(image)
        img = img.read(1)

        import pdb; pdb.set_trace()

        width = int(label_dict['annotation']['size']['width'])
        height = int(label_dict['annotation']['size']['height'])
        extras = {'FilePath':jpg_path, 'Split': split_nm, 'ImageWidth': width, 'ImageHeight': height}

        import pdb; pdb.set_trace()


        return None
