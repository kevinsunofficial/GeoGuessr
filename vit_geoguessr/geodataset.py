import os
import os.path as osp
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class GeoDataset(Dataset):
    def __init__(self, root_dir, img_w=256, img_h=128, label_name='coords_data.csv', transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.images = np.load(osp.join(root_dir, f'images_{img_w}x{img_h}.npy'))
        self.df = pd.read_csv(osp.join(root_dir, label_name), header=0, index_col=None)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        image = self.images[index]
        coord = np.array(self.df.iloc[index, :2].values, dtype=np.float32)
        if self.transform is not None:
            image = self.transform(image)
        
        return image, coord
    

class rawGeoDataset(Dataset):
    def __init__(self, root_dir, img_w=256, img_h=128, label_name='images.csv', transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.img_w = img_w
        self.img_h = img_h
        self.df = pd.read_csv(osp.join(root_dir, label_name), header=None, index_col=None)
        self.df = self.df[self.df[6] == 1]
        self.df = self.df.iloc[:, :4]
        self.df = self.df.rename({
            0: 'id', 1: 'lat', 2: 'lng', 3: 'date'
        })
        self.df.lat /= 90
        self.df.lng /= 180
        self.transform = transform
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        image_id = self.df.id[index]
        image = np.array(Image.open(osp.join(self.root_dir, f'{image_id}.jpeg')).resize((self.img_w, self.img_h)), dtype=np.float32)
        coord = np.array(self.df.loc[index, ['lat', 'lng']].values, dtype=np.float32)
        if self.transform is not None:
            image = self.transform(image)
        
        return image, coord
