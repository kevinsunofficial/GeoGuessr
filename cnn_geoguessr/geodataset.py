import os
import os.path as osp
import numpy as np
import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class GeoDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.images = np.load(osp.join(root_dir, f'{mode}_images.npy'))
        self.df = pd.read_csv(osp.join(root_dir, f'{mode}_images.csv'), header=0, index_col=None)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        image = self.images[index]
        coord = np.array(self.df.iloc[index, 1:3].values, dtype=np.float32)
        if self.transform is not None:
            image = self.transform(image)
        
        return image, coord
