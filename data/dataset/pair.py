import os
import os.path as osp

import numpy as np
from PIL import Image
import torch


__all__ = [ "PairFaceDataset" ]

class PairFaceDataset:
    """Pairing Faces for evaluation"""
    def __init__(self, root, transform=None):
        # Save arguments
        self.root = root
        self.transform = transform

        # Read dataset
        self.data = np.load(osp.join(root, 'data.npy'))
        img_height = self.data.shape[-2]
        img_width = self.data.shape[-1]
        self.data = self.data.reshape(-1, 2, 3, img_height, img_width)
        self.label = np.load(osp.join(root, 'label.npy'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        img1 = Image.fromarray(pair[0])
        img2 = Image.fromarray(pair[1])
        label = self.label[idx]

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), label
