#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :buildDataset.py
@说明        :
@时间        :2021/05/29 15:47:29
@作者        :Oasis
@版本        :1.0
'''


import os

import h5py

import numpy as np
from imageUtils.ImageUtils import *
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class DIV2KCropTrainDataset(Dataset):
    def __init__(self, h5_file):
        super(DIV2KCropTrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, index):
        with h5py.File(self.h5_file, 'r') as f:
            return np.transpose(f['lr'][str(index)][:, :, :].astype(numpy.float32), (2, 0, 1))/255., np.transpose(f['hr'][str(index)][:, :, :].astype(numpy.float32), (2, 0, 1))/255.

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

class DIV2KCropValidDataset(Dataset):
    def __init__(self, h5_file):
        super(DIV2KCropValidDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, index):
        with h5py.File(self.h5_file, 'r') as f:
            return np.transpose(f['lr'][str(index)][:, :, :].astype(numpy.float32), (2, 0, 1))/255., np.transpose(f['hr'][str(index)][:, :, :].astype(numpy.float32), (2, 0, 1))/255.

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])