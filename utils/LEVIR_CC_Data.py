#!/usr/bin/env python
# @Project  : Lite_Chag2cap
# @Time     : 2023/11/9
# @Author   : Sun Dongwei
# @File     : LEVIR_CC_Data.py

from torch.utils.data import Dataset

import os


class LEVIR_CC_Dataset(Dataset):
    def __init__(self, data_path, list_path, split, max_length):

        self.mean = [100.6790,  99.5023,  84.9932]
        self.std = [50.9820, 48.4838, 44.7057]
        self.data_path = data_path
        self.list_path = list_path
        self.split = split
        self.max_length = max_length

        assert self.split in ['train', 'val', 'test']
        self.img_ids = [img_id.strip() for img_id in open(os.path.join(self.list_path, self.split+'.txt'))]


    def __getitem__(self, index):
        pass

    def __len__(self):
        pass