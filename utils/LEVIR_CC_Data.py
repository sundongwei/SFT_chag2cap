#!/usr/bin/env python
# @Project  : Lite_Chag2cap
# @Time     : 2023/11/9
# @Author   : Sun Dongwei
# @File     : LEVIR_CC_Data.py
import json

import numpy as np
from imageio.v3 import imread
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

from preprocess_data import token_encode_process
import torchvision.transforms as transforms


class LEVIR_CC_Dataset(Dataset):
    def __init__(self, data_path, list_path, split, max_length=40, vocab_file=None, allow_unknown=0, max_iters=None,
                 token_folder=None):

        self.mean = [100.6790, 99.5023, 84.9932]
        self.std = [50.9820, 48.4838, 44.7057]
        self.data_path = data_path
        self.list_path = list_path
        self.split = split
        self.max_length = max_length

        assert self.split in ['train', 'val', 'test']
        self.img_details = [img_detail.strip() for img_detail in
                            open(os.path.join(self.list_path, self.split + '.txt'))]
        if vocab_file is not None:
            with open(os.path.join(list_path + vocab_file + '.json'), 'r') as f:
                self.word_vocab = json.load(f)
            self.allow_unknown = allow_unknown

        if max_iters is not None:
            n_repeat = (max_iters // len(self.img_details)) + 1
            self.img_details = self.img_details * n_repeat + self.img_details[
                                                             :max_iters - n_repeat * len(self.img_details)]
        self.files = []
        if split == 'train':
            for img_detail in self.img_details:
                img_A = os.path.join(self.data_path + '/' + split + '/A/' + img_detail.split('-')[0])
                img_B = os.path.join(self.data_path + '/' + split + '/B/' + img_detail.split('-')[0])
                token_id = img_detail.split('-')[-1]
                if token_folder is not None:
                    token_file = os.path.join(token_folder + img_detail.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "img_A": img_A,
                    "img_B": img_B,
                    "token": token_file,
                    "token_id": token_id,
                    "img_detail": img_detail.split('-')[0]
                })

        elif split == 'val':
            for img_detail in self.img_details:
                img_A = os.path.join(self.data_path + '/' + split + '/A/' + img_detail)
                img_B = os.path.join(self.data_path + '/' + split + '/B/' + img_detail)
                # token_id = img_detail.split('-')[-1]
                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + img_detail.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "img_A": img_A,
                    "img_B": img_B,
                    "token": token_file,
                    "token_id": token_id,
                    "img_detail": img_detail
                })

        elif split == 'test':
            for img_detail in self.img_details:
                img_A = os.path.join(self.data_path + '/' + split + '/A/' + img_detail)
                img_B = os.path.join(self.data_path + '/' + split + '/B/' + img_detail)
                # token_id = img_detail.split('-')[-1]
                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + img_detail.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "img_A": img_A,
                    "img_B": img_B,
                    "token": token_file,
                    "token_id": token_id,
                    "img_detail": img_detail
                })
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 转换为tensor并归一化到[0, 1]
        ])
        
        
    def __getitem__(self, index):
        datafiles = self.files[index]
        img_detail = datafiles["img_detail"]
        img_A = imread(datafiles["img_A"])
        img_B = imread(datafiles["img_B"])
        
        
        img_A = np.asarray(img_A, np.float32)
        img_B = np.asarray(img_B, np.float32)



        img_A = np.moveaxis(img_A, -1, 0)
        img_B = np.moveaxis(img_B, -1, 0)

        for i in range(len(self.mean)):
            img_A[i, :, :] -= self.mean[i]
            img_A[i, :, :] /= self.std[i]
            img_B[i, :, :] -= self.mean[i]
            img_B[i, :, :] /= self.std[i]

        if datafiles["token"] is not None:
            caption = open(datafiles["token"])
            caption = caption.read()
            caption_list = json.loads(caption)

            token_all = np.zeros((len(caption_list), self.max_length), dtype=int)
            token_all_len = np.zeros((len(caption_list), 1), dtype=int)
            for i, tokens in enumerate(caption_list):
                token_encode = token_encode_process(tokens, self.word_vocab, allow_unknown=self.allow_unknown == 1)
                token_all[i, :len(token_encode)] = token_encode
                token_all_len[i] = len(token_encode)

            if datafiles["token_id"] is not None:
                id = int(datafiles["token_id"])
                token = token_all[id]
                token_len = token_all_len[id].item()
            else:
                i = np.random.randint(len(caption_list) - 1)
                token = token_all[i]
                token_len = token_all_len[i].item()
        else:
            token_all = np.zeros(1, dtype=int)
            token_all_len = np.zeros(1, dtype=int)
            token = np.zeros(1, dtype=int)
            token_len = np.zeros(1, dtype=int)

        return (img_A.copy(), img_B.copy(), token_all.copy(), token_all_len.copy(), token.copy(), np.array(token_len),
                img_detail)

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    train_dataset = LEVIR_CC_Dataset(data_path='./data/LEVIR_CC/images', list_path='/home/sdw/paper_projects/Lite_Chag2cap/data/LEVIR_CC',
                                     split='train', token_folder=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True)
    print("ok")
