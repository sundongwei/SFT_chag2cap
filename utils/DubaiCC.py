import torch
from torch.utils.data import Dataset
from preprocess_data import token_encode_process
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
#import cv2 as cv
from imageio.v3 import imread
from random import *


class DubaiCCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, list_path, split, token_folder = None, vocab_file = None, max_length = 80, allow_unk = 0, max_iters=None):
        """
        :param data_folder: folder where image files are stored
        :param list_path: folder where the file name-lists of Train/val/test.txt sets are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param token_folder: folder where token files are stored
        :param vocab_file: the name of vocab file
        :param max_length: the maximum length of each caption sentence
        :param max_iters: the maximum iteration when loading the data
        :param allow_unk: whether to allow the tokens have unknow word or not
        """
        self.mean=[156.0681, 115.0891, 92.0745]
        self.std=[76.5139, 52.0616, 44.4209]
        self.list_path = list_path
        self.split = split
        self.max_length = max_length

        assert self.split in {'train', 'val', 'test'}
        self.img_ids = [i_id.strip() for i_id in open(os.path.join(list_path + split + '.txt'))]
        if vocab_file is not None:
            with open(os.path.join(list_path + vocab_file + '.json'), 'r') as f:
                self.word_vocab = json.load(f)
            self.allow_unk = allow_unk
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]
        self.files = []
        if split =='train':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/500_2000/' + name.split('-')[0])
                img_fileB = img_fileA.replace('2000', '2010')
                token_id = name.split('-')[-1]
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA": img_fileA,
                    "imgB": img_fileB,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name.split('-')[0]
                })
        elif split =='val':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/500_2000/' + name)
                img_fileB = img_fileA.replace('2000', '2010')
                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA": img_fileA,
                    "imgB": img_fileB,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name
                })
        elif split =='test':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/500_2000/' + name)
                img_fileB = img_fileA.replace('2000', '2010')
                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA": img_fileA,
                    "imgB": img_fileB,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]
        imgA = imread(datafiles["imgA"])
        imgB = imread(datafiles["imgB"])
        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)  
        imgA = np.moveaxis(imgA, -1, 0)     
        imgB = np.moveaxis(imgB, -1, 0)

        for i in range(len(self.mean)):
            imgA[i,:,:] -= self.mean[i]
            imgA[i,:,:] /= self.std[i]
            imgB[i,:,:] -= self.mean[i]
            imgB[i,:,:] /= self.std[i]      
        if datafiles["token"] is not None:
            caption = open(datafiles["token"])
            caption = caption.read()
            caption_list = json.loads(caption)

            token_all = np.zeros((len(caption_list),self.max_length),dtype=int)
            token_all_len = np.zeros((len(caption_list),1),dtype=int)
            for j, tokens in enumerate(caption_list):
                tokens_encode = token_encode_process(tokens, self.word_vocab, allow_unknown=self.allow_unk == 1)
                token_all[j,:len(tokens_encode)] = tokens_encode
                token_all_len[j] = len(tokens_encode)
            if datafiles["token_id"] is not None:
                id = int(datafiles["token_id"])
                token = token_all[id]
                token_len = token_all_len[id].item()
            else:
                j = randint(0, len(caption_list) - 1)
                token = token_all[j]
                token_len = token_all_len[j].item()
        else:
            token_all = np.zeros(1,dtype=int)
            token = np.zeros(1,dtype=int)
            token_len = np.zeros(1,dtype=int)
            token_all_len = np.zeros(1,dtype=int)

        return imgA.copy(), imgB.copy(), token_all.copy(), token_all_len.copy(), token.copy(), np.array(token_len), name


if __name__ == '__main__':
    
    train_dataset = DubaiCCDataset(data_folder='/home/sdw/paper_projects/Lite_Chag2cap/data/Dubai_CC/images',
                                   list_path='/home/sdw/paper_projects/Lite_Chag2cap/data/Dubai_CC/', split='test',
                                   token_folder=None)
    train_loader = DataLoader(dataset=train_dataset,batch_size=1,shuffle=False,pin_memory=True)
    channels_sumA,channel_squared_sumA,channels_sumB,channel_squared_sumB = 0,0,0,0
    num_batches = len(train_loader)
    index = 0
    for dataA,dataB,_,_,_,_,_ in train_loader:
        index += 1
        if index%1000==0:
           print(index,num_batches)
        channels_sumA += torch.mean(dataA, dim=[0,2,3])
        channel_squared_sumA += torch.mean(dataA**2, dim=[0,2,3])
        channels_sumB += torch.mean(dataB,dim=[0,2,3])   
        channel_squared_sumB += torch.mean(dataB**2, dim=[0,2,3])
        channels_sum = channels_sumA + channels_sumB
        channel_squared_sum = channel_squared_sumA + channel_squared_sumB
    meanA = channels_sumA/num_batches
    stdA = (channel_squared_sumA/num_batches - meanA**2)**0.5
    meanB = channels_sumB/num_batches
    stdB = (channel_squared_sumB/num_batches - meanB**2)**0.5
    mean = (channels_sum)/(num_batches*2)
    std = ((channel_squared_sum) / (num_batches*2) - mean**2)**0.5   
    print("meanA=", meanA)
    print("meanB=", meanB)
    print("stdA=", stdA)
    print("stdB=", stdB)
    print("mean=", mean)
    print("std", std)

#tensor([197.9497, 143.1499, 119.2271]) tensor([78.9855, 54.2107, 43.4881]) tensor([114.1864,  87.0282,  64.9218]) tensor([44.2934, 30.1198, 24.0976]) 
#tensor([156.0681, 115.0891,  92.0745]) tensor([76.5139, 52.0616, 44.4209])
