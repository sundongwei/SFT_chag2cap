#! /usr/bin/env python
# @Project  : Lite_Chag2cap
# @Time     : 2023/11/19
# @Author   : Sun Dongwei
# @File     : CNN_Nets.py
import torch
import torch.nn as nn
import torchvision.models as models


class Con_Net(nn.Module):
    def __int__(self, network):
        super().__int__()
        self.network = network
        if self.network == 'resnet18':
            cnn = models.resnet18(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network == 'resnet34':
            cnn = models.resnet34(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network == 'resnet50':
            cnn = models.resnet50(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network == 'resnet101':
            cnn = models.resnet101(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network == 'resnet152':
            cnn = models.resnet152(pretrained=True)
            modules = list(cnn.children())[:-2]

        self.cnn = nn.Sequential(*modules)
        self.fine_tuning()

    def forward(self, imageA, imageB):
        featureA = self.cnn(imageA)
        featureB = self.cnn(imageB)
        return featureA, featureB

    def fine_tuning(self, fine_tuning=True):
        for p in self.cnn.parameters():
            p.requires_grad = False
        #  If fine-tuning, only fine-tune convolutional blocks A through B
        for c in list(self.cnn.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tuning
