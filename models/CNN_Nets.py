#! /usr/bin/env python
# @Project  : Lite_Chag2cap
# @Time     : 2023/11/19
# @Author   : Sun Dongwei
# @File     : CNN_Nets.py
import torch
import torch.nn as nn
import torchvision.models as models


class Con_Net(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
        if self.network == 'resnet18':
            cnn = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
            modules = list(cnn.children())[:-2]
        elif self.network == 'resnet34':
            cnn = models.resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')
            modules = list(cnn.children())[:-2]
        elif self.network == 'resnet50':
            cnn = models.resnet50(weights='ResNet50_Weights.DEFAULT')
            modules = list(cnn.children())[:-2]
        elif self.network == 'resnet101':
            cnn = models.resnet101(weights='ResNet101_Weights.DEFAULT')
            modules = list(cnn.children())[:-2]
            
        elif self.network == 'resnet152':
            cnn = models.resnet152(weights='ResNet152_Weights.IMAGENET1K_V1')
            modules = list(cnn.children())[:-2]
        elif self.network == 'vgg':
            cnn = models.vgg16_bn(weights='VGG16_BN_Weights.DEFAULT')
            cnn.features[40] = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=3, padding=1)
            cnn.features[41] = nn.BatchNorm2d(2048)
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

