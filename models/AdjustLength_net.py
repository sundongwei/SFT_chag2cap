#!/usr/bin/env python
# @Project  : Lite_Chag2cap
# @Time     : 2023/12/11
# @Author   : Sun Dongwei
# @File     : AdjustLength_net.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
from supplemental.heatmap import *

class AdjustableLengthAttention(nn.Module):
    """ Adjustable Length Row and Column Attention Module """
    def __init__(self, in_dim, length=8):
        super(AdjustableLengthAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.length = length

    def forward(self, x1, x2, x3):
        m_batchsize, c, height, width = x1.shape
        proj_query = self.query_conv(x1)
        
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        
        proj_key = self.key_conv(x2)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        
        proj_value = self.value_conv(x3)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        
        length = self.length
        INF = lambda b, h, w: -torch.diag(torch.tensor(float('inf')).repeat(h), 0).unsqueeze(0).repeat(b * w, 1, 1)

        if length < height:
            mask_H = torch.ones(m_batchsize * width, height, height)
            mask_H[:, :, length:] = 0
            mask_H[:, length:, :] = 0
            proj_query_H = proj_query_H * mask_H
            proj_key_H = proj_key_H * mask_H

        if length < width:
            mask_W = torch.ones(m_batchsize * height, width, width)
            mask_W[:, :, length:] = 0
            mask_W[:, length:, :] = 0
            proj_query_W = proj_query_W * mask_W
            proj_key_W = proj_key_W * mask_W
        
        energy_H = ((torch.bmm(proj_query_H, proj_key_H) + INF(m_batchsize, height, width).cuda()).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)).cuda()
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        
        return self.gamma * (out_H + out_W) + x1



class Adjust_Trans(nn.Module):
    def __init__(self, n_layers, feature_size):
        super().__init__()
        h_feat, w_feat, channels = feature_size

        self.embedding_h = nn.Embedding(h_feat, int(channels/2))
        self.embedding_w = nn.Embedding(w_feat, int(channels/2))
        self.sparse_att = nn.ModuleList([])
        for i in range(n_layers):
            self.sparse_att.append(nn.ModuleList([
                AdjustableLengthAttention(channels, length=8),
                AdjustableLengthAttention(channels, length=8)
            ]))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, imgA, imgB):
        batch, c, h, w = imgA.shape
        pos_h = torch.arange(h).cuda()
        pos_w = torch.arange(w).cuda()
        embed_h = self.embedding_h(pos_h)
        embed_w = self.embedding_w(pos_w)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                       embed_h.unsqueeze(1).repeat(1, w, 1)],
                                       dim=-1)

        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        imgA = imgA + position_embedding
        imgB = imgB + position_embedding

        img_sa1, img_sa2 = imgA, imgB

        for (l, m) in self.sparse_att:
            img_fea_a = l(img_sa1, img_sa1, img_sa1)
            img_fea_b = m(img_sa2, img_sa2, img_sa2)

        return img_fea_a, img_fea_b


