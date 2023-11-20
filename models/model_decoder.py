#! /usr/bin/env python
# @Project  : Lite_Chag2cap
# @Time     : 2023/11/19
# @Author   : Sun Dongwei
# @File     : model_decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class resblock(nn.Module):
    def __init__(self, In_Channel, Out_Channel, strode=1, shortcut=None):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(In_Channel, int(Out_Channel / 2), kernel_size=1),
            nn.BatchNorm2d(int(Out_Channel / 2)),
            nn.ReLU(),
            nn.Conv2d(int(Out_Channel / 2), int(Out_Channel / 2), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(Out_Channel / 2)),
            nn.ReLU(),
            nn.Conv2d(int(Out_Channel / 2), Out_Channel, kernel_size=1),
            nn.BatchNorm2d(Out_Channel),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x
        out = x + out
        return F.relu(out)


class Mesh_Transformer_Decoder_Layer(nn.Module):
    __constants = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multi_head_attn = nn.MultiheadAttention(d_model, int(nhead), dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.fc_alpha1 = nn.Linear(d_model +  d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model +  d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model +  d_model, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)

        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        # cross self-attention
        enc_att, att_weight = self.mha_head_block(self_att_tgt, memory, memory_mask, memory_key_padding_mask)
        x = self.norm2(self_att_tgt + enc_att)
        x = self.norm3(x + self._ff_block(x))
        return x + tgt

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def mha_head_block(self, x, memory, attn_mask, key_padding_mask):
        x, att_weight = self.multi_head_attn(x, memory, memory, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                                             need_weights=True)
        return self.dropout3(x), att_weight


class StackTransformer(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        #TODO
        self.layers = torch.nn.modules.Transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Decoder_Generator(nn.Module):
    def __init__(self, encoder_dim, n_layers, feature_dim, vocab_size, n_head, max_lengths, word_vocab, dropout):
        super().__init__()

        print("decoder_n_layer=", n_layers)

        self.feature_dim = feature_dim
        self.embedding_dim = feature_dim
        self.vocab_size = vocab_size
        self.max_length = max_lengths
        self.word_vocab = word_vocab
        self.dropout = dropout
        self.conv1 = nn.Conv2d(encoder_dim * 2, feature_dim, kernel_size=1)
        self.LN = resblock(feature_dim, feature_dim)
        # embedding layer, vocabulary embedding
        self.vocab_embedding = nn.Embedding(vocab_size, feature_dim)
        # Transformer later
        decoder_layer = Mesh_Transformer_Decoder_Layer(d_model=feature_dim, nhead=n_head, dim_feedforward=feature_dim * 4,
                                                       dropout=dropout)

        self.transformer = StackTransformer(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoder(feature_dim, max_len=max_lengths)

        # Linear layer to find  scores over vocabulary
        self.fc = nn.Linear(feature_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.init_weights()

    def init_weights(self):
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x1, x2, encoded_captions, caption_lengths):
        """
        :param x1: encoded image feature, tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param xi2: encoded image feature, tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        x_sam = self.cos(x1, x2)
        x = torch.cat([x1, x2], dim=1) + x_sam.unsqueeze(1)
        x = self.LN(self.conv1(x))

        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(2, 0, 1)

        word_length = encoded_captions.size(1)
        # Create masks
        mask = torch.triu(torch.ones(word_length, word_length) * float('-inf'), diagonal=1)
        mask = mask.cuda()
        tgt_pad_mask = (encoded_captions == self.word_vocab['<NULL>']) |  (encoded_captions == self.word_vocab['<END>'])
        word_embed = self.vocab_embedding(encoded_captions)
        word_embed = word_embed.transpose(1, 0)

        word_embed = self.position_encoding(word_embed)

        pred = self.transformer(word_embed, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)
        pred = self.fc(self.dropout(pred))
        pred = pred.permute(1, 0, 2)

        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()

        return pred, encoded_captions, decode_lengths, sort_ind





