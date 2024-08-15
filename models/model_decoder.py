#! /usr/bin/env python
# @Project  : Lite_Chag2cap
# @Time     : 2023/11/19
# @Author   : Sun Dongwei
# @File     : model_decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class LightTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt


class StackTransformer(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
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

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LightDecoderGenerator(nn.Module):
    def __init__(self, encoder_dim, n_layers, feature_dim, vocab_size, n_head, max_lengths, word_vocab, dropout):
        super().__init__()

        self.feature_dim = feature_dim
        self.embedding_dim = feature_dim
        self.vocab_size = vocab_size
        self.max_length = max_lengths
        self.word_vocab = word_vocab
        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv2d(encoder_dim * 2, feature_dim, kernel_size=1)
        self.resblock = ResBlock(feature_dim, feature_dim)
        self.vocab_embedding = nn.Embedding(vocab_size, feature_dim)
        decoder_layer = LightTransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=feature_dim * 2, dropout=dropout)
        self.transformer = StackTransformer(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoder(feature_dim, max_len=max_lengths)
        self.fc = nn.Linear(feature_dim, vocab_size)
        self.cos = nn.CosineSimilarity(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.vocab_embedding.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x1, x2, encoded_captions, caption_lengths):
        x_sam = self.cos(x1, x2)
        x = torch.cat([x1, x2], dim=1) + x_sam.unsqueeze(1) 
        x = self.resblock(self.conv1(x))
        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(2, 0, 1)

        word_length = encoded_captions.size(1)
        mask = torch.triu(torch.ones(word_length, word_length) * float('-inf'), diagonal=1).cuda()
        tgt_pad_mask = (encoded_captions == self.word_vocab['<NULL>']) | (encoded_captions == self.word_vocab['<END>'])
        word_embed = self.vocab_embedding(encoded_captions).transpose(1, 0)
        word_embed = self.position_encoding(word_embed)

        pred = self.transformer(word_embed, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)
        pred = self.fc(pred)
        pred = pred.permute(1, 0, 2)

        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()

        return pred, encoded_captions, decode_lengths, sort_ind

    def sample(self, x1, x2, k=1):
        """
        :param x1, x2: encoded images, a tensor of dimension (batch_size, channel, enc_image_size, enc_image_size)
        """
        x_sam = self.cos(x1, x2)
        x = torch.cat([x1, x2], dim=1) + x_sam.unsqueeze(1)  # (batch_size, 2channel, enc_image_size, enc_image_size)

        x = self.resblock(self.conv1(x))
        # x = self.LN(self.conv1(x))
        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(2, 0, 1)  # (hw, batch_size, feature_dim)

        tgt = torch.zeros(batch, self.max_length).to(torch.int64).cuda()

        mask = torch.triu(torch.ones(self.max_length, self.max_length) * float('-inf'), diagonal=1)
        mask = mask.cuda()
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] * batch).cuda()  # (batch_size*k, 1)
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] * batch).cuda()
        # Weight = torch.zeros(1, self.max_length, x.size(0)).cuda()
        for step in range(self.max_length):
            tgt_pad_mask = (tgt == self.word_vocab['<NULL>'])
            word_emb = self.vocab_embedding(tgt)
            word_emb = word_emb.transpose(1, 0)  # (length, batch, feature_dim)

            word_emb = self.position_encoding(word_emb)
            pred = self.transformer(word_emb, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)

            pred = self.fc(self.dropout(pred))  # (length, batch, vocab_size)
            scores = pred.permute(1, 0, 2)  # (batch, length, vocab_size)
            scores = scores[:, step, :].squeeze(1)  # [batch, 1, vocab_size] -> [batch, vocab_size]
            predicted_id = torch.argmax(scores, axis=-1)
            seqs = torch.cat([seqs, predicted_id.unsqueeze(1)], dim=-1)
            # Weight = torch.cat([Weight, weight], dim = 0)
            if predicted_id == self.word_vocab['<END>']:
                break
            if step < (self.max_length - 1):  # except <END> node
                tgt[:, step + 1] = predicted_id
        seqs = seqs.squeeze(0)
        seqs = seqs.tolist()

        # feature=x.clone()
        # Weight1=Weight.clone()
        return seqs

    def sample1(self, x1, x2, k=1):
        """
        :param x1, x2: encoded images, a tensor of dimension (batch_size, channel, enc_image_size, enc_image_size)
        :param max_lengths: maximum length of the generated captions
        :param k: beam_size
        """

        x = torch.cat([x1, x2], dim = 1)
        x = self.LN(self.conv1(x))
        batch, channel, h, w = x.shape
        x = x.view(batch, channel, -1).unsqueeze(0).expand(k, -1, -1, -1).reshape(batch*k, channel, h*w).permute(2, 0, 1) #(h*w, batch, feature_dim)

        tgt = torch.zeros(k*batch, self.max_length).to(torch.int64).cuda()

        mask = (torch.triu(torch.ones(self.max_length, self.max_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.cuda()
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] *batch*k).cuda() #(batch_size*k, 1)
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] *batch*k).cuda()
        top_k_scores = torch.zeros(k*batch, 1).cuda()
        complete_seqs = []
        complete_seqs_scores = []
        for step in range(self.max_length):
            word_emb = self.vocab_embedding(tgt)
            word_emb = word_emb.transpose(1, 0)
            word_emb = self.position_encoding(word_emb)
            pred = self.transformer(word_emb, x, tgt_mask=mask)
            pred = self.fc(self.dropout(pred))  # (length, batch, vocab_size)
            scores = pred.permute(1, 0, 2) # (batch, length, vocab_size)
            scores = scores[:, step, :].squeeze(1)  # [batch, 1, vocab_size] -> [batch, vocab_size]
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
            prev_word_inds = torch.div(top_k_words, self.vocab_size, rounding_mode='floor')
            next_word_inds = top_k_words % self.vocab_size  # (s)
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim = 1)
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != self.word_vocab['<END>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            x = x[:,prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            tgt = tgt[incomplete_inds]
            if step<self.max_length-1:
                tgt[:, :step+2] = seqs


        if complete_seqs == []:
            complete_seqs.extend(seqs[incomplete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[incomplete_inds])
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        return seq

