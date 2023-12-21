#!/usr/bin/env python
# @Project  : Lite_Chag2cap
# @Time     : 2023/11/8
# @Author   : Sun Dongwei
# @File     : train.py
import argparse
import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from utils.LEVIR_CC_Data import LEVIR_CC_Dataset

from torch.nn.utils.rnn import pack_padded_sequence
from utils.utils import get_eval_score, accuracy
from models.CNN_Nets import Con_Net
from models.model_decoder import Decoder_Generator
# from models.axial_attention.axial_attention import AxialImageTransformer
from models.cc_net import CC_Trans


def main(args):
    """Training and Validation"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # save checkpoint
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

    start_epoch = 0
    best_bleu4 = 0.

    # Read Word Map
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_map = json.load(f)

    """Initialize Network Details"""
    # CNN Extractor
    extractor = Con_Net(args.cnn_net)
    extractor.fine_tuning(args.fine_tune)

    # FIXME
    # Transformer Encoder
    # encoder = TransformerEncoder(n_layers=args.n_layers, d_model=args.d_model, n_heads=args.n_heads)
    # encoder = AxialImageTransformer(dim=args.d_model, depth=12, heads=args.n_heads, reversible=True,
    #                                 axial_pos_emb_shape=None)
    encoder = CC_Trans(n_layers=args.n_layer, feature_size=[args.feat_size, args.feat_size, args.encoder_dim])

    # Caption Generator
    generator = Decoder_Generator(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_map),
                                  max_lengths=args.max_length, word_vocab=word_map, n_head=args.n_heads,
                                  n_layers=args.decoder_n_layers, dropout=args.dropout)

    # Optimizers
    extractor_optimizer = torch.optim.Adam(extractor.parameters(), lr=args.cnn_lr)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.encoder_lr)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.decoder_lr)

    # Move models to GPU
    extractor = extractor.cuda()
    encoder = encoder.cuda()
    generator = generator.cuda()

    # Parameters Info Print
    print("------------Checkpoint-SavePath------------{}".format(args.save_path))
    print("------------extractor_CNN------------{}".format(args.cnn_net))
    #FIXME
    # print("------------encoder_Transformer------------{}".format())
    # print("------------generator_Transformer------------{}".format())

    # Loss Function
    criterion = nn.CrossEntropyLoss().cuda()

    # Custom DataLoad
    if args.data_name == 'LEVIR_CC':
        train_dataloader = data.DataLoader(LEVIR_CC_Dataset(args.data_path, args.list_path, split='train',
                                                            token_folder=args.token_folder, vocab_file=args.vocab_file,
                                                            max_length=args.max_length,
                                                            allow_unknown=args.allow_unknown),
                                           batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers,
                                           pin_memory=True)
        print("----------train_dataloader length-------------)", len(train_dataloader))
        valid_dataloader = data.DataLoader(LEVIR_CC_Dataset(args.data_path, args.list_path, split='val',
                                                            token_folder=args.token_folder, vocab_file=args.vocab_file,
                                                            max_length=args.max_length,
                                                            allow_unknown=args.allow_unknown),
                                           batch_size=args.valid_batch_size, shuffle=True, num_workers=args.num_workers,
                                           pin_memory=True)

    elif args.data_name == 'Dubai_CC':
        train_dataloader = Dubai_CC_DataLoader(args.list_path, word_map, args.batch_size, args.num_workers)
        valid_dataloader = Dubai_CC_DataLoader(args.list_path, word_map, args.batch_size, args.num_workers)

    extractor_lr_scheduler = torch.optim.lr_scheduler.StepLR(extractor_optimizer, step_size=1, gamma=0.95)
    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=1, gamma=0.95)
    generator_lr_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=1, gamma=0.95)

    # track metric variable
    index_i = 0
    hist = np.zeros((args.num_epochs * len(train_dataloader), 3))

    # Start Training
    for epoch in range(start_epoch, args.num_epochs):
        for id, (imgA, imgB, _, _, token, token_len, _) in enumerate(train_dataloader):
            start_time = time.time()
            extractor.train()
            encoder.train()
            generator.train()

            if extractor_optimizer is not None:
                extractor_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            generator_optimizer.zero_grad()

            # Move Data to GPU
            imgA = imgA.cuda()
            imgB = imgB.cuda()

            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()

            feat_A, feat_B = extractor(imgA, imgB)
            feat_A, feat_B = encoder(feat_A, feat_B)
            score, caps_sorted, decode_lengths, sort_ind = generator(feat_A, feat_B, token, token_len)

            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(score, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Backward pass
            loss.backward()

            # Clip gradients
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(encoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(generator.parameters(), args.grad_clip)
                if encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(extractor.parameters(), args.grad_clip)

            # Update weights
            generator_optimizer.step()
            encoder_optimizer.step()
            if extractor_optimizer is not None:
                extractor_optimizer.step()

            # keep track metric
            hist[index_i, 0] = time.time() - start_time
            hist[index_i, 1] = loss.item()
            hist[index_i, 2] = accuracy(scores, targets, 5)
            index_i += 1

            # Print status
            if index_i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time: {3:.3f}\t'
                      'Loss: {4:.4f}\t'
                      'Top-5 Accuracy: {5:.3f}'.format(epoch, index_i, args.num_epochs * len(train_dataloader),
                                                       np.mean(hist[index_i - args.print_freq:index_i - 1, 0]) *
                                                       args.print_freq,
                                                       np.mean(hist[index_i - args.print_freq:index_i - 1, 1]),
                                                       np.mean(hist[index_i - args.print_freq:index_i - 1, 2])))

        # Per epoch's validation
        if extractor is not None:
            extractor.eval()
        encoder.eval()
        generator.eval()

        val_start_time = time.time()
        references = list()  # a list to store references (true captions)
        hypotheses = list()  # a list to store hypothesis (prediction)

        with torch.no_grad():
            for id, (imgA, imgB, token_all, token_len, _, _, _) in enumerate(valid_dataloader):
                # Move to GPU
                imgA = imgA.cuda()
                imgB = imgB.cuda()

                token_all = token_all.squeeze(0).cuda()

                if extractor is not None:
                    feat_A, feat_B = extractor(imgA, imgB)
                feat_A, feat_B = encoder(feat_A, feat_B)

                sequence = generator.sample(feat_A, feat_B, k=1)

                img_token = token_all.tolist()

                img_tokens = list(map(lambda c: [w for w in c if w not in {word_map['<START>'], word_map['<END>'],
                                                                           word_map['<NULL>']}], img_token))
                references.append(img_tokens)

                pred_sequence = [w for w in sequence if
                                 w not in {word_map['<START>'], word_map['<END>'], word_map['<NULL>']}]
                hypotheses.append(pred_sequence)
                assert len(references) == len(hypotheses)

                if id % args.print_freq == 0:
                    pred_caption = ""
                    ref_caption = ""
                    for i in pred_sequence:
                        pred_caption += (list(word_map.keys())[i]) + " "
                    ref_caption = ""
                    for i in img_tokens:
                        for j in i:
                            ref_caption += (list(word_map.keys())[j]) + " "
                        ref_caption += ".    "

            val_time = time.time() - val_start_time
            # Calculate BLEU/Meteor/Rouge... scores
            score_dict = get_eval_score(references, hypotheses)
            Bleu_1 = score_dict['Bleu_1']
            Bleu_2 = score_dict['Bleu_2']
            Bleu_3 = score_dict['Bleu_3']
            Bleu_4 = score_dict['Bleu_4']
            Meteor = score_dict['METEOR']
            Rouge_L = score_dict['ROUGE_L']
            Cider = score_dict['CIDEr']
            print('Validation:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.4f}\t' 'BLEU-2: {2:.4f}\t' 'BLEU-3: {3:.4f}\t'
                  'BLEU-4: {4:.4f}\t' 'Meteor: {5:.4f}\t' 'Rouge: {6:.4f}\t' 'Cider: {7:.4f}\t'
                  .format(val_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge_L, Cider))

        # Adjust Learning Rate
        generator_lr_scheduler.step()
        print("generator_lr: ", generator_optimizer.param_groups[0]['lr'])
        encoder_lr_scheduler.step()
        print("encoder_lr: ", encoder_optimizer.param_groups[0]['lr'])
        if extractor_lr_scheduler is not None:
            extractor_lr_scheduler.step()
            print("extractor_lr: ", extractor_optimizer.param_groups[0]['lr'])

        # Check whether to save best model
        if Bleu_4 > best_bleu4:
            best_bleu4 = max(Bleu_4, best_bleu4)
            print("Save Model")
            state = {
                'extractor_dict': extractor.state_dict(),
                'encoder_dict': encoder.state_dict(),
                'generator_dict': generator.state_dict(),
            }
            model_name = (str(args.data_name) + '_BatchSize' + str(args.train_batch_size) + str(args.cnn_net) + 'Bleu-4' +
                          str(round(10000 * best_bleu4)) + '.pth')
            torch.save(state, os.path.join(args.save_path, model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A lite model for remote sensing change2captioning')

    # LEVIR-CC paramters
    parser.add_argument("--gpu_id", type=int, default=0, help='gpu id of the devices')
    parser.add_argument("--save_path", type=str, default='./checkpoints', help='path to save checkpoints')
    parser.add_argument("--fine_tune", type=bool, default=True, help='fine tune cnn')
    parser.add_argument("--n_layer", type=int, default=2, help='number of layers')
    parser.add_argument("--d_model", type=int, default=512, help='dimension of model')
    parser.add_argument("--n_heads", type=int, default=8, help='number of heads')
    parser.add_argument("--vocab_file", type=str, default='vocab', help='vocab file')
    parser.add_argument("--data_name", type=str, default='LEVIR_CC', help='data name')
    parser.add_argument("--cnn_net", default='resnet101', help='extractor network')
    parser.add_argument("--encoder_dim", type=int, default=2048, help='the dim of extracted features by diff nets')
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--feat_size", type=int, default=16)
    parser.add_argument("--data_path", type=str, default='./data/LEVIR_CC/images/', help='data files path')
    parser.add_argument("--list_path", type=str, default='./data/LEVIR_CC/', help='list path')
    parser.add_argument("--token_folder", type=str, default='./data/LEVIR_CC/tokens/', help='token files path')
    parser.add_argument("--max_length", type=int, default=41, help='max length of each caption sentence')
    parser.add_argument("--allow_unknown", type=int, default=1, help='whether unknown tokens are allowed')
    parser.add_argument("--train_batch_size", type=int, default=32, help='batch size of training')
    parser.add_argument("--valid_batch_size", type=int, default=1, help='batch size of validation')
    parser.add_argument("--num_workers", type=int, default=0, help='to accelerate data load')
    parser.add_argument("--cnn_lr", type=float, default=1e-4, help='cnn learning rate')
    parser.add_argument("--encoder_lr", type=float, default=1e-4, help='encoder learning rate')
    parser.add_argument("--decoder_lr", type=float, default=1e-4, help='decoder learning rate')
    parser.add_argument("--num_epochs", type=int, default=40, help='number of epochs')
    parser.add_argument("--grad_clip", default=None, help='clip gradients')
    parser.add_argument("--print_freq", type=int, default=100, help='print frequency')
    parser.add_argument("--dropout", type=float, default=0.1,help='dropout')
    parser.add_argument("--decoder_n_layers", type=int, default=1)
    args = parser.parse_args()
    main(args)
