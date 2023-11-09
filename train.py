#!/usr/bin/env python
# @Project  : Lite_Chag2cap
# @Time     : 2023/11/8
# @Author   : Sun Dongwei
# @File     : train.py
import argparse
import os
import time

import json

import torch

import torch.nn as nn


def main(args):

    """Training and Validation"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # save checkpoint
    if os.path.exists(args.save_path) == False:
        os.mkdir(args.save_path)

    print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

    start_epoch = 0
    best_bleu4 = 0.

    # Read Word Map
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_map = json.load(f)

    """Initialize Network Details"""
    # CNN Extractor

    extractor = CNNNet()
    extractor.fine_tune(args.fint_tune_cnn)

    # Transformer Encoder
    encoder = TransformerEncoder(n_layers=args.n_layers, d_model=args.d_model, n_heads=args.n_heads)

    # Caption Generator
    generator = TransformerGenerator(d_model=args.d_model, vocab_size=len(word_map))

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
    print("------------extractor_CNN------------{}".format())
    print("------------encoder_Transformer------------{}".format())
    print("------------generator_Transformer------------{}".format())

    # Loss Function
    criterion = nn.CrossEntropyLoss().cuda()

    # Custom DataLoad
    if args.data_name == 'LEVIR-CC':
        train_dataloader = LEVIR_CC_DataLoader(args.list_path, word_map, args.batch_size, args.num_workers)
        valid_dataloader = LEVIR_CC_DataLoader(args.list_path, word_map, args.batch_size, args.num_workers)

    elif args.data_name == 'Dubai_CC':
        train_dataloader = Dubai_CC_DataLoader(args.list_path, word_map, args.batch_size, args.num_workers)
        valid_dataloader = Dubai_CC_DataLoader(args.list_path, word_map, args.batch_size, args.num_workers)

    extractor_lr_scheduler = torch.optim.lr_scheduler.StepLR(extractor_optimizer, step_size=1, gamma=0.95)
    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=1, gamma=0.95)
    generator_lr_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=1, gamma=0.95)

    # Start Training
    for epoch in range(start_epoch, args.num_epochs):
        for id, (imgA, imgB, _, _, token, token_len, _) in enumerate(train_dataloader):
            pass




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A lite model for remote sensing change2captioning')

    # LEVIR-CC paramters
    parser.add_argument("gpu_id", type=int, default=0, help='gpu id of the devices')
    parser.add_argument("save_path", type=str, default='./checkpoints', help='path to save checkpoints')
    parser.add_argument("fine_tune", type=bool, default=True, help='fine tune cnn')
    parser.add_argument("n_layer", type=int, default=6, help='number of layers')
    parser.add_argument("d_model", type=int, default=512, help='dimension of model')
    parser.add_argument("n_heads", type=int, default=8, help='number of heads')
    parser.add_argument("vocab_file", type=str, default='word_map', help='vocab file')
    parser.add_argument("list_path", type=str, default='./data/', help='list path')
    parser.add_argument("cnn_lr", type=float, default=1e-4, help='cnn learning rate')
    parser.add_argument("encoder_lr", type=float, default=1e-4, help='encoder learning rate')
    parser.add_argument("decoder_lr", type=float, default=1e-4, help='decoder learning rate')
    parser.add_argument("num_epochs", type=int, default=40, help='number of epochs')



    args = parser.parse_args()
    main(args)