#!/usr/bin/env python
# @Project  : Lite_Chag2cap
# @Time     : 2023/12/17
# @Author   : Sun Dongwei
# @File     : caption.py

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import json
from imageio.v3 import imread

from models.CNN_Nets import Con_Net
from models.AdjustLength_net import CC_Trans
from models.model_decoder import Decoder_Generator

from torchcam.methods import SmoothGradCAMpp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_captions(args, word_map, hypotheses):
    result_json_file = {}
    kkk = -1
    for item in hypotheses:
        kkk += 1
        line_hypo = " ".join([get_key(word_map, word_idx)[0] for word_idx in item])
        result_json_file[str(kkk)] = [line_hypo]

    print(result_json_file)

    with open(os.path.join('eval_results', f'{args.extractor}_{args.encoder_feat}_{args.decoder}.json'), 'w') as f:
        json.dump(result_json_file, f)


def get_key(dict_, value):
    return [k for k, v in dict_.items() if v == value]


def main(args):
    l_resize1 = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    l_resize2 = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    word_map_file = os.path.join(args.data_folder, args.vocab_file + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    checkpoints = torch.load(args.path, map_location='cuda:0')
    if args.state_only:
        # Load Model
        extractor = Con_Net(args.network)

        encoder = CC_Trans(n_layers=args.n_layers, feature_size=[args.feat_size, args.feat_size, args.encoder_dim])

        decoder = Decoder_Generator(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_map),
                                    max_lengths=args.max_length, word_vocab=word_map, n_head=args.n_heads,
                                    n_layers=args.decoder_n_layers, dropout=args.dropout)

        extractor.load_state_dict(checkpoints['extractor_dict'])
        extractor.cuda()
        encoder.load_state_dict(checkpoints['encoder_dict'])
        encoder.cuda()
        decoder.load_state_dict(checkpoints['generator_dict'])
        decoder.cuda()
    else:
        extractor = checkpoints['extractor']
        encoder = checkpoints['encoder']
        decoder = checkpoints['generator']

    extractor.eval()
    decoder.eval()
    decoder.eval()

    hypotheses = []

    with torch.no_grad():
        if args.data_name == 'LEVIR_CC':
            img_A = imread(args.img_path_A)
            img_A = img_A.transpose(2, 0, 1)
            img_A = img_A / 255.
            img_A = torch.FloatTensor(img_A).cuda()

            img_B = imread(args.img_path_B)
            img_B = img_B.transpose(2, 0, 1)
            img_B = img_B / 255.
            img_B = torch.FloatTensor(img_B).cuda()

            normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
            transform = transforms.Compose([normalize])

            img_A = transform(img_A)
            img_A = img_A.unsqueeze(0)

            img_B = transform(img_B)
            img_B = img_B.unsqueeze(0)
            # start reference
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            feat_A, feat_B = extractor(img_A, img_B)
            feat_A, feat_B = encoder(feat_A, feat_B)  # (1, 2048, 8, 8)



            seq = decoder.sample1(feat_A, feat_B)
            pred_seq = [w for w in seq if w not in {word_map['<START>'], word_map['<END>'], word_map['<NULL>']}]
            hypotheses.append(pred_seq)

            memory_allocated = torch.cuda.memory_allocated()
            peak_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"GPU 显存占用：{memory_allocated / 1024 / 1024:.2f} MB")
            print(f"GPU 显存峰值占用：{peak_memory_allocated / 1024 / 1024:.2f} MB")
            # finish reference

        elif args.data_name == 'Dubai_CC':
            img_A = imread(args.img_path_A)
            img_A = img_A.transpose(2, 0, 1)
            img_A = torch.FloatTensor(img_A).cuda()
            img_A = img_A.unsqueeze(0)
            img_A = l_resize1(img_A)
            img_A = img_A / 255.

            img_B = imread(args.img_path_B)
            img_B = img_B.transpose(2, 0, 1)
            img_B = torch.FloatTensor(img_B).cuda()
            img_B = img_B.unsqueeze(0)
            img_B = l_resize1(img_B)
            img_B = img_B / 255.

            normal = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
            transform = transforms.Compose([normal])
            # normal = transforms.Compose([transforms.Normalize(mean=[-6.9173e-07, -1.3428e-06, -1.1393e-06],
            #                                                   std=[1.0000, 1.0000, 1.0000])])
            # transform = transforms.Compose([normal])
            img_A = transform(img_A)
            img_B = transform(img_B)

            # start reference
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            feat_A, feat_B = extractor(img_A, img_B)
            feat_A, feat_B = encoder(feat_A, feat_B)
            seq = decoder.sample1(feat_A, feat_B)

            pred_seq = [w for w in seq if w not in {word_map['<START>'], word_map['<END>'], word_map['<NULL>']}]
            hypotheses.append(pred_seq)

            memory_allocated = torch.cuda.memory_allocated()
            peak_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"GPU 显存占用：{memory_allocated / 1024 / 1024:.2f} MB")
            print(f"GPU 显存峰值占用：{peak_memory_allocated / 1024 / 1024:.2f} MB")
            # finish reference
        save_captions(args, word_map, hypotheses)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote sensing change caption inference demo')
    parser.add_argument('--path', default="/home/sdw/paper_projects/Lite_Chag2cap/checkpoints/LEVIR_CC_BatchSize_32_resnet101_Bleu-4_6414.pth",
                        help='model checkpoint')
    parser.add_argument('--encoder_feat', default='sparse_CCNet')
    parser.add_argument('--decoder', default='generator', help='decoder to img2cap')
    parser.add_argument('--data_folder', default='./data/LEVIR_CC/')
    parser.add_argument('--vocab_file', default='vocab', help='epoch to test')
    parser.add_argument('--beam_size', default='beam_size')
    parser.add_argument('--img_path_A', default='/home/sdw/Desktop/test_file/input_path/Change_caption/image1/crop_A_3_3.png')
    parser.add_argument('--img_path_B', default='/home/sdw/Desktop/test_file/input_path/Change_caption/image2/crop_B_3_3.png')
    parser.add_argument('--extractor', default='resnet101')
    parser.add_argument('--epoch', default="epoch", help='epoch')
    parser.add_argument('--network', default='resnet101')
    parser.add_argument("--n_layers", type=int, default=1, help='number of layers')
    parser.add_argument("--encoder_dim", type=int, default=2048, help='the dim of extracted features by diff nets')
    parser.add_argument("--feat_size", type=int, default=16)
    parser.add_argument("--decoder_n_layers", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=41, help='max length of each caption sentence')
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--n_heads", type=int, default=8, help='number of heads')
    parser.add_argument("--dropout", type=float, default=0.1, help='dropout')
    parser.add_argument('--state_only', type=bool, default=False, help='whether to save model.state_dict only'
                                                                       'when training')
    parser.add_argument('--data_name', type=str, default='LEVIR_CC')

    args = parser.parse_args()

    main(args)

