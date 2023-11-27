#!/usr/bin/env python
# @Project  : Lite_Chag2cap
# @Time     : 2023/11/11
# @Author   : Sun Dongwei
# @File     : preprocess_data.py

import os
import json
import argparse

SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<UNK>': 1,
    '<START>': 2,
    '<END>': 3,
}


def main(args):
    global input_vocab_json
    if args.dataset == 'LEVIR_CC':
        input_caption_json = 'data/LEVIR_CC/LevirCCcaptions.json'
        input_image_dir = 'data/LEVIR_CC/images'
        input_vocab_json = ''
        output_vocab_json = 'vocab.json'
        save_dir = 'data/LEVIR_CC/'
    elif args.dataset == 'Dubai_CC':
        input_captions_json = '/root/Data/Dubai_CC/DubaiCC500impair/datasetDubaiCCPublic/description_jsontr_te_val/'
        input_image_dir = '/root/Data/Dubai_CC/DubaiCC500impair/datasetDubaiCCPublic/RGB'
        input_vocab_json = ''
        output_vocab_json = 'vocab.json'
        save_dir = './data/Dubai_CC/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + 'tokens/'):
        os.makedirs(os.path.join(save_dir + 'tokens/'))
    print('--------Loading Captions--------')

    if args.dataset == 'LEVIR_CC':
        with open(input_caption_json, 'r') as f:
            data = json.load(f)
        max_length = -1
        all_cap_tokens = []
        for img in data['images']:
            captions = []
            for cap in img['sentences']:
                assert len(cap['raw']) > 0, 'error: some image has no caption'
                captions.append(cap['raw'])
            tokens_list = []
            for caps in captions:
                cap_tokens = tokenize(caps,
                                      add_start_token=True,
                                      add_end_token=True,
                                      punt_to_keep=[';', ','],
                                      punt_to_remove=['?', '.']
                                      )
                tokens_list.append(cap_tokens)
                max_length = max(max_length, len(cap_tokens))
            all_cap_tokens.append((img['filename'], tokens_list))

        print('--------Saving Captions--------')
        for img, token_list in all_cap_tokens:
            i = img.split('.')[0]
            token_len = len(token_list)
            token_list = json.dumps(token_list)
            f = open(os.path.join(save_dir + 'tokens/', i + '.txt'), 'w')
            f.write(token_list)
            f.close()

            if i.split('_')[0] == 'train':
                f = open(os.path.join(save_dir + 'train_captions' + '.txt'), 'a')
                for j in range(token_len):
                    f.write(img + '-' + str(j) + '\n')
                f.close()
            elif i.split('_')[0] == 'val':
                f = open(os.path.join(save_dir + 'val_captions' + '.txt'), 'a')
                f.write(img + '\n')
                f.close()
            elif i.split('_')[0] == 'test':
                f = open(os.path.join(save_dir + 'test_captions' + '.txt'), 'a')
                f.write(img + '\n')
                f.close()

    print('max_length of the dataset : ', max_length)
    if input_vocab_json == '':
        print('--------Building Vocab--------')
        word_freq = build_vocab(all_cap_tokens, args.word_count_threshold)
    else:
        print('--------Loading Vocab--------')
        with open(input_vocab_json, 'r') as f:
            word_freq = json.load(f)
    if output_vocab_json  is not None:
        print('--------Saving Vocab--------')
        with open(output_vocab_json, 'w') as f:
            json.dump(word_freq, f)


def build_vocab(sequences, min_token_count=1):
    token_to_count = {}

    for seq in sequences:
        for tokens in seq[1]:
            for token in tokens:
                if token not in token_to_count:
                    token_to_count[token] = 0
                token_to_count[token] += 1

    token_to_idx = {}
    for token, inx in SPECIAL_TOKENS.items():
        token_to_idx[token] = inx

    for token, count in sorted(token_to_count.items()):
        if token in token_to_idx.keys():
            continue
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def tokenize(s, delim=' ', add_start_token=True, add_end_token=True,
             punt_to_keep=None, punt_to_remove=None):
    if punt_to_keep is not None:
        for p in punt_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

        if punt_to_remove is not None:
            for p in punt_to_remove:
                s = s.replace(p, '')

        tokens = s.split(delim)
        for q in tokens:
            if q == '':
                tokens.remove(q)
        if tokens[0] == '':
            tokens.remove(tokens[0])
        elif tokens[-1] == '':
            tokens.remove(tokens[-1])

        if add_start_token:
            tokens = ['<START>'] + tokens
        if add_end_token:
            tokens.append('<END>')
        return tokens


def token_encode_process(seq_tokens, token2idx, allow_unknown=False):
    seq_ids = []
    for token in seq_tokens:
        if token not in token2idx:
            if allow_unknown:
                token = '<UNK>'
            else:
                raise KeyError(f'Unknown token {token} in vocab')
        seq_ids.append(token2idx[token])
    return seq_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='LEVIR_CC', help="dataset name")
    parser.add_argument("--word_count_threshold", type=int, default=5, help="word count threshold")

    args = parser.parse_args()
    main(args)
