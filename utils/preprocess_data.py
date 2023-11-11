#!/usr/bin/env python
# @Project  : Lite_Chag2cap
# @Time     : 2023/11/11
# @Author   : Sun Dongwei
# @File     : preprocess_data.py

import os
import json


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
