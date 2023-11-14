#!/usr/bin/env python
# @Project  : Lite_Chag2cap
# @Time     : 2023/11/14
# @Author   : Sun Dongwei
# @File     : utils.py

def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hyp = [ [str(x) for x in hyp] for hyp in hypotheses]

def accuracy(scores, targets, k):
    pass