#!/usr/bin/env python
# @Project  : Lite_Chag2cap
# @Time     : 2023/11/14
# @Author   : Sun Dongwei
# @File     : utils.py
import os
import torch

from eval_funtion.bleu.bleu import Bleu
from eval_funtion.meteor.meteor import Meteor
from eval_funtion.rouge.rouge import Rouge
from eval_funtion.cider.cider import Cider


def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hyp = [[' '.join(hyp)] for hyp in [[str(x) for x in hyp] for hyp in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]
    score = []
    method = []
    for scorer, method in scorers:
        score_i, method_i = scorer.compute_score(ref, hyp)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
    score_dict = dict(zip(method, score))

    return score_dict


def accuracy(scores, targets, k):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
