from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append("/home/yingshac/CYS/WebQnA/VLP")

import logging
import glob
import json, time
import argparse
import math
from tqdm import tqdm, trange
from pathlib import Path

import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import random
import pickle
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForWebqaDecoder
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from vlp.loader_utils import batch_list_to_batch_tensors
import vlp.seq2seq_loader as seq2seq_loader
from misc.data_parallel import DataParallelImbalance

import vlp.webqa_loader as webqa_loader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

from datetime import datetime
from pytz import timezone
from word2number import w2n
import re

# SPECIAL_TOKEN = ["[UNK]", "[PAD]", "[CLS]", "[MASK]"]
def toNum(word):
    try: return w2n.word_to_num(word)
    except:
        return word

def normalize_text(s):
    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text): # additional: converting numbers to digit form
        return " ".join([str(toNum(w)) for w in text.split()])
        #return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    if len(s.strip().split()) == 1:
        return white_space_fix(lower(s))

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Language eval with Caption metrics
class Evaluate(object):
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            #(Cider(), "CIDEr"),
            #(Spice(), "Spice")
        ]
    
    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    print(m)
                    final_scores[m] = s
            else:
                print(method)
                final_scores[method] = score
        return final_scores

    def evaluate(self, return_scores=False, **kwargs):
        ans = kwargs.pop('ref', {}) # only support one ans per sample
        cand = kwargs.pop('cand', {}) # only support one cand per sample, but the input cand has size batch_size x K

        hypo = {}
        ref = {}
        i = 0
        for i in range(len(cand)):
            hypo[i] = [cand[i][0]]
            ref[i] = [ans[i]]
        
        final_scores = self.score(ref, hypo)
        print ('Bleu_1:\t', final_scores['Bleu_1'])
        print ('Bleu_2:\t', final_scores['Bleu_2'])
        print ('Bleu_3:\t', final_scores['Bleu_3'])
        print ('Bleu_4:\t', final_scores['Bleu_4'])
        print ('METEOR:\t', final_scores['METEOR'])
        print ('ROUGE_L:', final_scores['ROUGE_L'])
        #print ('CIDEr:\t', final_scores['CIDEr'])
        #print ('Spice:\t', final_scores['Spice'])

        if return_scores:
            return final_scores

# BertScore
from datasets import load_metric
METRIC = load_metric("bertscore")
def compute_bertscore(cands, a):
    METRIC.add_batch(predictions = cands, references = [a]*len(cands))
    score = METRIC.compute(lang='en')
    return np.mean(score['f1']), np.max(score['f1'])

# VQA Eval (SQuAD style EM, F1)
def compute_vqa_metrics(cands, a):
    if len(cands) == 0: return (0,0,0)
    bow_a = normalize_text(a).split()
    #bow_a = [str(toNum(w)) for w in bow_a]
    F1 = []
    EM = 0
    for c in cands:
        bow_c = normalize_text(c).split()
        #bow_c = [str(toNum(w)) for w in bow_c]
        if bow_c == bow_a:
            EM = 1
        #print(bow_a, bow_c)
        overlap = float(len([w for w in bow_c if w in bow_a]))
        precision = overlap/(len(bow_c) + 1e-5)
        recall = overlap / (len(bow_a) + 1e-5)
        f1 = 2*precision*recall / (precision + recall + 1e-5)
        F1.append(f1)
    
    F1_avg = np.mean(F1)
    F1_max = np.max(F1)
    return (F1_avg, F1_max, EM)

answer_mod = 'img'
human_img_json = json.load(open("/home/yingshac/CYS/WebQnA/WebQnA_data_new/human_{}.json".format(answer_mod), "r"))
C = [[str(datum['CuratedEvalAnswer'])] for datum in human_img_json]
A = [str(datum['A']).replace('"', "") for datum in human_img_json]
F1_avg_scores = []
F1_max_scores = []
EM_scores = []
#pr_scores = []
#re_scores = []
F1_avg_bertscores = []
F1_max_bertscores = []
for cands, a in zip(C, A):
    assert len(cands)==1
    #cands=["yes", "no"]
    #cands = [cands[0]]
    F1_avg, F1_max, EM= compute_vqa_metrics(cands, a)
    F1_avg_scores.append(F1_avg)
    F1_max_scores.append(F1_max)
    EM_scores.append(EM)
    
    #pr_scores.append(pr)
    #re_scores.append(re)
    
    F1_avg_bertscore, F1_max_bertscore = compute_bertscore(cands, a)
    F1_avg_bertscores.append(F1_avg_bertscore)
    F1_max_bertscores.append(F1_max_bertscore)
F1_avg = np.mean(F1_avg_scores)
F1_max = np.mean(F1_max_scores)
EM = np.mean(EM_scores)
F1_avg_bertscore = np.mean(F1_avg_bertscores)
F1_max_bertscore = np.mean(F1_max_bertscores)
print("F1_avg = {}".format(F1_avg))
print("F1_max = {}".format(F1_max))
print("EM = {}".format(EM))
print("F1_avg_bertscore = {}".format(F1_avg_bertscore))
print("F1_max_bertscore = {}".format(F1_max_bertscore))

eval_f = Evaluate()
scores = eval_f.evaluate(cand=C, ref=A, return_scores=True)

with open("/home/yingshac/CYS/WebQnA/VLP/vlp/tmp/Human/scores_{}.txt".format(answer_mod), "w") as f:
    f.write(datetime.now(tz=timezone('US/Eastern')).strftime("%y-%m-%d %H:%M:%S") + '\n')
    f.write('\n --------------------- metrics -----------------------\n')
    f.write(str(scores))
    f.write('\n\n')
    f.write('\n'.join(["F1_avg = {}".format(F1_avg), "EM = {}".format(EM)]))
    f.write('\n\n')
    f.write('\n'.join(["F1_avg_bertscore = {}".format(F1_avg_bertscore)]))
