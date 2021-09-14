from random import randint, shuffle
from random import random as rand
import pickle
import json
from collections import namedtuple
import torch
import torch.nn as nn
import unicodedata
from multiprocessing import Lock


def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]


def batch_list_to_batch_tensors(batch):

    batch_tensors = []
    for x in zip(*batch):
        if all(y is None for y in x):
            batch_tensors.append(torch.zeros(1))
            #batch_tensors.append(None)
        elif any(y is None for y in x):
            z = [y for y in x if y is not None]
            batch_tensors.append(torch.cat(z, dim=0))
        elif isinstance(x[0], list) and (len(x[0])==0 or isinstance(x[0][0], int)): # cxt_modality_label
            batch_tensors.append(x)

        elif isinstance(x[0], torch.Tensor):
            try:
                batch_tensors.append(torch.stack(x))
            except:
                #print([i.size() for i in x])
                f = torch.cat(x, dim=0)
                #print("After torch.cat vis_feats, size = ", f.size())
                batch_tensors.append(f)
        elif isinstance(x[0], str):
            batch_tensors.append(x)
        
        else:
            try:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))
            except:
                batch_tensors.append(x)

    return batch_tensors


class Pipeline():
    """ Pre-process Pipeline Class : callable """

    def __init__(self):
        super().__init__()
        self.mask_same_word = None
        self.skipgram_prb = None
        self.skipgram_size = None

    def __call__(self, instance):
        raise NotImplementedError
