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
    # previous schema: (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next_label, do_filter_task, filter_label, logit_mask, ori_choices, task_idx, img, vis_pe, context, cxt_modality_label, example_id)
    #          schema: (input_ids, segment_ids, position_ids, input_mask, ----------------------------------------------------------------------------------------------- task_idx, img, vis_pe, context, cxt_modality_label, example_id)
    batch_size = len(batch)
    keys = set([key for b in batch for key in b.keys()])
    if len(keys) == 17:
        keys =  ["input_ids", "segment_ids", "input_mask", "masked_ids", "masked_pos", "masked_weights", "is_next_label", "do_filter_task", "filter_label", "logit_mask", "ori_choices", "task_idx", "img", "vis_pe", "context", "cxt_modality_label", "example_id"]
    else:
        keys =  ["input_ids", "segment_ids", "position_ids", "input_mask", "task_idx", "img", "vis_pe", "context", "cxt_modality_label", "example_id"]
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
    
    for k in keys:
        x = dict_batch[k]
        if all(y is None for y in x): # cuz you have to pass a tensor to .to(device)
            batch_tensors.append(torch.zeros(1))
            #print("1 ", k)
        elif any(y is None for y in x): # deprecated: no input enter this branch
            z = [y for y in x if y is not None]
            batch_tensors.append(torch.cat(z, dim=0))
            #print("2 ", k)
        elif isinstance(x[0], list) and (len(x[0])==0 or isinstance(x[0][0], int)): # cxt_modality_label
            batch_tensors.append(x)
            #print("3 ", k)

        elif isinstance(x[0], torch.Tensor):
            try:
                batch_tensors.append(torch.stack(x))
                #print("4 ", k)
            except:
                f = torch.cat(x, dim=0)
                #print("After torch.cat vis_feats, size = ", f.size())
                batch_tensors.append(f)
                #print("5 ", k)
        elif isinstance(x[0], str): # context, example_id
            batch_tensors.append(x)
            #print("6 ", k)
        
        else:
            try:
                batch_tensors.append(torch.tensor(x, dtype=torch.long)) # do_filter_task, is_next_label, task_idx
                #print("7 ", k)
            except:
                batch_tensors.append(x) # ori_choices
                #print("8 ", k)
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
