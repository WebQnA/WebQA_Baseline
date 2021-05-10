from random import randint, shuffle, choices
from random import random as rand
import pickle
import math
import json
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from vlp.loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline

import os
import imghdr
import numpy as np
import sys


def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b


class webqaDataset(torch.utils.data.Dataset):
    """ Load image feature path, q, a """
    def __init__(self, dataset_json_path, split, batch_size, tokenizer, gold_feature_folder, distractor_feature_folder, use_num_samples, processor, device=None):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.instance_list = []
        if device is not None:
            self.device=device
        assert os.path.exists(dataset_json_path), "loader.Dataset: dataset json file doesn't exist!"
        assert os.path.exists(gold_feature_folder), "loader.Dataset: gold feature folder doesn't exist!"
        assert os.path.exists(distractor_feature_folder), "loader.Dataset: distractor feature folder doesn't exist!"
        with open(dataset_json_path, "r") as f:
            dataset_J = json.load(f)

        count = 0
        for i in dataset_J:
            datum = dataset_J[i]
            if datum['split'] in split:
                if use_num_samples == -1 or count < use_num_samples:
                    Q = self.tokenizer.tokenize(datum['Q'])
                    A = self.tokenizer.tokenize(datum['A'])
                    gold_feature_paths = []
                    for im in datum['GoldIds']:
                        image_feature_path = os.path.join(gold_feature_folder, str(im)+'.pkl')
                        assert os.path.exists(image_feature_path), "loader.Dataset: gold image feature for {} doesn't exist!".format(im)
                        gold_feature_paths.append(image_feature_path)
                        self.instance_list.append(([image_feature_path], Q, A, True, False, True)) # schema: ( .pkl file path, Q, A, is_distractor(bool) )
                    for im in datum['DistractorIds']:
                        image_feature_path = os.path.join(distractor_feature_folder, str(im)+'.pkl')
                        if os.path.exists(image_feature_path):
                            self.instance_list.append(([image_feature_path], Q, A, True, True, True)) # schema: ( .pkl file path, Q, A, is_distractor(bool) )
                    self.instance_list.append((gold_feature_paths, Q, A, False, False, True))
                    count += 1

        print("Load {} instances from {} samples".format(len(self.instance_list), count))

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):
        instance = self.instance_list[idx]
        instance = self.processor(instance, self.device)
        # Processor returns:
        # (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, 
        #       -1, is_distractor, self.task_idx, img, vis_pe, context_is_img)
        return instance

    def __iter__(self): # iterator to load data
        for __ in range(math.ceil(len(self.instance_list) / float(self.batch_size))):
            batch = []
            for _ in range(self.batch_size):
                idx = randint(0, len(self.instance_list)-1) # allow overlap between batches???
                batch.append(self.__getitem__(idx))
            yield batch_list_to_batch_tensors(batch)

class Preprocess4webqa(Pipeline):

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len, len_vis_input, max_len_a, max_len_b, new_segment_ids=True, truncate_config={}, local_rank=-1):
        super().__init__()
        self.task_idx = 3 # use task_idx for s2s in relaxed projection layer
        self.max_pred = max_pred
        self.mask_prob = mask_prob
        self.len_vis_input = len_vis_input
        self.vocab_words = vocab_words
        self.indexer = indexer
        
        self._tril_matrix = torch.tril(torch.ones((max_len, max_len), dtype=torch.long))
        self.always_truncate_tail = truncate_config.get('always_truncate_tail', False)
        self.max_len_b = max_len_b
        self.max_len_a = max_len_a
        self.max_len = max_len
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.new_segment_ids = new_segment_ids
        assert max_len_a+max_len_b <= max_len, "loader Processor: max_len_a + max_len_b > max_len"

    def __call__(self, instance, device=None):
        context, Q, A, do_filter_task, is_distractor, context_is_img = instance
        if context_is_img:
            tokens_a = ['[UNK]'] * (self.len_vis_input*len(context))
        else:
            tokens_a = context

        # truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
        tokens_b = Q+A
        truncate_tokens_pair(tokens_a, tokens_b, max_len=self.max_len_a + self.max_len_b, max_len_a=self.max_len_a, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)

        effective_len_a = len(tokens_a)
        # pad tokens_a to max_len_a
        n_pad = self.max_len_a - len(tokens_a)
        tokens_a.extend(['[PAD]'] * n_pad)

        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        
        if self.new_segment_ids:
            segment_ids = [4] * (len(tokens_a)+2) + [5] * (len(tokens_b)+1)
        else:
            segment_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b)+1)

        effective_length = len(A)
        n_pred = min(self.max_pred, max(1, int(round(effective_length * self.mask_prob))))
        cand_pos = []
        for i, tk in enumerate(tokens):
            # only mask tk in A
            if (i >= len(tokens_a)+2+len(Q)) and (tk != '[CLS]'): 
                # 有点问题因为算n_pred时effective_length没有加上末尾的[SEP] 
                # 而且 tk != '[CLS]' 也很匪夷所思
                cand_pos.append(i)
        shuffle(cand_pos)
        masked_pos = cand_pos[:n_pred]
        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:
                tokens[pos] = get_random_word(self.vocab_words)
        
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens) # will be pad to length=max_pred later

        # Token Indexing
        try:
            input_ids = self.indexer(tokens)
        except:
            print("\ntokens = ", tokens)
            print("\ntokens_b = ", tokens_b)
            raise
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # self-attention mask
        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        pred_st, pred_end = len(tokens_a)+2 + len(Q), len(tokens)

        # Everybody can attend to context
        input_mask[:, :effective_len_a].fill_(1)
        # Everybody can attend to Q
        input_mask[:, len(tokens_a)+1:len(tokens_a)+1 + len(Q)].fill_(1)
        # Tokens in A can attend to previous tokens in A
        input_mask[pred_st:pred_end, pred_st:pred_end].copy_(\
            self._tril_matrix[:pred_end-pred_st, :pred_end-pred_st])
        
        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            masked_weights.extend([0] * n_pad)

        if context_is_img:
            # Load img features
            img_list = []
            vis_pe_list = []
            for c in context:
                assert os.path.exists(c), "loader Processor: .pkl file doesn't exist! {}".format(context)
                try:
                    with open(c, "rb") as f:
                        features = pickle.load(f)
                except:
                    print(c)
                    raise
                img = features['fc1_features'].detach().cpu().float()
                cls_label = features['cls_features'].detach().cpu().float()
                vis_pe = features['pred_boxes'].detach().cpu()

                # Lazy normalization of the coordinates
                w_est = torch.max(vis_pe[:, [0, 2]])*1.+1e-5
                h_est = torch.max(vis_pe[:, [1, 3]])*1.+1e-5
                vis_pe[:, [0, 2]] /= w_est
                vis_pe[:, [1, 3]] /= h_est
                assert h_est > 0, 'loader Processor: box h_est should greater than 0! {}'.format(h_est)
                assert w_est > 0, 'loader Processor: box w_est should greater than 0! {}'.format(w_est)
                rel_area = (vis_pe[:, 3]-vis_pe[:, 1])*(vis_pe[:, 2]-vis_pe[:, 0])
                rel_area.clamp_(0)

                vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), features['scores'].detach().cpu().view(-1, 1)), -1)
                normalized_coord = F.normalize(vis_pe.data[:, :5] - 0.5, dim=-1)
                vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), F.layer_norm(cls_label, [1601])), dim=-1)

                img_list.append(img)
                vis_pe_list.append(vis_pe)
            img = torch.cat(img_list, dim=0)
            vis_pe = torch.cat(vis_pe_list, dim=0)
            assert img.size(0) == vis_pe.size(0), "img features and vis_pe should have the same token length!"
            vis_pad = torch.zeros((self.max_len_a - img.size(0), img.size(-1)))#.to(device)
            img = torch.cat((img, vis_pad), dim=0)
            vis_pad = torch.zeros((self.max_len_a - vis_pe.size(0), vis_pe.size(-1)))#.to(device)
            vis_pe = torch.cat((vis_pe, vis_pad), dim=0)
            assert vis_pe.size(0) == self.max_len_a
            assert img.size(0) == self.max_len_a
        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, -1, do_filter_task, is_distractor, self.task_idx, img, vis_pe, context_is_img)


        








