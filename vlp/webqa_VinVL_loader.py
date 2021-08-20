import random
from random import randint, shuffle, choices
from random import random as rand
import pickle
import math, time
import json
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from vlp.loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline
from vlp.ImgDataTsv import ImgDataTsv

import os
import imghdr
import numpy as np
import sys

def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) <= max_len_a and len(tokens_b) <= max_len_b:
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

class webqaDataset_filter(torch.utils.data.Dataset):
    """ Load image feature path, q, a """
    def __init__(self, dataset_json_path, split, Qcate, batch_size, tokenizer, use_num_samples, processor, filter_max_choices=10, device=None):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.filter_max_choices = filter_max_choices
        self.instance_list = []
        if device is not None:
            self.device=device
        assert os.path.exists(dataset_json_path), "loader.Dataset: dataset json file doesn't exist! {}".format(dataset_json_path)
        with open(dataset_json_path, "r") as f:
            dataset_J = json.load(f)
        count = 0
        for i in dataset_J:
            datum = dataset_J[i]
            if datum['Split'] in split: # modify here after we create split!
                if ('all' in Qcate) or datum['Qcate'] in Qcate:
                    if use_num_samples == -1 or count < use_num_samples:
                        Q = self.tokenizer.tokenize(datum['Q'])
                        A = self.tokenizer.tokenize(datum['A'])
                        gold_facts = []
                        distractor_facts = []
                        for fa in datum['txt_posFacts']:
                            gold_facts.append(self.tokenizer.tokenize(fa['fact']))

                        for fa in datum['txt_negFacts']:
                            distractor_facts.append(self.tokenizer.tokenize(fa['fact']))
                        shuffle(gold_facts)
                        shuffle(distractor_facts)
                        self.instance_list.append((gold_facts, distractor_facts, [], [], Q, A, True, "txt", i)) # do_filter_task, context
                        
                        count += 1

        print("Load {} instances from {} samples".format(len(self.instance_list), count))

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):
        gold_facts, distractor_facts, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id = self.instance_list[idx]
        
        sample_size = self.filter_max_choices - len(gold_facts)
        
        if len(distractor_facts) < sample_size: sample_size = len(distractor_facts)
        distractor_facts = distractor_facts[:sample_size]
        instance = (gold_facts, distractor_facts, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id)
        instance = self.processor(instance, self.filter_max_choices, self.device)

        return instance

    def __iter__(self): # iterator to load data
        for __ in range(math.ceil(len(self.instance_list) / float(self.batch_size))):
            batch = []
            for _ in range(self.batch_size):
                idx = randint(0, len(self.instance_list)-1) # allow overlap between batches???
                batch.append(self.__getitem__(idx))
            yield batch_list_to_batch_tensors(batch)

class webqaDataset_qa(torch.utils.data.Dataset):
    """ Load image feature path, q, a """
    def __init__(self, dataset_json_path, split, Qcate, batch_size, tokenizer, use_num_samples, processor, device=None):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.instance_list = []
        if device is not None:
            self.device=device
        assert os.path.exists(dataset_json_path), "loader.Dataset: dataset json file doesn't exist! {}".format(dataset_json_path)
        with open(dataset_json_path, "r") as f:
            dataset_J = json.load(f)
        count = 0
        for i in dataset_J:
            datum = dataset_J[i]
            if datum['Split'] in split: # modify here after we have split!!!!
                if ('all' in Qcate) or datum['Qcate'] in Qcate:
                    if use_num_samples == -1 or count < use_num_samples:
                        Q = self.tokenizer.tokenize(datum['Q'].replace('"', ""))
                        A = self.tokenizer.tokenize(datum['A'].replace('"', ""))
                        gold_facts = []
                        distractor_facts = []
                        for fa in datum['txt_posFacts']:
                            gold_facts.append(self.tokenizer.tokenize(fa['fact']))

                        self.instance_list.append((gold_facts, [], [], [], Q, A, False, "txt", i)) # do_filter_task, context
                        
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

    def get_QA_list(self):
        return [i[4] for i in self.instance_list], [i[5] for i in self.instance_list]

class webqaDataset_filter_with_img(torch.utils.data.Dataset):
    """ Load image feature path, q, a """
    def __init__(self, dataset_json_path, split, Qcate, batch_size, tokenizer, use_num_samples, processor, filter_max_choices=10, device=None):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.filter_max_choices = filter_max_choices
        self.instance_list = []
        if device is not None:
            self.device=device
        assert os.path.exists(dataset_json_path), "loader.Dataset: dataset json file doesn't exist!"
        #assert os.path.exists(img_metadata_path), "loader.Dataset: img metadata json file doesn't exist!"
        with open(dataset_json_path, "r") as f:
            dataset_J = json.load(f)
        #with open(img_metadata_path, "r") as f:
            #img_meta = json.load(f)
        count = 0
        for i in dataset_J:
            datum = dataset_J[i]
            if datum['split'] in split:
                if ('all' in Qcate) or datum['Qcate'] in Qcate:
                    if use_num_samples == -1 or count < use_num_samples:
                        Q = self.tokenizer.tokenize(datum['Q'])
                        A = self.tokenizer.tokenize(datum['A'])

                        gold_img_and_caps = []
                        distractor_img_and_caps = []

                        for im in datum['img_posFacts']:
                            image_id = int(im['image_id'])
                            cxt = self.tokenizer.tokenize(im['caption'].strip())
                            gold_img_and_caps.append((image_id, cxt))

                        for im in datum['img_negFacts']:
                            image_id = int(im['image_id'])
                            cxt = self.tokenizer.tokenize(im['caption'].strip())
                            distractor_img_and_caps.append((image_id, cxt))
                            
                        shuffle(gold_img_and_caps)
                        shuffle(distractor_img_and_caps)

                        gold_image_ids = [x[0] for x in gold_img_and_caps]
                        distractor_image_ids = [x[0] for x in distractor_img_and_caps]
                        gold_cxt_list = [x[1] for x in gold_img_and_caps]
                        distractor_cxt_list = [x[1] for x in distractor_img_and_caps]
                        
                        self.instance_list.append((gold_image_ids, distractor_image_ids, gold_cxt_list, distractor_cxt_list, Q, A, True, "img", i)) # do_filter_task, context
                        
                        count += 1

        print("Load {} instances from {} samples".format(len(self.instance_list), count))

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):
        gold_image_ids, distractor_image_ids, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id = self.instance_list[idx]
        assert len(distractor_cxt_list) == len(distractor_image_ids)
        assert len(gold_cxt_list) == len(gold_image_ids)
        sample_size = self.filter_max_choices - len(gold_image_ids)
        if len(distractor_image_ids) < sample_size: sample_size = len(distractor_image_ids)
        distractor_image_ids = distractor_image_ids[:sample_size]
        distractor_cxt_list = distractor_cxt_list[:sample_size]

        instance = (gold_image_ids, distractor_image_ids, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id)
        instance = self.processor(instance, self.filter_max_choices, self.device)
        
        return instance

    def __iter__(self): # iterator to load data
        for __ in range(math.ceil(len(self.instance_list) / float(self.batch_size))):
            batch = []
            for _ in range(self.batch_size):
                idx = randint(0, len(self.instance_list)-1) # allow overlap between batches???
                batch.append(self.__getitem__(idx))
            yield batch_list_to_batch_tensors(batch)

class webqaDataset_qa_with_img(torch.utils.data.Dataset):
    """ Load image feature path, q, a """
    def __init__(self, dataset_json_path, split, Qcate, batch_size, tokenizer, use_num_samples, processor, device=None):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.instance_list = []
        if device is not None:
            self.device=device
        assert os.path.exists(dataset_json_path), "loader.Dataset: dataset json file doesn't exist!"
        #assert os.path.exists(img_metadata_path), "loader.Dataset: img metadata json file doesn't exist!"
        with open(dataset_json_path, "r") as f:
            dataset_J = json.load(f)
        #with open(img_metadata_path, "r") as f:
            #img_meta = json.load(f)
        count = 0
        for i in dataset_J:
            datum = dataset_J[i]
            if datum['split'] in split:
                if ('all' in Qcate) or datum['Qcate'] in Qcate:
                    if use_num_samples == -1 or count < use_num_samples:
                        Q = self.tokenizer.tokenize(datum['Q'].replace('"', ""))
                        A = self.tokenizer.tokenize(datum['A'].replace('"', ""))
                        gold_image_ids = []
                        gold_cxt_list = []
                        for im in datum['img_posFacts']:
                            image_id = int(im['image_id'])
                            gold_image_ids.append(image_id)
                            cxt = self.tokenizer.tokenize(im['caption'].strip())
                            gold_cxt_list.append(cxt)
                        self.instance_list.append((gold_image_ids, [], gold_cxt_list, [], Q, A, False, "img", i)) # do_filter_task, context )
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

    def get_QA_list(self):
        return [i[4] for i in self.instance_list], [i[5] for i in self.instance_list]

class webqaDataset_filter_with_both(torch.utils.data.Dataset):
    ## TODO: define a new Dataset, return img+cap in a tuple instead of two separate lists
    """ Load image feature path, q, a """
    def __init__(self, dataset_json_path, split, Qcate, batch_size, tokenizer, use_num_samples, processor, answer_provided_by, max_snippets=10, max_imgs=10, device=None):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.answer_provided_by = answer_provided_by
        self.max_snippets = max_snippets
        self.max_imgs = max_imgs
        self.instance_list = []
        if device is not None:
            self.device=device
        assert os.path.exists(dataset_json_path), "loader.Dataset: dataset json file doesn't exist!"
        #assert os.path.exists(img_metadata_path), "loader.Dataset: img metadata json file doesn't exist!"
        with open(dataset_json_path, "r") as f:
            dataset_J = json.load(f)
        #with open(img_metadata_path, "r") as f:
            #img_meta = json.load(f)
        count = 0
        for i in dataset_J:
            datum = dataset_J[i]
            if datum['split'] in split:
                if ('all' in Qcate) or datum['Qcate'] in Qcate:
                    if use_num_samples == -1 or count < use_num_samples:
                        Q = self.tokenizer.tokenize(datum['Q'])
                        A = self.tokenizer.tokenize(datum['A'])

                        gold_facts = []
                        distractor_facts = []

                        if 'txt_posFacts' in datum:
                            for fa in datum['txt_posFacts']: gold_facts.append(self.tokenizer.tokenize(fa['fact']))
                        for fa in datum['txt_negFacts']:
                            distractor_facts.append(self.tokenizer.tokenize(fa['fact']))
                        shuffle(gold_facts)
                        shuffle(distractor_facts)

                        gold_img_and_caps = []
                        distractor_img_and_caps = []

                        if 'img_posFacts' in datum:
                            for im in datum['img_posFacts']:
                                image_id = int(im['image_id'])
                                cxt = self.tokenizer.tokenize(im['caption'].strip())
                                gold_img_and_caps.append((image_id, cxt))

                        for im in datum['img_negFacts']:
                            image_id = int(im['image_id'])
                            cxt = self.tokenizer.tokenize(im['caption'].strip())
                            distractor_img_and_caps.append((image_id, cxt))
                            
                        shuffle(gold_img_and_caps)
                        shuffle(distractor_img_and_caps)

                        self.instance_list.append((gold_facts, distractor_facts, gold_img_and_caps, distractor_img_and_caps, Q, A, True, "both", i)) # do_filter_task, context
                        
                        count += 1

        print("Load {} instances from {} samples".format(len(self.instance_list), count))

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):
        gold_facts, distractor_facts, gold_img_and_caps, distractor_img_and_caps, Q, A, do_filter_task, context, example_id = self.instance_list[idx]
        
        if self.answer_provided_by == 'img':
            sample_size = self.max_imgs - len(gold_img_and_caps)
            if len(distractor_img_and_caps) < sample_size: sample_size = len(distractor_img_and_caps)
            distractor_img_and_caps = distractor_img_and_caps[:sample_size]
            distractor_facts = distractor_facts[:self.max_snippets]
        elif self.answer_provided_by == 'txt':
            sample_size = self.max_snippets - len(gold_facts)
            if len(distractor_facts) < sample_size: sample_size = len(distractor_facts)
            distractor_facts = distractor_facts[:sample_size]
            distractor_img_and_caps = distractor_img_and_caps[:self.max_imgs]
        else:
            raise ValueError("Invalid answer modality. args.answer_provided_by must be one of {'img', 'txt'}")
        
        instance = (gold_facts, distractor_facts, gold_img_and_caps, distractor_img_and_caps, Q, A, do_filter_task, context, example_id)
        instance = self.processor(instance, self.max_imgs + self.max_snippets, self.device)
        
        return instance

    def __iter__(self): # iterator to load data
        for __ in range(math.ceil(len(self.instance_list) / float(self.batch_size))):
            batch = []
            for _ in range(self.batch_size):
                idx = randint(0, len(self.instance_list)-1) # allow overlap between batches???
                batch.append(self.__getitem__(idx))
            yield batch_list_to_batch_tensors(batch)


class Preprocess4webqa_VinVL(Pipeline):

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, seed, max_len, len_vis_input, max_len_a, max_len_b, max_len_img_cxt=200, new_segment_ids=True, truncate_config={}, use_img_meta=True, use_img_content=True, use_txt_fact=True, ImgDataTsv_dict=None):
        super().__init__()
        self.task_idx = 3 # use task_idx for s2s in relaxed projection layer
        self.max_pred = max_pred
        self.mask_prob = mask_prob
        self.len_vis_input = len_vis_input
        self.vocab_words = vocab_words
        self.indexer = indexer
        self.max_len_img_cxt = max_len_img_cxt
        self._tril_matrix = torch.tril(torch.ones((max_len, max_len), dtype=torch.long))
        self.always_truncate_tail = truncate_config.get('always_truncate_tail', False)
        self.max_len_b = max_len_b
        self.max_len_a = max_len_a
        self.max_len = max_len
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.new_segment_ids = new_segment_ids
        self.use_img_meta = use_img_meta
        self.use_img_content = use_img_content
        self.use_txt_fact = use_txt_fact

        random.seed(seed)
        np.random.seed(seed)
        assert max_len_a+max_len_b <= max_len, "loader Processor: max_len_a + max_len_b > max_len"

        self.img_data_tsv = {}
        for k in ImgDataTsv_dict:
            self.img_data_tsv[k] = ImgDataTsv(ImgDataTsv_dict[k])

    def detokenize(self, tk_list):
        r_list = []
        for tk in tk_list:
            if tk.startswith('##') and len(r_list) > 0:
                r_list[-1] = r_list[-1] + tk[2:]
            else:
                r_list.append(tk)
        return r_list

    def __call__(self, instance, filter_max_choices=None, device=None):
        _, __, ___, ____, _____, ______, do_filter_task, context, example_id = instance
        if do_filter_task:
            assert filter_max_choices is not None, "must pass in a valid filter_max_choices when doing filter task"
            if context == 'both':
                gold_facts, distractor_facts, gold_img_and_caps, distractor_img_and_caps, Q, A, do_filter_task, context, example_id = instance
                ## TODO: define a new Dataset, return img+cap in a tuple instead of two separate lists
                '''
                0: pos snippet
                1: pos img
                2: neg snippet
                3: neg img
                '''
                order = [0] * len(gold_facts) + [1] * len(gold_img_and_caps) + [2] * len(distractor_facts) + [3] * len(distractor_img_and_caps)
                order = np.random.permutation(order)
                label = torch.tensor([1. if o<=1 else 0. for o in order])
                label = torch.stack([label, 1-label], dim=0).transpose(1,0)
                ori_choices = []

                input_ids_list = []
                segment_ids_list = []
                input_mask_list = []
                img_list = []
                vis_pe_list = []

                for o in order:
                    if o%2 == 1: # context is img
                        if o == 1: # pos img
                            image_id, cxt = gold_img_and_caps.pop()
                        else: # neg img
                            image_id, cxt = distractor_img_and_caps.pop()
                        ori_choices.append(image_id)

                        tokens_a = ['[UNK]'] * self.max_len_img_cxt # 200
                        tokens_b = Q+A
                        max_len_cxt_meta = self.max_len_a - self.max_len_img_cxt # 200
                        truncate_tokens_pair(cxt, tokens_b, max_len=max_len_cxt_meta + self.max_len_b, max_len_a=max_len_cxt_meta, max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)
                        if self.use_img_meta: tokens_a += cxt
                        # it seems that there is no need to pad cxt_meta to 200
                        #n_pad = self.max_len_a+1 - len(tokens_a) # +1 for the middle SEP
                        #tokens_a.extend(['[PAD]'] * n_pad)
                        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
                        if self.new_segment_ids:
                            segment_ids = [4] * (len(tokens_a)+2) + [5] * (len(tokens_b)+1)
                        else:
                            segment_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b)+1)


                        # self-attention mask
                        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                        # everyone can attend to img, cxt_meta and Q. Nobody cares attention to A for filter task
                        img_end_pos = 1+self.len_vis_input
                        if self.use_img_content: input_mask[:, :img_end_pos].fill_(1)
                        st, end = 1 + self.max_len_img_cxt, len(tokens_a) + 2 + len(Q)
                        input_mask[:, st:end].fill_(1)
                        #st, end = 2 + self.max_len_a, 2 + self.max_len_a + len(Q)
                        #input_mask[:, st:end].fill_(1)
                        input_ids = self.indexer(tokens)
                        n_pad = self.max_len - len(input_ids)
                        input_ids.extend([0] * n_pad)
                        segment_ids.extend([0] * n_pad)

                        vis_pe, scores, img, cls_label = self.img_data_tsv[image_id//10000000][image_id % 10000000]
                        
                        #img = features['fc1_features'].detach().cpu().float()
                        #cls_label = features['cls_features'].detach().cpu().float()
                        #vis_pe = features['pred_boxes'].detach().cpu()

                        # Lazy normalization of the coordinates
                        w_est = torch.max(vis_pe[:, [0, 2]])*1.+1e-5
                        h_est = torch.max(vis_pe[:, [1, 3]])*1.+1e-5
                        vis_pe[:, [0, 2]] /= w_est
                        vis_pe[:, [1, 3]] /= h_est
                        assert h_est > 0, 'loader Processor: box h_est should greater than 0! {}'.format(h_est)
                        assert w_est > 0, 'loader Processor: box w_est should greater than 0! {}'.format(w_est)
                        rel_area = (vis_pe[:, 3]-vis_pe[:, 1])*(vis_pe[:, 2]-vis_pe[:, 0])
                        rel_area.clamp_(0)

                        #vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), features['scores'].detach().cpu().view(-1, 1)), -1)
                        vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), scores.view(-1, 1)), -1)
                        normalized_coord = F.normalize(vis_pe.data[:, :5] - 0.5, dim=-1)
                        vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), F.layer_norm(cls_label, [1595])), dim=-1) # prev vg: 1601, VinVL: 1595

                        assert img.size(0) == vis_pe.size(0), "img features and vis_pe should have the same token length!"
                        vis_pad = torch.zeros((self.max_len_img_cxt - img.size(0), img.size(-1)))
                        img = torch.cat((img, vis_pad), dim=0) 
                        pe_pad = torch.zeros((self.max_len_img_cxt - vis_pe.size(0), vis_pe.size(-1)))
                        vis_pe = torch.cat((vis_pe, pe_pad), dim=0)
                        assert vis_pe.size(0) == self.max_len_img_cxt
                        assert img.size(0) == self.max_len_img_cxt
                        input_ids_list.append(torch.tensor(input_ids))
                        segment_ids_list.append(torch.tensor(segment_ids))
                        input_mask_list.append(input_mask)
                        if not self.use_img_content: 
                            img = torch.zeros_like(img).float()
                            vis_pe = torch.zeros_like(vis_pe).float()
                        img_list.append(img)
                        vis_pe_list.append(vis_pe)

                    else: # context is snippet
                        tokens_a = []
                        if self.use_txt_fact:
                            if o == 0: # pos snippet
                                tokens_a = gold_facts.pop()
                            else: # neg snippet
                                tokens_a = distractor_facts.pop()
                        ori_choices.append(' '.join(self.detokenize(tokens_a)))
                        
                        tokens_b = Q+A
                        truncate_tokens_pair(tokens_a, tokens_b, max_len=self.max_len_a+self.max_len_b, max_len_a=self.max_len_a, max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)
                        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

                        if self.new_segment_ids:
                            segment_ids = [4] * (len(tokens_a)+2) + [5] * (len(tokens_b)+1)
                        else:
                            segment_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b)+1)

                        # self-attention mask
                        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                        # everyone can attend to cxt and Q. Nobody cares attention to A for filter task
                        input_mask[:, :len(tokens_a)+2+len(Q)].fill_(1)

                        input_ids = self.indexer(tokens)
                        n_pad = self.max_len - len(input_ids)
                        input_ids.extend([0] * n_pad)
                        segment_ids.extend([0] * n_pad)

                        input_ids_list.append(torch.tensor(input_ids))
                        segment_ids_list.append(torch.tensor(segment_ids))
                        input_mask_list.append(input_mask)
                    
                logit_mask = [1.] * len(input_ids_list)
                if len(input_ids_list) < filter_max_choices:
                    num_placeholder = filter_max_choices - len(input_ids_list)
                    input_ids_list.extend([input_ids_list[-1]] * num_placeholder)
                    segment_ids_list.extend([segment_ids_list[-1]] * num_placeholder)
                    input_mask_list.extend([input_mask_list[-1]] * num_placeholder)
                    logit_mask.extend([0.] * num_placeholder)
                    label = torch.cat([label, torch.tensor([[0., 0.]] * num_placeholder)], dim=0)
                input_ids = torch.stack(input_ids_list, dim=0) 
                segment_ids = torch.stack(segment_ids_list, dim=0)
                input_mask = torch.stack(input_mask_list, dim=0)
                assert len(img_list) == len(vis_pe_list)
                if len(img_list) == 0:
                    img = None
                    vis_pe = None
                else:
                    img = torch.stack(img_list, dim=0)
                    vis_pe = torch.stack(vis_pe_list, dim=0)
                logit_mask = torch.tensor(logit_mask)
                
                cxt_modality_label = [i for i in range(len(order)) if order[i]%2 == 1]

                ## TODO: for the other two cases in the loader, modify schema of return values to include cxt_modality_label & change context_is_img --> context
                # schema: (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next_label, do_filter_task, filter_label, logit_mask, ori_choices, self.task_idx, img, vis_pe, context, cxt_modality_label, example_id)
                return (input_ids, segment_ids, input_mask,       None,        None,        None,         -1,         do_filter_task,        label, logit_mask, ori_choices, self.task_idx, img, vis_pe, context, cxt_modality_label, example_id) 
                

            elif context == 'img':
                gold_image_ids, distractor_image_ids, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id = instance
                num_gold = len(gold_image_ids)
                filter_num_choices = num_gold + len(distractor_image_ids)
                perm = np.random.permutation(filter_num_choices)
                all_choices_image_ids = gold_image_ids + distractor_image_ids
                all_choices_cxt_list = gold_cxt_list + distractor_cxt_list
                assert len(all_choices_cxt_list) == filter_num_choices and len(all_choices_image_ids) == filter_num_choices
                all_choices_image_ids = [all_choices_image_ids[p] for p in perm]
                all_choices_cxt_list = [all_choices_cxt_list[p] for p in perm]
                label = torch.tensor([1. if p<num_gold else 0. for p in perm])
                label = torch.stack([label, 1-label], dim=0).transpose(1,0)
                input_ids_list = []
                segment_ids_list = []
                input_mask_list = []
                img_list = []
                vis_pe_list = []
                for i in range(filter_num_choices):
                    cxt = all_choices_cxt_list[i]
                    image_id = all_choices_image_ids[i]
                    tokens_a = ['[UNK]'] * self.max_len_img_cxt # 200
                    tokens_b = Q+A
                    max_len_cxt_meta = self.max_len_a - self.max_len_img_cxt # 200
                    truncate_tokens_pair(cxt, tokens_b, max_len=max_len_cxt_meta + self.max_len_b, max_len_a=max_len_cxt_meta, max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)
                    if self.use_img_meta: tokens_a += cxt
                    
                    # it seems that there is no need to pad cxt_meta to 200
                    #n_pad = self.max_len_a+1 - len(tokens_a) # +1 for the middle SEP
                    #tokens_a.extend(['[PAD]'] * n_pad)
                    tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

                    if self.new_segment_ids:
                        segment_ids = [4] * (len(tokens_a)+2) + [5] * (len(tokens_b)+1)
                    else:
                        segment_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b)+1)

                    

                    # self-attention mask
                    input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                    # everyone can attend to img, cxt_meta and Q. Nobody cares attention to A for filter task
                    img_end_pos = 1+self.len_vis_input
                    if self.use_img_content: input_mask[:, :img_end_pos].fill_(1)
                    st, end = 1 + self.max_len_img_cxt, len(tokens_a) + 2 + len(Q)
                    input_mask[:, st:end].fill_(1)

                    input_ids = self.indexer(tokens)
                    n_pad = self.max_len - len(input_ids)
                    input_ids.extend([0] * n_pad)
                    segment_ids.extend([0] * n_pad)

                    vis_pe, scores, img, cls_label = self.img_data_tsv[image_id//10000000][image_id % 10000000]
                    
                    #img = features['fc1_features'].detach().cpu().float()
                    #cls_label = features['cls_features'].detach().cpu().float()
                    #vis_pe = features['pred_boxes'].detach().cpu()

                    # Lazy normalization of the coordinates
                    w_est = torch.max(vis_pe[:, [0, 2]])*1.+1e-5
                    h_est = torch.max(vis_pe[:, [1, 3]])*1.+1e-5
                    vis_pe[:, [0, 2]] /= w_est
                    vis_pe[:, [1, 3]] /= h_est
                    assert h_est > 0, 'loader Processor: box h_est should greater than 0! {}'.format(h_est)
                    assert w_est > 0, 'loader Processor: box w_est should greater than 0! {}'.format(w_est)
                    rel_area = (vis_pe[:, 3]-vis_pe[:, 1])*(vis_pe[:, 2]-vis_pe[:, 0])
                    rel_area.clamp_(0)

                    vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), scores.view(-1, 1)), -1)
                    normalized_coord = F.normalize(vis_pe.data[:, :5] - 0.5, dim=-1)
                    vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), F.layer_norm(cls_label, [1595])), dim=-1)

                    assert img.size(0) == vis_pe.size(0), "img features and vis_pe should have the same token length!"
                    vis_pad = torch.zeros((self.max_len_img_cxt - img.size(0), img.size(-1)))
                    img = torch.cat((img, vis_pad), dim=0) 
                    pe_pad = torch.zeros((self.max_len_img_cxt - vis_pe.size(0), vis_pe.size(-1)))
                    vis_pe = torch.cat((vis_pe, pe_pad), dim=0)
                    assert vis_pe.size(0) == self.max_len_img_cxt
                    assert img.size(0) == self.max_len_img_cxt
                    input_ids_list.append(torch.tensor(input_ids))
                    segment_ids_list.append(torch.tensor(segment_ids))
                    input_mask_list.append(input_mask)
                    if not self.use_img_content: 
                        img = torch.zeros_like(img).float()
                        vis_pe = torch.zeros_like(vis_pe).float()
                        #print("zero placeholder for img content")
                    img_list.append(img)
                    vis_pe_list.append(vis_pe)
                
                logit_mask = [1.] * len(input_ids_list)
                if len(input_ids_list) < filter_max_choices:
                    num_placeholder = filter_max_choices - len(input_ids_list)
                    input_ids_list.extend([input_ids_list[-1]] * num_placeholder)
                    segment_ids_list.extend([segment_ids_list[-1]] * num_placeholder)
                    input_mask_list.extend([input_mask_list[-1]] * num_placeholder)

                    # TODO: 其实有了cxt_modality_label之后img_feats&vix_pe就不需要补placeholder了，后面把这里删了试试
                    #img_list.extend([img_list[-1]] * num_placeholder)
                    #vis_pe_list.extend([vis_pe_list[-1]] * num_placeholder)
                    logit_mask.extend([0.] * num_placeholder)
                    label = torch.cat([label, torch.tensor([[0., 0.]] * num_placeholder)], dim=0)
                input_ids = torch.stack(input_ids_list, dim=0)
                segment_ids = torch.stack(segment_ids_list, dim=0)
                input_mask = torch.stack(input_mask_list, dim=0)
                img = torch.stack(img_list, dim=0)
                vis_pe = torch.stack(vis_pe_list, dim=0)
                logit_mask = torch.tensor(logit_mask)
                ori_choices = [all_choices_image_ids]

                cxt_modality_label = range(filter_num_choices)
                # schema: (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next_label, do_filter_task, filter_label, logit_mask, ori_choices, self.task_idx, img, vis_pe, context, cxt_modality_label, example_id)
                return (input_ids, segment_ids, input_mask,       None,       None,       None,       -1,       do_filter_task,        label,       logit_mask, ori_choices, self.task_idx, img, vis_pe, context, cxt_modality_label, example_id)

            elif context == 'txt': # do_filter_task && context_is_text
                gold_facts, distractor_facts, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id = instance
                num_gold = len(gold_facts)
                filter_num_choices = num_gold + len(distractor_facts)
                perm = np.random.permutation(filter_num_choices)
                all_choices_facts = gold_facts + distractor_facts
                all_choices_facts = [all_choices_facts[p] for p in perm]
                #print(all_choices_facts)
                label = torch.tensor([1. if p<num_gold else 0. for p in perm])
                label = torch.stack([label, 1-label], dim=0).transpose(1,0)
                input_ids_list = []
                segment_ids_list = []
                input_mask_list = []
                for i in range(filter_num_choices):
                    tokens_a = []
                    if self.use_txt_fact: tokens_a = all_choices_facts[i].copy()
                    tokens_b = Q+A
                    truncate_tokens_pair(tokens_a, tokens_b, max_len=self.max_len_a+self.max_len_b, max_len_a=self.max_len_a, max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)
                    tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

                    if self.new_segment_ids:
                        segment_ids = [4] * (len(tokens_a)+2) + [5] * (len(tokens_b)+1)
                    else:
                        segment_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b)+1)

                    # self-attention mask
                    input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                    # everyone can attend to cxt and Q. Nobody cares attention to A for filter task
                    input_mask[:, :len(tokens_a)+2+len(Q)].fill_(1)

                    input_ids = self.indexer(tokens)
                    n_pad = self.max_len - len(input_ids)
                    input_ids.extend([0] * n_pad)
                    segment_ids.extend([0] * n_pad)

                    input_ids_list.append(torch.tensor(input_ids))
                    segment_ids_list.append(torch.tensor(segment_ids))
                    input_mask_list.append(input_mask)

                logit_mask = [1.] * len(input_ids_list)
                if len(input_ids_list) < filter_max_choices:
                    num_placeholder = filter_max_choices - len(input_ids_list)
                    input_ids_list.extend([input_ids_list[-1]] * num_placeholder)
                    segment_ids_list.extend([segment_ids_list[-1]] * num_placeholder)
                    input_mask_list.extend([input_mask_list[-1]] * num_placeholder)
                    logit_mask.extend([0.] * num_placeholder)
                    label = torch.cat([label, torch.tensor([[0., 0.]] * num_placeholder)], dim=0)
                input_ids = torch.stack(input_ids_list, dim=0) # 不确定，stack可能需要在collator里面操作
                segment_ids = torch.stack(segment_ids_list, dim=0)
                input_mask = torch.stack(input_mask_list, dim=0)
                logit_mask = torch.tensor(logit_mask)
                ori_choices = [' '.join(self.detokenize(c)) for c in all_choices_facts]

                cxt_modality_label = []
                # schema: (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next_label, do_filter_task, filter_label, logit_mask, ori_choices, self.task_idx, img, vis_pe, context, cxt_modality_label, example_id)
                return (input_ids, segment_ids, input_mask,       None,        None,        None,         -1,         do_filter_task,        label, logit_mask, ori_choices, self.task_idx, None, None, context, cxt_modality_label, example_id)
                raise NotImplementedError
        
        else: # qa task
            if context == 'img':
                gold_image_ids, distractor_image_ids, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id = instance
                tokens_a = ['[UNK]'] * self.max_len_img_cxt
                tokens_b = Q+A
                
                cxt = sum(gold_cxt_list, [])

                num_truncated_a, num_truncated_b = truncate_tokens_pair(cxt, tokens_b, max_len=self.max_len_a - self.max_len_img_cxt + self.max_len_b, max_len_a=self.max_len_a - self.max_len_img_cxt, max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)
                if self.use_img_meta: tokens_a += cxt
                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
                #print(tokens)
                #time.sleep(2)
                if self.new_segment_ids:
                    segment_ids = [4] * (len(tokens_a)+2) + [5] * (len(tokens_b)+1)
                else:
                    segment_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b)+1)

                effective_len_A = len(A) +1 - num_truncated_b[1]
                n_pred = min(self.max_pred, max(min(3, effective_len_A), int(round(effective_len_A * self.mask_prob)))) # predict everything if answer has less than 3 tokens
                cand_pos = []
                for i, tk in enumerate(tokens):
                    # only mask tk in A
                    if (i >= len(tokens_a)+2+len(Q)- num_truncated_b[0]):
                        cand_pos.append(i)
                
                shuffle(cand_pos)
                masked_pos = cand_pos[:n_pred]
                masked_tokens = [tokens[pos] for pos in masked_pos] # gth token in masked_pos
                for pos in masked_pos:
                    if rand() < 0.8:
                        #print("<0.8")
                        tokens[pos] = '[MASK]'
                    elif rand() < 0.5:
                        #print("<0.5")
                        random_word = get_random_word(self.vocab_words)
                        #print("\n-----------------RRRRRRRRRRRRRR----------------", random_word)
                        tokens[pos] = random_word


                masked_weights = [1] * len(masked_tokens)
                masked_ids = self.indexer(masked_tokens)
                #time.sleep(2)
                input_ids = self.indexer(tokens)
                n_pad = self.max_len - len(input_ids)
                input_ids.extend([0] * n_pad)
                segment_ids.extend([0] * n_pad)

                # self-attention mask
                num_img = len(gold_image_ids)
                input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)

                img_end_pos = 1 + self.len_vis_input*num_img
                if self.use_img_content: input_mask[:, :img_end_pos].fill_(1)
                st, end = 1 + self.max_len_img_cxt, 2 + len(tokens_a) + len(Q)
                input_mask[:, st:end].fill_(1)
                # Tokens in A can attend to previous tokens in A
                pred_st, pred_end = 2 + len(tokens_a) + len(Q), len(tokens)
                input_mask[pred_st:pred_end, pred_st:pred_end].copy_(self._tril_matrix[:pred_end-pred_st, :pred_end-pred_st])
                # Zero padding for masked target
                
                if self.max_pred > n_pred:
                    n_pad = self.max_pred - n_pred
                    masked_ids.extend([0] * n_pad)
                    masked_pos.extend([0] * n_pad)
                    masked_weights.extend([0] * n_pad)
                
                # Convert some inputs to tensors
                input_ids = torch.LongTensor(input_ids)
                segment_ids = torch.LongTensor(segment_ids)
                masked_ids = torch.LongTensor(masked_ids)
                masked_pos = torch.LongTensor(masked_pos)
                masked_weights = torch.LongTensor(masked_weights)

                img_list = []
                vis_pe_list = []
                for image_id in gold_image_ids:
                    vis_pe, scores, img, cls_label = self.img_data_tsv[image_id//10000000][image_id % 10000000]

                    #img = features['fc1_features'].detach().cpu().float()
                    #cls_label = features['cls_features'].detach().cpu().float()
                    #vis_pe = features['pred_boxes'].detach().cpu()

                    # Lazy normalization of the coordinates
                    w_est = torch.max(vis_pe[:, [0, 2]])*1.+1e-5
                    h_est = torch.max(vis_pe[:, [1, 3]])*1.+1e-5
                    vis_pe[:, [0, 2]] /= w_est
                    vis_pe[:, [1, 3]] /= h_est
                    assert h_est > 0, 'loader Processor: box h_est should greater than 0! {}'.format(h_est)
                    assert w_est > 0, 'loader Processor: box w_est should greater than 0! {}'.format(w_est)
                    rel_area = (vis_pe[:, 3]-vis_pe[:, 1])*(vis_pe[:, 2]-vis_pe[:, 0])
                    rel_area.clamp_(0)

                    vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), scores.view(-1, 1)), -1)
                    normalized_coord = F.normalize(vis_pe.data[:, :5] - 0.5, dim=-1)
                    vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), F.layer_norm(cls_label, [1595])), dim=-1)

                    img_list.append(img)
                    vis_pe_list.append(vis_pe)

                img = torch.cat(img_list, dim=0)
                vis_pe = torch.cat(vis_pe_list, dim=0)
                assert img.size(0) == vis_pe.size(0), "img features and vis_pe should have the same token length!"
                vis_pad = torch.zeros((self.max_len_img_cxt - img.size(0), img.size(-1)))#.to(device)
                img = torch.cat((img, vis_pad), dim=0)
                vis_pad = torch.zeros((self.max_len_img_cxt - vis_pe.size(0), vis_pe.size(-1)))#.to(device)
                vis_pe = torch.cat((vis_pe, vis_pad), dim=0)
                assert vis_pe.size(0) == self.max_len_img_cxt
                assert img.size(0) == self.max_len_img_cxt
                if len(masked_pos) < self.max_pred: 
                    print("num_truncated_b = ", num_truncated_b)
                    print(masked_pos)
                    print(n_pred)
                    print(self.max_pred)
                    print("effective_len_A = ", effective_len_A)
                    print("len(A) = ", len(A))
                    print("len(Q) = ", len(Q))
                    print("--------------")
                    print(tokens)
                cxt_modality_label = [1]
                # schema: (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next_label, do_filter_task, filter_label, logit_mask, ori_choices, self.task_idx, img, vis_pe, context, cxt_modality_label, example_id)
                return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights,      -1,      do_filter_task,        None,        None,        None,     self.task_idx, img, vis_pe, context, cxt_modality_label, example_id)
            
            
            else: # qa task, context is txt
                gold_facts, distractor_facts, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id = instance
                tokens_a = []
                if self.use_txt_fact: tokens_a = sum(gold_facts, [])
                tokens_b = Q+A
                num_truncated_a, num_truncated_b = truncate_tokens_pair(tokens_a, tokens_b, max_len=self.max_len_a+self.max_len_b, max_len_a=self.max_len_a, max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)
                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
                #print("\n", tokens)
                if self.new_segment_ids:
                    segment_ids = [4] * (len(tokens_a)+2) + [5] * (len(tokens_b)+1)
                else:
                    segment_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b)+1)

                effective_len_A = len(A)+1 - num_truncated_b[1]
                n_pred = min(self.max_pred, max(1, int(round(effective_len_A * self.mask_prob))))
                cand_pos = []
                for i, tk in enumerate(tokens):
                    # only mask tk in A
                    if (i >= len(tokens_a)+2+len(Q)- num_truncated_b[0]):
                        cand_pos.append(i)
                
                shuffle(cand_pos)
                masked_pos = cand_pos[:n_pred]
                masked_tokens = [tokens[pos] for pos in masked_pos] # gth token in masked_pos
                for pos in masked_pos:
                    if rand() < 0.8:
                        tokens[pos] = '[MASK]'
                    elif rand() < 0.5:
                        tokens[pos] = get_random_word(self.vocab_words)

                masked_weights = [1] * len(masked_tokens)
                masked_ids = self.indexer(masked_tokens)

                input_ids = self.indexer(tokens)
                n_pad = self.max_len - len(input_ids)
                input_ids.extend([0] * n_pad)
                segment_ids.extend([0] * n_pad)

                input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                input_mask[:, :len(tokens_a)+2+len(Q)].fill_(1)
                pred_st, pred_end = 2 + len(tokens_a) + len(Q), len(tokens)
                input_mask[pred_st:pred_end, pred_st:pred_end].copy_(self._tril_matrix[:pred_end-pred_st, :pred_end-pred_st])

                # Zero padding for masked target
                if self.max_pred > n_pred:
                    n_pad = self.max_pred - n_pred
                    masked_ids.extend([0] * n_pad)
                    masked_pos.extend([0] * n_pad)
                    masked_weights.extend([0] * n_pad)

                # Convert some inputs to tensors
                input_ids = torch.LongTensor(input_ids)
                segment_ids = torch.LongTensor(segment_ids)
                masked_ids = torch.LongTensor(masked_ids)
                masked_pos = torch.LongTensor(masked_pos)
                masked_weights = torch.LongTensor(masked_weights)
                
                # schema: (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next_label, do_filter_task, filter_label, logit_mask, ori_choices, self.task_idx, img, vis_pe, context, cxt_modality_label, example_id)
                return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights,       -1,      do_filter_task,      None,      None,       None,         self.task_idx, None, None,  context, None,               example_id)
                raise NotImplementedError

class Preprocess4webqaDecoder_VinVL(Pipeline):

    def __init__(self, vocab_words, indexer, seed, max_len, len_vis_input, max_len_a, max_len_Q, max_len_img_cxt=200, max_tgt_len=30, new_segment_ids=True, truncate_config={}, use_img_meta=True, use_img_content=True, use_txt_fact=True, ImgDataTsv_dict=None):
        super().__init__()
        self.task_idx = 3 # use task_idx for s2s in relaxed projection layer
        self.len_vis_input = len_vis_input
        self.vocab_words = vocab_words
        self.indexer = indexer
        self.max_len_img_cxt = max_len_img_cxt
        self._tril_matrix = torch.tril(torch.ones((max_len, max_len), dtype=torch.long))
        self.always_truncate_tail = truncate_config.get('always_truncate_tail', False)
        self.max_len_Q = max_len_Q
        self.max_len_a = max_len_a
        self.max_len = min(max_len, max_len_a + 2 + max_len_Q + max_tgt_len)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        assert max_len_a+max_len_Q <= max_len, "loader Processor: max_len_a + max_len_b > max_len"
        self.new_segment_ids = new_segment_ids
        self.use_img_meta = use_img_meta
        self.use_img_content = use_img_content
        self.use_txt_fact = use_txt_fact
        random.seed(seed)
        np.random.seed(seed)
        print("loader.use_img_meta = ", use_img_meta)
        print("loader.use_img_content = ", use_img_content)
        
        self.img_data_tsv = {}
        for k in ImgDataTsv_dict:
            self.img_data_tsv[k] = ImgDataTsv(ImgDataTsv_dict[k])

    def __call__(self, instance, filter_max_choices=None, device=None):
        _, __, ___, ____, _____, ______, do_filter_task, context, example_id = instance
        if do_filter_task:
            raise ValueError("Processor for decoder does not support filter task. \nFor filter task inference, please use run_webqa.py by setting args.do_train=False")
        else:
            if context in ['img', 'both']:
                gold_image_ids, distractor_image_ids, gold_cxt_list, distractor_cxt_list, Q, _, do_filter_task, context, example_id = instance # '_' as a placeholder for 'A'
                tokens_a = ['[UNK]'] * self.max_len_img_cxt
                cxt = sum(gold_cxt_list, [])

               
                tokens_b = Q.copy() # without copy Q will change as we modify tokens_b during padding!!!!!
                truncate_tokens_pair(cxt, tokens_b, max_len=self.max_len_a - self.max_len_img_cxt + self.max_len_Q, max_len_a=self.max_len_a - self.max_len_img_cxt, max_len_b=self.max_len_Q, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)
                if self.use_img_meta: tokens_a += cxt                
                
                n_pad = self.max_len_Q + self.max_len_a - len(tokens_a) - len(tokens_b)
                tokens_b += ['[PAD]'] * n_pad

                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b # + ['[SEP]'] # start generating right after Q
                #print(tokens_b)
                if self.new_segment_ids:
                    segment_ids = [4] * (len(tokens_a)+2) + [5] * len(tokens_b) + [5] * (self.max_len - len(tokens))
                else:
                    segment_ids = [0] * (len(tokens_a)+2) + [1] * len(tokens_b) + [5] * (self.max_len - len(tokens))

                
                # Q 和 A中间的position_id不连续真的会出问题。。。
                # position_ids
                ori_Q_len = min(len(Q), self.max_len_Q)
                position_ids = []
                for i in range(len(tokens_a) + 2 + ori_Q_len):
                    position_ids.append(i)
                for i in range(len(tokens_a) + 2 + ori_Q_len, len(tokens)):
                    position_ids.append(0)
                for i in range(len(tokens), self.max_len):
                    position_ids.append(i - len(tokens) + len(tokens_a) + 2 + ori_Q_len)
                #print(position_ids[202:302])
                
                

                # Token Indexing
                input_ids = self.indexer(tokens)

                # self-attention mask
                num_img = len(gold_image_ids)
                input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)

                img_end_pos = 1 + self.len_vis_input*num_img
                if self.use_img_content: input_mask[:, :img_end_pos].fill_(1)
                st, end = 1 + self.max_len_img_cxt, len(tokens_a) + 2 + ori_Q_len # paddings at the end of tokens_b don't need attention
                input_mask[:, st:end].fill_(1)
                # Tokens in A can attend to previous tokens in A
                pred_st, pred_end = len(tokens), self.max_len
                input_mask[pred_st:pred_end, pred_st:pred_end].copy_(self._tril_matrix[:pred_end-pred_st, :pred_end-pred_st])
                
                # Convert some inputs to tensors
                input_ids = torch.LongTensor(input_ids)
                segment_ids = torch.LongTensor(segment_ids)
                position_ids = torch.LongTensor(position_ids)

                img_list = []
                vis_pe_list = []
                for image_id in gold_image_ids:
                    vis_pe, scores, img, cls_label = self.img_data_tsv[image_id//10000000][image_id % 10000000]

                    #img = features['fc1_features'].detach().cpu().float()
                    #cls_label = features['cls_features'].detach().cpu().float()
                    #vis_pe = features['pred_boxes'].detach().cpu()

                    # Lazy normalization of the coordinates
                    w_est = torch.max(vis_pe[:, [0, 2]])*1.+1e-5
                    h_est = torch.max(vis_pe[:, [1, 3]])*1.+1e-5
                    vis_pe[:, [0, 2]] /= w_est
                    vis_pe[:, [1, 3]] /= h_est
                    assert h_est > 0, 'loader Processor: box h_est should greater than 0! {}'.format(h_est)
                    assert w_est > 0, 'loader Processor: box w_est should greater than 0! {}'.format(w_est)
                    rel_area = (vis_pe[:, 3]-vis_pe[:, 1])*(vis_pe[:, 2]-vis_pe[:, 0])
                    rel_area.clamp_(0)

                    vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), scores.view(-1, 1)), -1)
                    normalized_coord = F.normalize(vis_pe.data[:, :5] - 0.5, dim=-1)
                    vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), F.layer_norm(cls_label, [1595])), dim=-1)
                    
                    img_list.append(img)
                    vis_pe_list.append(vis_pe)
                    if len(img_list) >= 2: break # harded coded, doesn't allow more than 2 imgs

                if len(img_list) == 0:
                    assert len(vis_pe_list) == 0
                    img = torch.zeros((self.max_len_img_cxt, 2048)) # 2048 is hard-coded
                    vis_pe = torch.zeros((self.max_len_img_cxt, 1607)) # 1607 is hard-coded
                else:
                    img = torch.cat(img_list, dim=0)
                    vis_pe = torch.cat(vis_pe_list, dim=0)
                    assert img.size(0) == vis_pe.size(0), "img features and vis_pe should have the same token length!"
                    vis_pad = torch.zeros((self.max_len_img_cxt - img.size(0), img.size(-1)))#.to(device)
                    img = torch.cat((img, vis_pad), dim=0)
                    vis_pad = torch.zeros((self.max_len_img_cxt - vis_pe.size(0), vis_pe.size(-1)))#.to(device)
                    vis_pe = torch.cat((vis_pe, vis_pad), dim=0)
                assert vis_pe.size(0) == self.max_len_img_cxt
                assert img.size(0) == self.max_len_img_cxt

                cxt_modality_label = [1]
                # schema: (input_ids, segment_ids, position_ids, input_mask, self.task_idx, img, vis_pe, context, cxt_modality_label, example_id)
                return    (input_ids, segment_ids, position_ids, input_mask, self.task_idx, img, vis_pe, context, cxt_modality_label, example_id)
                
            
            else: # qa task, context is txt
                #raise NotImplementedError
                gold_facts, distractor_facts, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id = instance
                tokens_a = []
                if self.use_txt_fact: tokens_a = sum(gold_facts, [])
                tokens_b = Q.copy()
                truncate_tokens_pair(tokens_a, tokens_b, max_len=self.max_len_a+self.max_len_Q, max_len_a=self.max_len_a, max_len_b=self.max_len_Q, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)
                
                n_pad  = self.max_len_Q + self.max_len_a - len(tokens_a) - len(tokens_b)
                tokens_b += ['[PAD]'] * n_pad

                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b
                if self.new_segment_ids:
                    segment_ids = [4] * (len(tokens_a)+2) + [5] * len(tokens_b) + [5] * (self.max_len - len(tokens))
                else:
                    segment_ids = [0] * (len(tokens_a)+2) + [1] * len(tokens_b) + [5] * (self.max_len - len(tokens))

                ori_Q_len = min(len(Q), self.max_len_Q)
                position_ids = []
                for i in range(len(tokens_a) + 2 + ori_Q_len):
                    position_ids.append(i)
                for i in range(len(tokens_a) + 2 + ori_Q_len, len(tokens)):
                    position_ids.append(0)
                for i in range(len(tokens), self.max_len):
                    position_ids.append(i - len(tokens) + len(tokens_a) + 2 + ori_Q_len)
                #time.sleep(2)

                input_ids = self.indexer(tokens)

                input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                input_mask[:, :len(tokens_a)+2+ori_Q_len].fill_(1)
                pred_st, pred_end = len(tokens), self.max_len
                input_mask[pred_st:pred_end, pred_st:pred_end].copy_(self._tril_matrix[:pred_end-pred_st, :pred_end-pred_st])
                
                # Convert some inputs to tensors
                input_ids = torch.LongTensor(input_ids)
                segment_ids = torch.LongTensor(segment_ids)
                position_ids = torch.LongTensor(position_ids)
                
                # schema: (input_ids, segment_ids, position_ids, input_mask, self.task_idx, img, vis_pe, context, cxt_modality_label, example_id)
                return    (input_ids, segment_ids, position_ids, input_mask, self.task_idx, None, None,  context, None,               example_id)
                raise NotImplementedError

