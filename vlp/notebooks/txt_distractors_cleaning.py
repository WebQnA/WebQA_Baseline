### Two snippets with more than one sentences may have one common sentence. 
### Remove such duplication by keeping the longer one (so that avg_len can be pulled towards slightly longer)
### Also, remove excessively long snippets (i.e. >200 after tokenization)

import json, random, time
import numpy as np
from collections import Counter, defaultdict
import copy
import os
import re
from nltk.corpus import stopwords
import pickle
import cdifflib
np.set_printoptions(precision=4)

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


txt_dataset = json.load(open("/home/yingshac/CYS/WebQnA/WebQnA_data_new/txt_dataset_0809.json", "r"))


def get_overlap(s1, s2):
    s = cdifflib.CSequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return s1[pos_a:pos_a+size]


txt_dataset_remove_dup_tmp = {}
before_dedup = []
after_dedup = []
for k in txt_dataset:
    txt_dataset_remove_dup_tmp[k] = copy.deepcopy(txt_dataset[k])
    #print(k)
    sentences = [] # (idx, sentence)
    facts = txt_dataset[k]['new_negFacts'] + txt_dataset[k]['DistractorFacts']
    before_dedup.append(len(facts))
    for i in range(len(facts)):
        fact = facts[i]['fact']
        if len(tokenizer.tokenize(fact)) > 200: 
            #print("TooLong: ", fact)
            continue
        add = True
        for s in sentences:
            if len(get_overlap(s[1], fact)) / len(fact) > 0.25:
                if len(fact) > len(s[1]):
                    sentences.remove(s)
                else:
                    add = False
        if add: sentences.append((i, fact))
    txt_dataset_remove_dup_tmp[k]['new_negFacts'] = []
    after_dedup.append(len(sentences))
    for s in sentences:
        txt_dataset_remove_dup_tmp[k]['new_negFacts'].append(copy.deepcopy(facts[s[0]]))
    del txt_dataset_remove_dup_tmp[k]['DistractorFacts']
    txt_dataset_remove_dup_tmp[k]['split'] = txt_dataset_remove_dup_tmp[k].pop('Split')
    if len(after_dedup)%500 == 0:
        print(len(after_dedup))
        json.dump(txt_dataset_remove_dup_tmp, open("/home/yingshac/CYS/WebQnA/WebQnA_data_new/txt_dataset_remove_dup_tmp_0817.json", "w"), indent=4)

        
json.dump(txt_dataset_remove_dup_tmp, open("/home/yingshac/CYS/WebQnA/WebQnA_data_new/txt_dataset_remove_dup_tmp_0817.json", "w"), indent=4)
print(Counter(before_dedup))
print(Counter(after_dedup))


### Remove excessively long img distractors
txt_dataset_remove_dup_tmp = json.load(open("/home/yingshac/CYS/WebQnA/WebQnA_data_new/txt_dataset_remove_dup_tmp_0817.json", "r"))
txt_dataset_remove_clean_longimg_tmp = {}
before_cleaning = []
after_cleaning = []
for k in txt_dataset_remove_dup_tmp:
    txt_dataset_remove_clean_longimg_tmp[k] = copy.deepcopy(txt_dataset_remove_dup_tmp[k])
    #print(k)
    facts = txt_dataset_remove_dup_tmp[k]['img_negFacts']
    before_cleaning.append(len(facts))
    
    txt_dataset_remove_clean_longimg_tmp[k]['img_negFacts'] = []
    for i in range(len(facts)):
        fact = facts[i]['caption']
        if len(tokenizer.tokenize(fact)) > 200: 
            #print("TooLong: ", fact)
            continue
        txt_dataset_remove_clean_longimg_tmp[k]['img_negFacts'].append(copy.deepcopy(facts[i]))
    after_cleaning.append(len(txt_dataset_remove_clean_longimg_tmp[k]['img_negFacts']))
    
    if len(after_cleaning)%500 == 0:
        print(len(after_cleaning))
        json.dump(txt_dataset_remove_clean_longimg_tmp, open("/home/yingshac/CYS/WebQnA/WebQnA_data_new/txt_dataset_remove_clean_longimg_tmp_0817.json", "w"), indent=4)

json.dump(txt_dataset_remove_clean_longimg_tmp, open("/home/yingshac/CYS/WebQnA/WebQnA_data_new/txt_dataset_remove_clean_longimg_tmp_0817.json", "w"), indent=4)
print(Counter(before_cleaning))
print(Counter(after_cleaning))