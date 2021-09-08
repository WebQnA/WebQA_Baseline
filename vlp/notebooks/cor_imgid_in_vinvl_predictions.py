import json, pickle, os, time, copy, random, base64
from collections import Counter, defaultdict
import numpy as np
import torch
from pprint import pprint

with open("/data/yingshac/MMMHQA/VinVL_output/neg_imgs_0_338842/predictions.lineidx", "r") as fp:
    pred_lineidx = [int(i.strip()) for i in fp.readlines()]
    print(len(pred_lineidx))
old_fp = open("/data/yingshac/MMMHQA/VinVL_output/neg_imgs_0_338842/predictions.tsv", "r")
new_fp = open("/data/yingshac/MMMHQA/VinVL_output/neg_imgs_0_338842/predictions_0901.tsv", "w")
new_fp_lineidx = open("/data/yingshac/MMMHQA/VinVL_output/neg_imgs_0_338842/predictions_0901.lineidx", "w")
try:
    for i in range(len(pred_lineidx)):
        seek = pred_lineidx[i]
        if seek == -1: 
            new_fp_lineidx.write('{0}\n'.format(-1))
            continue
        old_fp.seek(seek)
        line = [i, 10000000+i] + old_fp.readline().strip().split('\t')
        v = '{0}\n'.format('\t'.join(map(str, line)))
        new_fp_lineidx.write('{0}\n'.format(new_fp.tell()))
        new_fp.write(v)
        if i%5000 == 4999: print(i)
except KeyboardInterrupt: 
    old_fp.close()
    new_fp.close()
    new_fp_lineidx.close()
print("Finish!!!")