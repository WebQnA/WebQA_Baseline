import json, pickle, os, time, copy, random, base64
from collections import Counter, defaultdict
import numpy as np
import torch
from pprint import pprint

fp = open("/data/yingshac/MMMHQA/detectron_output/distractors.tsv", "w")
fp_lineidx = open("/data/yingshac/MMMHQA/detectron_output/distractors.lineidx", "w")
try:
    for i in range(340194):
        if i%2000 == 1999: print(i)
        im = str(10000000+i)
        if not os.path.exists("/data/yingshac/MMMHQA/imgFeatures_upd/distractors/{}.pkl".format(im)): 
            print("NONE: ", im)
            fp_lineidx.write('{0}\n'.format(-1))
            #fp.write('{0}\n'.format('NONE'))
            continue
        features = pickle.load(open("/data/yingshac/MMMHQA/imgFeatures_upd/distractors/{}.pkl".format(im), "rb"))
        #print(features.keys())
        row = {}
        row['image_size'] = features['image_size']
        row['num_instances'] = features['num_instances']
        row['pred_classes'] = base64.b64encode(features['pred_classes'].detach().cpu().numpy()).decode("utf-8")
        row['scores'] = base64.b64encode(features['scores'].detach().cpu().numpy()).decode("utf-8")
        row['pred_boxes'] = base64.b64encode(features['pred_boxes'].detach().cpu().numpy()).decode("utf-8")
        row['fc1_features'] = base64.b64encode(features['fc1_features'].detach().cpu().numpy()).decode("utf-8")
        row['cls_features'] = base64.b64encode(features['cls_features'].detach().cpu().numpy()).decode("utf-8")
        x = json.dumps(row)
        line = [i, im, x]
        v = '{0}\n'.format('\t'.join(map(str, line)))
        fp_lineidx.write('{0}\n'.format(fp.tell()))
        fp.write(v)

    fp.close()
    fp_lineidx.close()

except KeyboardInterrupt:
    fp.close()
    fp_lineidx.close()