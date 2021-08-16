import os, json
import numpy as np
import pickle

l = os.listdir("/data/yingshac/MMMHQA/x_distractors")
max_idx = max([int(i.split('.')[0]) for i in l])
print("max_idx = ", max_idx)
x = np.full(max_idx+1, -1, dtype=np.longlong)
print(x.shape)
with open("/data/yingshac/maskrcnn-benchmark-1/datasets1/visualgenome/visualgenome/x_neg_hw.lineidx", "r") as fp:
    hw_lineidx = [int(i.strip()) for i in fp.readlines()]
with open("/data/yingshac/maskrcnn-benchmark-1/datasets1/visualgenome/visualgenome/x_neg_img.lineidx", "r") as fp:
    img_lineidx = [int(i.strip()) for i in fp.readlines()]
with open("/data/yingshac/MMMHQA/VinVL_output/x_neg_imgs_0_240661/predictions.lineidx", "r") as fp:
    pred_lineidx = [int(i.strip()) for i in fp.readlines()]
assert len(hw_lineidx) == len(img_lineidx) == len(pred_lineidx)

# Generate lineidx with the correct order
count = 0
with open("/data/yingshac/MMMHQA/VinVL_output/x_neg_imgs_0_240661/predictions.tsv", "r") as fp:
    for i in pred_lineidx:
        fp.seek(i)
        #print(count)
        #print(fp.readline().strip().split('\t')[0][:30])
        imgid = int(fp.readline().strip().split('\t')[0])
        x[imgid] = i
        count += 1
        if count % 5000 == 0: 
            print(count)
            pickle.dump(x, open("./Fix_VinVL_predictions_order_x_neg_imgs_tmp.pkl", "wb"))
pickle.dump(x, open("./Fix_VinVL_predictions_order_x_neg_imgs_tmp.pkl", "wb"))
# Write to lineidx file.
# write -1 if the corresponding id doesn't have an associated img
count = 0
with open('/data/yingshac/MMMHQA/VinVL_output/x_neg_imgs_0_240661/predictions_cor.lineidx', "w") as fp:
    for i in x:
        fp.write('{0}'.format(i) + '\n')
        if i == -1: count += 1
print("num of non-existing imgs = ", count)