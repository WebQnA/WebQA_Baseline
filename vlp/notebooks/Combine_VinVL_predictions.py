import os, json, time
import numpy as np
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument("--start", type=int)

# Concat two tsv files, adjust lineidx correspondingly
tsv_list = ["/data/yingshac/MMMHQA/VinVL_output/x_neg_imgs_0_108000/inference/vinvl_vg_x152c4/predictions.tsv",
            "/data/yingshac/MMMHQA/VinVL_output/x_neg_imgs_108000/inference/vinvl_vg_x152c4/predictions.tsv"]
lineidx_list = ["/data/yingshac/MMMHQA/VinVL_output/x_neg_imgs_0_108000/inference/vinvl_vg_x152c4/predictions.lineidx",
            "/data/yingshac/MMMHQA/VinVL_output/x_neg_imgs_108000/inference/vinvl_vg_x152c4/predictions.lineidx"]
output_path = "/data/yingshac/MMMHQA/VinVL_output/x_neg_imgs_0_240661"
if not os.path.isdir(output_path): os.mkdir(output_path)

print("\ntsv_list = ", tsv_list)
print("\nlineidx_list = ", lineidx_list)
print("\noutput_path = ", output_path)
time.sleep(5)
offset = 0
with open(os.path.join(output_path, "predictions.tsv"), "w") as output_tsv, open(os.path.join(output_path, "predictions.lineidx"), "w") as output_lineidx:
    for T, L in zip(tsv_list, lineidx_list):
        with open(L, "r") as fp:
            lineidx = [int(i.strip()) for i in fp.readlines()]
        print("len(lineidx) = ", len(lineidx))
        with open(T, "r") as fp:
            for i in range(len(lineidx)):
                fp.seek(lineidx[i])
                output_tsv.write(fp.readline())
                output_lineidx.write('{0}'.format(i+offset) + '\n')
                if i%5000 == 4999: print(i)
        offset += output_tsv.tell()
        print(offset, i)
        