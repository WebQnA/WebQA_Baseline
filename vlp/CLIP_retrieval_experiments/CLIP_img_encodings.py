from tqdm import tqdm
import json, random, time, os, base64
import clip, pickle
import numpy as np
from pprint import pprint
from io import BytesIO
from collections import Counter, defaultdict
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

data_dir = "/home/yingshac/CYS/WebQnA/WebQnA_data_new/"

with open("/data/yingshac/WebQA/base64_0904/imgs.lineidx", "r") as fp_img:
    img_lineidx = [int(i.strip()) for i in fp_img.readlines()]
    print(len(img_lineidx))

def get_image_input(i):
    with open("/data/yingshac/WebQA/base64_0904/imgs.tsv", "r") as fp_img:
        fp_img.seek(img_lineidx[i])
        imgid, img_base64 = fp_img.readline().strip().split('\t')
    image = Image.open(BytesIO(base64.b64decode(img_base64)))
    image_input = preprocess(image)
    return image_input

### Generate 390k x 512 matrix
rows = []
bs = 512
num_bs = 389750//bs
for j in tqdm(range(num_bs)):
    batched_ids = list(range(j*bs, j*bs+bs))
    image_input = torch.tensor(np.stack([get_image_input(i) for i in batched_ids])).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    rows.append(image_features)
if not 389750 % bs == 0:
    batched_ids = list(range(num_bs*bs, 389750))
    image_input = torch.tensor(np.stack([get_image_input(i) for i in batched_ids])).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    rows.append(image_features)

I390Kx512 = torch.cat(rows)
print(I390Kx512.size())
torch.save(I390Kx512, os.path.join(data_dir, "CLIP_retrieval_experiments/I390Kx512.pt"))