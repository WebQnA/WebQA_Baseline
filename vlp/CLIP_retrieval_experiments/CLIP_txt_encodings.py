import torch
import clip, pickle
from PIL import Image
import random, os, base64, json, copy
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from collections import Counter, defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

data_dir = "/home/yingshac/CYS/WebQnA/WebQnA_data_new/"

# Read fact2uniid
fact2uniid = pickle.load(open(os.path.join(data_dir, "CLIP_retrieval_experiments/fact2uniid.pkl"), "rb"))
uniid2fact = {i:fact for fact, i in fact2uniid.items()}
print(len(uniid2fact), uniid2fact[299999])

### Generate 540k x 512 matrix
rows = []
bs = 512
batched_ids = []
num_bs = len(uniid2fact)//bs
for j in tqdm(range(num_bs)):
    batched_ids = list(range(j*bs, j*bs+bs))
    text_input = clip.tokenize([uniid2fact[i] for i in batched_ids], truncate=True).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    rows.append(text_features)
if not len(uniid2fact) % bs == 0:
    batched_ids = list(range(num_bs*bs, len(uniid2fact)))
    text_input = clip.tokenize([uniid2fact[i] for i in batched_ids], truncate=True).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    rows.append(text_features)

T540Kx512 = torch.cat(rows)
print(T540Kx512.size())
torch.save(T540Kx512, os.path.join(data_dir, "CLIP_retrieval_experiments/T540Kx512.pt"))