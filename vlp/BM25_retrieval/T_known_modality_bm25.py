import numpy as np
import torch
import os, random, json, pickle, re
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from tqdm import tqdm

data_dir = "/home/yingshac/CYS/WebQnA/WebQnA_data_new/"


# Read fact2uniid
fact2uniid = pickle.load(open(os.path.join(data_dir, "CLIP_retrieval_experiments/fact2uniid.pkl"), "rb"))
uniid2fact = {i:fact for fact, i in fact2uniid.items()}
print(len(uniid2fact), uniid2fact[199999])

### load imgid2caption and image_id_map_0904
#imgid2caption = pickle.load(open(os.path.join(data_dir, "imgid2caption.pkl"), "rb"))
#image_id_map_0904 = pickle.load(open(os.path.join(data_dir, "image_id_map_0904.pkl"), "rb"))
#r_image_id_map_0904 = {newid:oldid for oldid, newid in image_id_map_0904.items()}
#print(len(imgid2caption), len(image_id_map_0904))

# Read test_imgguid2qid, test_txtguid2qid
#test_imgguid2qid = pickle.load(open(os.path.join(data_dir, "CLIP_retrieval_experiments/test_imgguid2qid.pkl"), "rb"))
test_txtguid2qid = pickle.load(open(os.path.join(data_dir, "CLIP_retrieval_experiments/test_txtguid2qid.pkl"), "rb"))

test_txtqid2guid = {i:guid for guid, i in test_txtguid2qid.items()}
#test_imgqid2guid = {i:guid for guid, i in test_imgguid2qid.items()}

#print(len(test_imgqid2guid), len(test_txtqid2guid), test_imgqid2guid[999], test_txtqid2guid[888])

T_corpus = [uniid2fact[i].split() for i in range(len(uniid2fact))]
#I_corpus = [imgid2caption[i].split() for i in range(30000000, 30000000+len(imgid2caption))]
print(len(T_corpus))
#print(len(I_corpus))


dataset = json.load(open("/home/yingshac/CYS/WebQnA/WebQnA_data_new/WebQA_0904_concat_newimgid_newguid.json", "r"))
print(Counter([dataset[k]['split'] for k in dataset]))
print(len(set([dataset[k]['Guid'] for k in dataset])))
print(Counter([dataset[k]['Qcate'] for k in dataset]))

T_queries = [dataset[test_txtqid2guid[i]]['Q'].replace('"', '').split() for i in range(len(test_txtqid2guid))]
#I_queries = [dataset[test_imgqid2guid[i]]['Q'].replace('"', '').split() for i in range(len(test_imgqid2guid))]


from gensim import corpora
from gensim.summarization import bm25

dictionary = corpora.Dictionary(T_corpus)
corpus = [dictionary.doc2bow(text) for text in T_corpus]
bm25_obj = bm25.BM25(corpus)

result_matrix = torch.full((len(T_queries), 2), -1)
for i, q in tqdm(enumerate(T_queries)):
    query_doc = dictionary.doc2bow(q)
    scores = bm25_obj.get_scores(query_doc)
    best_docs = sorted(range(len(scores)), key=lambda i: scores[i])[-2:]
    result_matrix[i] = torch.tensor(best_docs)
    if i%200 == 199:
        torch.save(result_matrix, "result_matrix_TT.pt")
torch.save(result_matrix, "result_matrix_TT.pt")