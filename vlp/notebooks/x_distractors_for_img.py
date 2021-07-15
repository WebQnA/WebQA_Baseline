import json, random
import numpy as np
from pprint import pprint
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
import requests
import urllib.request
import re
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import pickle
import copy
import spacy
from spacy import displacy
from itertools import tee
import wikipedia
import pylcs
import string

PUNCTUATIONS = set(string.punctuation)
img_meta = json.load(open("/home/yingshac/CYS/WebQnA/WebQnA_data/img_metadata-Copy1.json", "r"))
img_dataset = json.load(open("/home/yingshac/CYS/WebQnA/WebQnA_data_new/img_dataset_J_0623-Copy1.json", "r"))
pos_list = ['NUM', 'NOUN', 'ADJ', 'PROPN']
nlp = spacy.load('en_core_web_sm')
def IoU(A, B):
    intersection = len(A.intersection(B))
    union = len(A.union(B))
    return round(intersection / (union+1e-7), 2)
def find_sentences_from_page_for_img_data(title, page, keywords, answerwords):
    try: 
        content = wikipedia.page(title, auto_suggest=False, redirect=True).content
        paragraphs = content[:content.find('== References ==')].split('\n')
        
    except: return {}
    #records = []
    sen2score = {}
    for p in paragraphs:
        if len(p.split()) >= 10:
            #records.append(-999)
            doc = nlp(p)
            for s in doc.sents:
                if len(s) < 10: 
                    continue
                nouns_in_s = [t.text for t in s if (t.pos_ in pos_list or ((not t.is_sent_start) and t.text[0].isupper()))]

                IoU_Q = IoU(set(nouns_in_s), keywords)
                IoU_A = IoU(set(nouns_in_s), answerwords)
                if IoU_Q -  IoU_A > 0.06:
                    sen2score[s.text] = {'scores': (IoU_Q, IoU_A, IoU_Q - IoU_A), 'link': page, 'title': title}
                #records.append(round(IoU_Q, 2))
    #print(records)

    #records = []
    for p in paragraphs:
        if len(p.split()) >= 10:
            #records.append(-999)
            doc = nlp(p)
            it1, it2 = tee(doc.sents)
            next(it2, None)
            for s1, s2 in zip(it1, it2):
                if len(s1) < 5 or len(s2) < 5 or len(s1)+len(s2) > 70 or len(s1)+len(s2) < 10: 
                    continue 
                nouns_in_s = [t.text for s in [s1, s2] for t in s if (t.pos_ in pos_list or ((not t.is_sent_start) and t.text[0].isupper()))]

                IoU_Q = IoU(set(nouns_in_s), keywords)
                IoU_A = IoU(set(nouns_in_s), answerwords)
                if IoU_Q -  IoU_A >= 0.06:
                    sen2score[" ".join([s1.text, s2.text])] = {'scores': (IoU_Q, IoU_A, IoU_Q - IoU_A), 'link': page, 'title': title}
                    #print(s)
                #records.append(round(IoU_Q, 2))
    #print(records)
    #print(len(sen2score))
    return sen2score

def get_keywords_from_img_sample(k):
    Q = img_dataset[str(k)]['Q'].replace('"', '')
    doc = nlp(Q)
    keywords = set([t.text for s in doc.sents for t in s if t.pos_ in ['NUM', 'PROPN', 'ADJ', 'NOUN'] or ((not t.is_sent_start) and t.text[0].isupper())])
    keywords = keywords - PUNCTUATIONS
    
    ### Extract noun chunks
    keywords = [t.text for s in doc.sents for t in s if t.pos_ in ['NUM', 'PROPN', 'ADJ'] or ((not t.is_sent_start) and t.text[0].isupper())]
    chunks = []
    for chunk in doc.noun_chunks:
        if any([n in keywords for n in chunk.text.split()]):
            chunks.append(chunk.text)
    if not chunks: chunks.extend([t.text for s in doc.sents for t in s if t.pos_ == 'PROPN' or ((not t.is_sent_start) and t.text[0].isupper())])
    
    A = img_dataset[str(k)]['A'].replace('"', '')
    doc = nlp(A)
    answerwords = set([t.text for t in doc if t.pos_ in pos_list or ((not t.is_sent_start) and t.text[0].isupper())])
    answerwords = answerwords - PUNCTUATIONS
    
    
    return keywords, answerwords, Q, A, chunks

# Given img_dataset indx & title, find sentences with word overlap with the question
def find_sentences_from_indx_for_img(k, keywords, answerwords, chunks):
    sen2score = {}
    candidate_pages = noun_chunk2candidate_page(chunks)
    print("num of candidate pages = {}\n".format(len(candidate_pages)))
    for title in candidate_pages:
        page = "https://en.wikipedia.org/wiki/" + "_".join(title.split())
        sen2score.update(find_sentences_from_page_for_img_data(title, page, keywords, answerwords))
    sen2score = dict(sorted(sen2score.items(), key=lambda x: x[1]['scores'][0], reverse=True))
    return sen2score

def add_html_row_x_distractor_for_img(k, sen2score, word_lists, chunks, colors=["(205, 245, 252)", "(255, 214, 222)"]):
    html = '<tr><td>{}.</td>'.format(k)
    Q = img_dataset[str(k)]['Q'].replace('"', '')
    html += '<td>Q: {}<br><br>'.format(highlight_words(word_lists, chunks, colors, Q))
    A = img_dataset[str(k)]['A'].replace('"', '')
    for gid in img_dataset[str(k)]['GoldIds']:
        img = img_meta[str(int(gid))]
        html += '<a href="{}" target="_blank"><img style="display:block; max-height:300px; max-width:100%;" src = "{}"></a>'.format(img['page'], img['src'])
        html += '<br>Title = {}<br>Description = {}<br><br>'.format(highlight_words(word_lists, [], colors, img['name'].replace("_", " ")), highlight_words(word_lists, [], colors, img['description'].replace("_", " ")))
    html += 'A: {}<br>'.format(highlight_words(word_lists, [], colors, A))
    html += '</td><td>'
    
    for s in list(sen2score.keys())[:10]:
        html += '{} --- {} '.format(highlight_words(word_lists, [], colors, s), str(sen2score[s]['scores']))
        html += '<a href="{}"  target="_blank"> {}</a><br><br>'.format(sen2score[s]['link'], sen2score[s]['title'])
    
    html += '</td></tr>'
    html += '<tr><td colspan=3><hr></td></tr>'
    return html.encode('ascii', 'xmlcharrefreplace').decode("utf-8") 

def highlight_words(word_lists, chunks, colors, sentence):
    s = copy.deepcopy(sentence)
    if "".join(chunks):
        s = re.sub(r'\s*(' + r'|'.join([re.escape(c) for c in chunks]) + r')\s*', lambda m: '<span class="chunk">{}</span>'.format(m.group()), s)
    for word_list, color in zip(word_lists, colors):
        if "".join(word_list): s = re.sub(r'\b(' + r'|'.join(word_list) + r')\b', lambda m: '<span style="background-color:rgb{}">{}</span>'.format(color, m.group()), s)
    return s

def noun_chunk2candidate_page(chunks):
    pages = []
    for chunk in chunks:
        pages.extend(wikipedia.search(chunk))
    return pages



html = '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">'
html += '<script src="https://code.jquery.com/jquery-3.2.1.min.js" integrity="sha256-hwg4gsxgFZhOsEEamdOYGBf13FyQuiTwlAQgxVSNgt4=" crossorigin="anonymous"></script>'
html += '<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>'
html += '<!DOCTYPE html><html><head><meta http-equiv="content-type" content="text/html; chatset="UTF-8"><body>'
html += '<script>$("img").on("error", function(){console.log($(this).attr("src"));});</script>'
html += '<style>table {border-collapse: separate;border-spacing: 10px;}\n'
html += '.chunk {text-decoration: underline solid rgb(227, 123, 253) 3px;}</style>'
html += '<table border="0" style="table-layout: fixed; width: 100%; word-break:break-word">'
html += '<tr bgcolor=lightblue style="text-align: center;"><td width=5%>Index</td><td width=35%>Q & Pos Facts</td><td width=60%>Neg Facts</td></tr>'
x = []
for k in random.sample(range(25467), 1):
    print(k)
    keywords, answerwords, Q, A, chunks = get_keywords_from_img_sample(k)
    print("Q = ", Q)
    print("Keywords = {}".format(keywords))
    print("A = ", A)
    print("answerwords = {}".format(answerwords))
    print("\nNoun chunks: ", chunks)
    print(' ')
    d = find_sentences_from_indx_for_img(k, keywords, answerwords, chunks)
    x.append(len(d))
    
    word_lists = [keywords, answerwords]
    html += add_html_row_x_distractor_for_img(k, d, word_lists, chunks, colors=["(193, 239, 253)", "(255, 214, 222)"])
    o = open('x_distractor_for_img_demo.html', 'wt')
    o.write(html)
    o.close()
html += '</table></body></html>'
o = open('x_distractor_for_img_demo.html', 'wt')
o.write(html)
o.close()

