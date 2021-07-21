import json, random, time, sys, os
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
import string
import wikipedia
import spacy
from itertools import tee
import pylcs
import argparse
np.set_printoptions(precision=4)

parser = argparse.ArgumentParser()
#parser.add_argument("--samples", type=str)
#parser.add_argument("--output_filename", type=str)
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)
parser.add_argument('--disable_print', action='store_true')
args = parser.parse_args()
assert args.end > args.start
args.boundary = ((args.end-1) // 1000 + 1) * 1000
if args.disable_print:
    sys.stdout = open(os.devnull, 'w')

API_URL = 'http://en.wikipedia.org/w/api.php'
nlp = spacy.load('en_core_web_sm')
USER_AGENT_LIST = [
            "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
            "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
            "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
            "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
            "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
            "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
            "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
            "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
            "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
            "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 TaoBrowser/2.0 Safari/536.11",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER",
            "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; LBBROWSER)",
            "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E; LBBROWSER)",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 LBBROWSER",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)",
            "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; QQBrowser/7.0.3698.400)",
            "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; 360SE)",
            "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)",
            "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.89 Safari/537.1",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.89 Safari/537.1",
            "Mozilla/5.0 (iPad; U; CPU OS 4_2_1 like Mac OS X; zh-cn) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8C148 Safari/6533.18.5",
            "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:2.0b13pre) Gecko/20110307 Firefox/4.0b13pre",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:16.0) Gecko/20100101 Firefox/16.0",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
            "Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        ]

url_blocklist = ['seal', 'sign ', 'pdf', 'gif', 'icon', 'notice', 'cartoon', 'publish', 'menu', 'logo', 'svg', 'webm', 'page', \
                     'ogg', 'flickr', 'poster', 'ogv', 'banner', 'tif', 'montage', 'centralautologin', 'footer']

pos_list = ['NUM', 'NOUN', 'ADJ', 'PROPN']
pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*|\(|\)|-')
PUNCTUATIONS = set(string.punctuation)
new_txt_data = json.load(open("/home/yingshac/CYS/WebQnA/WebQnA_data_new/output_mine_all_schema.json", "r"))

def IoU(A, B):
    intersection = len(A.intersection(B))
    union = len(A.union(B))
    return round(intersection / (union+1e-7), 2)

def _wiki_request(params):
  
    global USER_AGENT_LIST

    params['format'] = 'json'
    if not 'action' in params:
        params['action'] = 'query'

    headers = {
        'User-Agent': random.choice(USER_AGENT_LIST)#'wikipedia (https://github.com/goldsmith/Wikipedia/)'
    }

    r = requests.get(API_URL, params=params, headers=headers)
    return r.json()

def content(title):

    query_params = {
        'prop': 'extracts|revisions',
        'explaintext': '',
        'rvprop': 'ids',
        'titles': title
    }
    request = _wiki_request(query_params)
    result = request['query']['pages']
    content = result[list(result.keys())[0]]['extract']
    return content

def get_html(url):
    req = urllib.request.Request(url, headers = {'User-Agent': random.choice(USER_AGENT_LIST)})
    with urllib.request.urlopen(req) as f:
        html = f.read().decode('utf-8')
    end_indx = html.find('<h2><span class="mw-headline" id="References">References</span>')
    html = html[:end_indx]
    return html

### Scrap imgs and their captions. 
def get_imgs_and_captions(html):
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all('img')
    imgUrls = []
    captions = []
    for l in links:
        imgUrl = l.get('src')
        try: 
            width = int(l['width'])
            height = int(l['height'])
        except:
            continue
        if width<100 or height<100:
            continue
        if any(b in imgUrl.lower() for b in url_blocklist): 
            continue
        imgUrls.append(imgUrl)

        # Special case for thumb images, which are located inside a table
        thumbinner_div = l.find_parent("div", class_='thumbinner')
        if thumbinner_div: 
            captions.append(thumbinner_div.text)
            continue
        
        segments = []
        prev_th = l.find_previous('th')
        if prev_th: segments.append(prev_th.get_text(strip=True))
            
        tr_parent = l.find_parent('tr')
        if tr_parent: segments.append(tr_parent.get_text(strip=True))

        captions.append(". ".join(segments))
    return imgUrls, captions

def find_pages_by_hyperlink(keywords, url):
    if 'en.wikipedia.org' not in url: return {}
    anchor2page = {}
    req = urllib.request.Request(url, headers = {'User-Agent': random.choice(USER_AGENT_LIST)})
    with urllib.request.urlopen(req) as f:
        html = f.read().decode('utf-8')
    end_indx = html.find('<h2><span class="mw-headline" id="References">References</span>')
    html = html[:end_indx]
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all('a', attrs={'href': re.compile("^/wiki/(?!.*(:|\(identifier\))).*")})
    for link in links:
        title = link.get('title')
        text = link.text
        if title is None or not text: continue
        #print(link)
        if any(b in keywords for b in title.split()):
            pagelink = 'https://en.wikipedia.org' + link.get('href')
            if pagelink.find("#") > -1:
                pagelink = pagelink[:pagelink.find("#")]
            anchor2page[__load(title)] = pagelink
        if len(text) == 0: print(link)
        elif pylcs.lcs(title.lower(), text.lower())/len(text) < 0.85:
            if any(b in keywords for b in text.split()):
                pagelink = 'https://en.wikipedia.org' + link.get('href')
                if pagelink.find("#") > -1:
                    pagelink = pagelink[:pagelink.find("#")]
                anchor2page[__load(title)] = pagelink
    if '' in anchor2page: del anchor2page['']
    return anchor2page

def get_keywords_and_relevant_pages_from_txt_sample(k):
    Q = new_txt_data[str(k)]['Q']
    doc = nlp(Q)
    keywords = set([t.text for s in doc.sents for t in s if t.pos_ in pos_list or ((not t.is_sent_start) and t.text[0].isupper())])
    titlewords = set()
    anchor2page = {}
    for f in new_txt_data[str(k)]['SupportingFacts']:
        fact_title_raw = ' '.join(urllib.parse.unquote(f['url']).split('/')[-1].split('_'))
        fact_title = pattern.sub('', fact_title_raw)
        titlewords = titlewords.union(fact_title.split())
        for title in wikipedia.search(fact_title_raw):
            anchor2page[__load(title)] = "https://en.wikipedia.org/wiki/" + urllib.parse.quote("_".join(title.split()))
    keywords = keywords - PUNCTUATIONS
    keywords = set(sum([[w.capitalize(), w.lower()] for w in keywords], []))
    titlewords = titlewords - PUNCTUATIONS
    
    #print("#pages before extension by hyperlink: ", len(anchor2page))
    for f in new_txt_data[str(k)]['SupportingFacts']:
        d = find_pages_by_hyperlink(keywords.union(titlewords), f['url'])
        anchor2page.update(d)
    if '' in anchor2page: del anchor2page['']
    #print("#pages after extension by hyperlink: ", len(anchor2page))
    
    A = new_txt_data[str(k)]['A']
    doc = nlp(A)
    answerwords = set([t.text for t in doc if t.pos_ in pos_list or ((not t.is_sent_start) and t.text[0].isupper())]) - keywords
    answerwords = answerwords - PUNCTUATIONS
    answerwords = set(sum([[w.capitalize(), w.lower()] for w in answerwords], []))
    goldfactwords = set()
    Q_A_words = keywords.union(answerwords)
    for f in new_txt_data[str(k)]['SupportingFacts']:
        doc = nlp(f['fact'])
        goldfactwords = goldfactwords.union(set([t.text for t in doc if t.pos_ in pos_list or ((not t.is_sent_start) and t.text[0].isupper())]) - Q_A_words)
    goldfactwords = goldfactwords - PUNCTUATIONS
    goldfactwords = set(sum([[w.capitalize(), w.lower()] for w in goldfactwords], []))
    
    for a in list(anchor2page.keys()):
        if is_disambiguation_page(anchor2page[a]):
            print(a, " is an disambiguation page")
            for t in recover_disambiguation_page(a):
                anchor2page[__load(t)] = "https://en.wikipedia.org/wiki/" + urllib.parse.quote("_".join(t.split()))
            del anchor2page[a]
    if '' in anchor2page: del anchor2page['']
    return titlewords, keywords, goldfactwords, answerwords, Q, A, anchor2page

def get_page_categories(title):
    url = 'https://en.wikipedia.org/w/api.php?format=xml&action=query&prop=categories&titles='+urllib.parse.quote(title)
    req = urllib.request.Request(url, headers = {'User-Agent': random.choice(USER_AGENT_LIST)})
    with urllib.request.urlopen(req) as f:
        xml = f.read().decode('utf-8')
    soup= BeautifulSoup(xml,"lxml-xml")
    tags = soup.find('categories')
    if tags is None: return []
    categories = [c.get('title').replace("Category:", "") for c in tags]
    return categories

def is_disambiguation_page(url):
    title = url.split('/')[-1]
    return 'disambiguation' in " ".join(get_page_categories(title))

def recover_disambiguation_page(title):
    try: 
        wikipedia.page(title)
        return []
    except Exception as e:
        err = str(e)
        return err[err.find("may refer to:")+13:].strip().split('\n')
    
def __load(title):
    '''
    Load basic information from Wikipedia.
    Confirm that page exists and is not a disambiguation/redirect.
    Does not need to be called manually, should be called automatically during __init__.
    '''
    query_params = {
        'prop': 'info|pageprops',
        'inprop': 'url',
        'ppprop': 'disambiguation',
        'redirects': '',
        'titles': title
    }

    request = _wiki_request(query_params)

    query = request['query']
    pageid = list(query['pages'].keys())[0]
    page = query['pages'][pageid]

    # missing is present if the page is missing
    if 'missing' in page:
        print("Page is missing: ", title)
        return ""

    # same thing for redirect, except it shows up in query instead of page for whatever silly reason
    elif 'redirects' in query:
        redirects = query['redirects'][0]
        return redirects['to']

    return title

def find_sentences_from_page(title, page, keywords, answerwords, goldfactwords):
    try: 
        cont = content(title)
        paragraphs = cont[:cont.find('== References ==')].split('\n')
        
    except: 
        print("Exception from find_sentences_from_page, title = ", title)
        return {}
    sen2score = {}
    for p in paragraphs:
        if len(p.split()) >= 10:
            doc = nlp(p)
            for s in doc.sents:
                if len(s) < 10: 
                    continue
                nouns_in_s = [t.text for t in s if (t.pos_ in pos_list or ((not t.is_sent_start) and t.text[0].isupper()))]

                IoU_Q = IoU(set(nouns_in_s), keywords)
                IoU_A = IoU(set(nouns_in_s), answerwords)
                if IoU_Q -  IoU_A > 0.06:
                    #print(IoU_Q, IoU_A, s.text)
                    IoU_G = IoU(set(nouns_in_s), goldfactwords)
                    sen2score[s.text] = {'scores': (IoU_Q, IoU_A, IoU_G, IoU_Q - IoU_A, IoU_Q - IoU_A - IoU_G), 'link': page, 'title': title}

    for p in paragraphs:
        if len(p.split()) >= 10:
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
                    #print(IoU_Q, IoU_A, " ".join([s1.text, s2.text]))
                    IoU_G = IoU(set(nouns_in_s), goldfactwords)
                    sen2score[" ".join([s1.text, s2.text])] = {'scores': (IoU_Q, IoU_A, IoU_G, IoU_Q - IoU_A, IoU_Q - IoU_A - IoU_G), 'link': page, 'title': title}
                    #print(s)
    return sen2score

def find_imgs_from_page(title, page, keywords, answerwords, goldfactwords):
    try: 
        html = get_html(page)
        imgs, caps = get_imgs_and_captions(html)
        
    except: 
        print("Exception from find_imgs_from_page, page = ", page)
        return {}
    
    cap2score = {}
    for im, cap in zip(imgs, caps):
        doc = nlp(cap)
        nouns_in_s = [t.text for t in doc if (t.pos_ in pos_list or ((not t.is_sent_start) and t.text[0].isupper()))]

        IoU_Q = IoU(set(nouns_in_s), keywords)
        IoU_A = IoU(set(nouns_in_s), answerwords)
        if IoU_Q -  IoU_A > 0.06:
            IoU_G = IoU(set(nouns_in_s), goldfactwords)
            cap2score[doc.text] = {'scores': (IoU_Q, IoU_A, IoU_G, IoU_Q - IoU_A, IoU_Q - IoU_A - IoU_G), 'img':im, 'link': page, 'title': title}    
    return cap2score

def get_sen2score_from_indx(k):
    print('k = ', k)
    titlewords, keywords, goldfactwords, answerwords, Q, A, anchor2page = get_keywords_and_relevant_pages_from_txt_sample(k)
    print("Q = ", Q)
    print("A = ", A)
    print("keywords = ", keywords)
    print("titlewords = ", titlewords)
    print("answerwords = ", answerwords)
    print("goldfactwords = ", goldfactwords)
    
    sen2score = {}
    cap2score = {}
    for title in anchor2page:
        sen2score.update(find_sentences_from_page(title, anchor2page[title], keywords, answerwords, goldfactwords))
        cap2score.update(find_imgs_from_page(title, anchor2page[title], keywords, answerwords, goldfactwords))
    sen2score = dict(sorted(sen2score.items(), key=lambda x: x[1]['scores'][-1], reverse=True))
    cap2score = dict(sorted(cap2score.items(), key=lambda x: x[1]['scores'][-1], reverse=True))
    print("total num of sentences found = ", len(sen2score))
    print("total num of imgs found = ", len(cap2score))
    
    word_lists = (titlewords, keywords, goldfactwords, answerwords)
    return sen2score, cap2score, word_lists

def highlight_words(word_lists, colors, sentence):
    s = copy.deepcopy(sentence)
    for word_list, color in zip(word_lists, colors):
        if "".join(word_list): s = re.sub(r'\b(' + r'|'.join([re.escape(c) for c in word_list]) + r')\b\s*', lambda m: '<span style="background-color:rgb{}">{}</span>'.format(color, m.group()), s)
    return s

def add_html_row(k, sen2score, cap2score, word_lists, colors = ["(223, 255, 238)", "(193, 239, 253)", "(253, 252, 152)", "(255, 214, 222)"]):
    html = ""
    html += '<tr><td>{}.</td><td>Q: {}<br>'.format(k, highlight_words(word_lists, colors, new_txt_data[str(k)]['Q']))
    for f in new_txt_data[str(k)]['SupportingFacts']:
        html += '<br><br>&nbsp;&nbsp;{}'.format(highlight_words(word_lists, colors, f['fact']))
        html += '<a href="{}"> link</a>'.format(f['url'])
    html += '<br><br>A: {}<br>'.format(highlight_words(word_lists, colors, new_txt_data[str(k)]['A']))
    html += '</td><td>'
    s_buckets = defaultdict(lambda: [])
    for s in sen2score:
        if sen2score[s]['scores'][2] == 0.0 and sen2score[s]['scores'][1] == 0.0 and len(s.split()) in range(22, 60):
            s_buckets['good'].append(s)
        elif sen2score[s]['scores'][1] > 0.0 or sen2score[s]['scores'][2] > 0.0:
            s_buckets['(maybe)falseneg'].append(s)
        
    for s in s_buckets['good'][:10]:
        html += '{} --- {} '.format(highlight_words(word_lists, colors, s), str(sen2score[s]['scores']))
        html += '<a href="{}"> {}</a><br><br>'.format(sen2score[s]['link'], sen2score[s]['title'])
    html += '<strong> ----------------- (maybe)falseneg ---------------- </strong><br>'
    for s in random.sample(s_buckets['(maybe)falseneg'], min(len(s_buckets['(maybe)falseneg']), 5)):
        html += '{} --- {} '.format(highlight_words(word_lists, colors, s), str(sen2score[s]['scores']))
        html += '<a href="{}"> {}</a><br><br>'.format(sen2score[s]['link'], sen2score[s]['title'])
        
    html += '</td><td>'
    for k in list(cap2score.keys())[:10]:
        html += '<a href="{}" class="tool-tip" target="_blank"><img style="display:block; max-height:300px; max-width:100%;" src = "{}"></a>'.format(cap2score[k]['img'], cap2score[k]['img'])
        html += '<br>Caption: {}<br>{}<br>'.format(highlight_words(word_lists, colors, k), str(cap2score[k]['scores']))
        html += '<a href="{}"> {}</a><br><br>'.format(cap2score[k]['link'], cap2score[k]['title'])
    html += '</td></tr>'
    html += '<tr><td colspan=4><hr></td></tr>'
    return html.encode('ascii', 'xmlcharrefreplace').decode("utf-8")

### Mining + Save as json
try: upd_txt_data = json.load(open("/home/yingshac/CYS/WebQnA/WebQnA_data_new/upd_txt_data/upd_txt_data_{}.json".format(args.boundary), "r"))
except: upd_txt_data = {}
for k in range(args.start, args.end):
    if k%1 == 0: json.dump(upd_txt_data, open("/home/yingshac/CYS/WebQnA/WebQnA_data_new/upd_txt_data/upd_txt_data_{}.json".format(args.boundary), "w"), indent=4)
    if str(k) in upd_txt_data: continue
    upd_txt_data[str(k)] = copy.deepcopy(new_txt_data[str(k)])
    upd_txt_data[str(k)]['new_negFacts'] = []
    upd_txt_data[str(k)]['img_negFacts'] = []
    try: sen2score, cap2score, word_lists = get_sen2score_from_indx(k)
    except KeyboardInterrupt: raise
    except: continue
    upd_txt_data[str(k)]['word_lists'] = {
        'titlewords': " || ".join(word_lists[0]), 
        'keywords': " || ".join(word_lists[1]), 
        'goldfactwords': " || ".join(word_lists[2]), 
        'answerwords': " || ".join(word_lists[3])
    }
    new_negFacts_count = 0
    for s in sen2score:
        if new_negFacts_count >= 40: break
        if sen2score[s]['scores'][2] == 0.0 and sen2score[s]['scores'][1] == 0.0 and len(s.split()) in range(22, 60):
            upd_txt_data[str(k)]['new_negFacts'].append({
                'title': sen2score[s]['title'],
                'scores': str(sen2score[s]['scores']),
                'fact': s,
                'url': sen2score[s]['link']
            })
            new_negFacts_count += 1
    
    img_negFacts_count = 0
    for c in cap2score:
        if img_negFacts_count >= 40: break
        upd_txt_data[str(k)]['img_negFacts'].append({
            'title': cap2score[c]['title'],
            'scores': str(cap2score[c]['scores']),
            'caption': c,
            'url': cap2score[c]['link'],
            'imgUrl': cap2score[c]['img']
        })
        img_negFacts_count += 1

json.dump(upd_txt_data, open("/home/yingshac/CYS/WebQnA/WebQnA_data_new/upd_txt_data/upd_txt_data_{}.json".format(args.boundary), "w"), indent=4)
print("finish")
'''
### Mining + Create demo
html = "<html><body>"
html += "<style>th {position: sticky; top: 0;background: FloralWhite;}</style>"
html += '<table border="0" style="table-layout: fixed; width: 100%; word-break:break-word">'
html += '<tr bgcolor=gray><th width=5%>Index</th><th width=25%>Q & Pos Snippets</th><th width=40%>Neg Snippets</th><th width=30%>X_modal Facts</th></tr>'
count = 0
samples = [int(k) for k in args.samples.split(',')]
for k in samples:
    count += 1
    sen2score, cap2score, word_lists = get_sen2score_from_indx(k)
    html += add_html_row(k, sen2score, cap2score, word_lists)
    o = open(args.output_filename, 'wt')

    o.write(html)
    o.close()
html += '</table></body></html>'
o = open(args.output_filename, 'wt')

o.write(html)
o.close()
'''