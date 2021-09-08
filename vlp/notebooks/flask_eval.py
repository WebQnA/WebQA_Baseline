import numpy as np
import os
import json, time, copy
import math

from tqdm import tqdm
import random
import pickle
from datetime import datetime
from pytz import timezone
from collections import Counter, defaultdict
from pprint import pprint

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--Qcate_breakdown", type=str, default='["all"]')
parser.add_argument("--file", type=str)
args = parser.parse_args()

with open(os.path.join("/home/yingshac/CYS/WebQnA/VLP/vlp/light_output/", args.file), "r") as fp:
    lines = fp.readlines()
    header = lines[0].strip().split('\t')
    rows = lines[1:]
key = dict(zip(header, range(len(header))))
pprint(key)

output_A = []
output_O = []
output_KA = []
output_QC = []
# Guid	Qcate	Q	A	Keywords_A	Output_conf	Output
for r in tqdm(rows):
    
    datum = r.strip().split('\t')
    Qcate = datum[key['Qcate']]
    if (not 'all' in Qcate_breakdown) and (not Qcate in Qcate_breakdown): continue

    O = json.loads(datum[key['Output']])
    C = [O[0]]
    Keywords_A = datum[key['Keywords_A']]
    A = json.loads(datum[key['A']])
    
    output_A.append(A)
    output_O.append(O)
    output_KA.append(Keywords_A)
    output_QC.append(Qcate)
    
params = {'A': json.dumps(output_A), 'C': json.dumps(output_O), 'Keyword': json.dumps(output_KA), 'Qcate': json.dumps(output_QC)}
result = request(params).json()
