import math
import json, os, base64
import os.path as op
import numpy as np
import torch

class TSVFile(object):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        self.lineidx = tsv_file.replace('.tsv', '.lineidx')
        self._fp = None
        self._lineidx = None
        # the process always keeps the process which opens the file. 
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None


    def __del__(self):
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[idx]
        except:
            #logging.info('{}-{}'.format(self.tsv_file, idx))
            print("\nseek error, lineidx file = {}, with {} lines, idx = {}".format(self.tsv_file, len(self._lineidx), idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_first_column(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        row = self._fp.readline().strip().split('\t')
        return int(row[0])

    def get_imgid(self, idx):
        return self.seek_first_column(idx)

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            print('loading lineidx: {}'.format(self.lineidx))
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]
                print("\nlineidx file {} is open, find {} lines".format(self.tsv_file, len(self._lineidx)))

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            print('re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

class ImgDataTsv(object):
    def __init__(self, img_file):
        self.img_file = img_file
        self.img_tsv = TSVFile(img_file)

    def __len__(self):
        return self.img_tsv.num_rows() # TODO:
    
    def __getitem__(self, idx):
        V = self.get_prediction(idx)
        pred_boxes_list = []
        scores_list = []
        fc1_features_list = []
        cls_features_list = []
        for v in V:
            pred_boxes_list.append(v['rect'])
            scores_list.append(v['conf'])
            fc1_features_list.append(np.frombuffer(base64.b64decode(v['feature']), np.float32))
            cls_features_list.append(np.frombuffer(base64.b64decode(v['scores_all']), np.float32))
        pred_boxes = torch.FloatTensor(pred_boxes_list)
        scores = torch.FloatTensor(scores_list)
        fc1_features = torch.FloatTensor(fc1_features_list)
        cls_features = torch.FloatTensor(cls_features_list)
        return pred_boxes, scores, fc1_features, cls_features

    def get_imgid(self, idx):
        return self.img_tsv.get_imgid(idx)

    def get_prediction(self, idx): 
        row = self.img_tsv.seek(idx)
        # row = [s.strip() for s in self._fp.readline().split('\t')]
        assert not row[0] == 'None' "Trying to access a non-existing image_id={} from file {}".format(idx, self.img_file)
        pred = json.loads(row[-1])['objects']
        return pred
