"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append("/home/yingshac/CYS/WebQnA/VLP")

import logging
import glob
import json, time
import argparse
import math
from tqdm import tqdm, trange
from pathlib import Path

import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import random
import pickle
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForWebqaDecoder
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from vlp.loader_utils import batch_list_to_batch_tensors
import vlp.seq2seq_loader as seq2seq_loader
from misc.data_parallel import DataParallelImbalance

import vlp.webqa_loader as webqa_loader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

from datetime import datetime
from pytz import timezone
from word2number import w2n
import re

# SPECIAL_TOKEN = ["[UNK]", "[PAD]", "[CLS]", "[MASK]"]
def toNum(word):
    try: return w2n.word_to_num(word)
    except:
        return word

# Language eval
class Evaluate(object):
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            #(Cider(), "CIDEr"),
            #(Spice(), "Spice")
        ]
    
    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    print(m)
                    final_scores[m] = s
            else:
                print(method)
                final_scores[method] = score
        return final_scores

    def evaluate(self, return_scores=False, **kwargs):
        ans = kwargs.pop('ref', {}) # only support one ans per sample
        cand = kwargs.pop('cand', {}) # only support one cand per sample, but the input cand has size batch_size x K

        hypo = {}
        ref = {}
        i = 0
        for i in range(len(cand)):
            hypo[i] = [cand[i][0]]
            ref[i] = [ans[i]]
        
        final_scores = self.score(ref, hypo)
        print ('Bleu_1:\t', final_scores['Bleu_1'])
        print ('Bleu_2:\t', final_scores['Bleu_2'])
        print ('Bleu_3:\t', final_scores['Bleu_3'])
        print ('Bleu_4:\t', final_scores['Bleu_4'])
        print ('METEOR:\t', final_scores['METEOR'])
        print ('ROUGE_L:', final_scores['ROUGE_L'])
        #print ('CIDEr:\t', final_scores['CIDEr'])
        #print ('Spice:\t', final_scores['Spice'])

        if return_scores:
            return final_scores

# VQA Eval (SQuAD style EM, F1)
def compute_vqa_metrics(cands, a):
    if len(cands) == 0: return (0,0,0)
    bow_a = re.sub(r'[^\w\s\']', '', a).lower().split()
    bow_a = [str(toNum(w)) for w in bow_a]
    F1 = []
    EM = 0
    for c in cands:
        bow_c = re.sub(r'[^\w\s\']', '', c).lower().split()
        bow_c = [str(toNum(w)) for w in bow_c]
        if bow_c == bow_a:
            EM = 1
        #print(bow_a, bow_c)
        overlap = float(len([w for w in bow_c if w in bow_a]))
        precision = overlap/(len(bow_c) + 1e-5)
        recall = overlap / (len(bow_a) + 1e-5)
        f1 = 2*precision*recall / (precision + recall + 1e-5)
        F1.append(f1)
    
    F1_avg = np.mean(F1)
    F1_max = np.max(F1)
    return (F1_avg, F1_max, EM)

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)

def _get_loader_from_dataset(infr_dataset, infr_batch_size, num_workers, collate_fn):
    #train_sampler = RandomSampler(train_dataset, replacement=False)
    print("\nSequentialSampler")
    infr_sampler = SequentialSampler(infr_dataset)

    infr_dataloader = torch.utils.data.DataLoader(infr_dataset,
        batch_size=infr_batch_size, sampler=infr_sampler, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True)
    return infr_dataloader

def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]
                   ) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None

def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased")
    parser.add_argument("--output_dir",
                        default='',
                        type=str,
                        help="The output directory where the model predictions and ckpts will be written")
    parser.add_argument("--output_suffix", default="", type=str)
    parser.add_argument("--log_file",
                        default="training.log",
                        type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path", 
                        default="/home/yingshac/CYS/WebQnA/cpts/cc_g8_lr1e-4_batch512_s0.75_b0.25/model.30.bin",
                        type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument('--from_scratch', action='store_true',
                        help="Initialize parameters with random values (i.e., training from scratch).")
    parser.add_argument('--max_position_embeddings', type=int, default=512, help="max position embeddings")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")

    # For decoding
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--max_tgt_len', type=int, default=30,
                        help="maximum length of target sequence")

    # webqa dataset
    parser.add_argument('--txt_dataset_json_path', type=str, default="/home/yingshac/CYS/WebQnA/VLP/vlp/tmp/txt_json.json")
    parser.add_argument('--img_dataset_json_path', type=str, default="/home/yingshac/CYS/WebQnA/WebQnA_data/dataset_J0501-Copy1.json")
    parser.add_argument('--gold_feature_folder', type=str, default="/data/yingshac/MMMHQA/imgFeatures_upd/gold")
    parser.add_argument('--distractor_feature_folder', type=str, default="/data/yingshac/MMMHQA/imgFeatures_upd/distractors")
    parser.add_argument('--img_metadata_path', type=str, default="/home/yingshac/CYS/WebQnA/WebQnA_data/img_metadata-Copy1.json", help="how many samples should be loaded into memory")
    parser.add_argument('--use_num_samples', type=int, default=-1, help="how many samples should be loaded into memory")
    parser.add_argument('--answer_provided_by', type=str, default="img|txt")

    parser.add_argument("--recover_ori_ckpt", action='store_true',
                        help="Whether to load original VLP checkpoint.")

    parser.add_argument('--no_img_meta', action='store_true')
    parser.add_argument('--no_img_content', action='store_true')

    # Others for VLP
    parser.add_argument("--src_file", default='/mnt/dat/COCO/annotations/dataset_coco.json', type=str,		
                        help="The input data file name.")		
    parser.add_argument('--dataset', default='coco', type=str,
                        help='coco | flickr30k | cc')
    parser.add_argument('--len_vis_input', type=int, default=100)
    parser.add_argument('--max_len_b', type=int, default=109,
                        help="Truncate_config: maximum length of segment B.")
    parser.add_argument('--max_len_a', type=int, default=400,
                        help="Truncate_config: maximum length of segment A.")
    parser.add_argument('--max_len_Q', type=int, default=70,
                        help="Truncate_config: maximum length of Question.")
    parser.add_argument('--max_len_img_cxt', type=int, default=200,
                        help="maximum length of segment image context.")
    parser.add_argument('--trunc_seg', default='b',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', action='store_true',
                        help="Truncate_config: Whether we should always truncate tail.")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers for the data loader.")

    parser.add_argument('--image_root', type=str, default='/mnt/dat/COCO/images')		
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--enable_butd', action='store_true',
                        help='set to take in region features')
    parser.add_argument('--region_bbox_file', default='coco_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5', type=str)
    parser.add_argument('--region_det_file_prefix', default='feat_cls_1000/coco_detection_vg_100dets_gvd_checkpoint_trainval', type=str)
    parser.add_argument('--file_valid_jpgs', default='', type=str)

    args = parser.parse_args()
    args.use_img_meta = not args.no_img_meta
    args.use_img_content = not args.no_img_content
    log_txt_content = []

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("device = ", device)
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(
        filename=os.path.join(args.output_dir, args.log_file),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logger = logging.getLogger(__name__)

    # output config
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'qa_infr'), exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)
    
    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    args.max_seq_length = args.max_len_b + args.max_len_a + 3 # +3 for 2x[SEP] and [CLS]
    tokenizer.max_len = args.max_seq_length

    processor = webqa_loader.Preprocess4webqaDecoder(list(tokenizer.vocab.keys()), \
            tokenizer.convert_tokens_to_ids, max_len=args.max_seq_length, len_vis_input=args.len_vis_input, \
            max_len_a=args.max_len_a, max_len_Q=args.max_len_Q, max_len_img_cxt=args.max_len_img_cxt, \
            max_tgt_len=args.max_tgt_len, new_segment_ids=args.new_segment_ids, \
            truncate_config={'trunc_seg': args.trunc_seg, 'always_truncate_tail': args.always_truncate_tail}, \
            use_img_meta=args.use_img_meta, use_img_content=args.use_img_content)

    infr_dataloaders = []
    if 'txt' in args.answer_provided_by:
        train_dataset = webqa_loader.webqaDataset_qa(dataset_json_path=args.txt_dataset_json_path, split=args.split, \
                batch_size=args.batch_size, tokenizer=tokenizer, use_num_samples=args.use_num_samples, \
                processor=processor, device=device)

        infr_dataloader = _get_loader_from_dataset(train_dataset, args.batch_size, args.num_workers, batch_list_to_batch_tensors)
        infr_dataloaders.append(infr_dataloader)

    if "img" in args.answer_provided_by:
        train_dataset = webqa_loader.webqaDataset_qa_with_img(dataset_json_path=args.img_dataset_json_path, img_metadata_path=args.img_metadata_path, split=args.split, \
                batch_size=args.batch_size, tokenizer=tokenizer, gold_feature_folder=args.gold_feature_folder, \
                distractor_feature_folder=args.distractor_feature_folder, use_num_samples=args.use_num_samples, \
                processor=processor, device=device)
        infr_dataloader = _get_loader_from_dataset(train_dataset, args.batch_size, args.num_workers, batch_list_to_batch_tensors)
        infr_dataloaders.append(infr_dataloader)

    loader_lengths = [len(l) for l in infr_dataloaders]
    print("\nnbatches (all loaders) = ", sum(loader_lengths))

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2 # this is not used by models
    type_vocab_size = 6 if args.new_segment_ids else 2
    mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]"])
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))
    
    recover_step = None
    if len(args.output_dir)>0:
        recover_step = _get_max_epoch_model(args.output_dir)
        print("detect output_dir, recover_step = ", recover_step)
    if args.from_scratch or args.recover_ori_ckpt: recover_step = None
    if args.from_scratch: args.model_recover_path = None

    if (recover_step is None) and (args.model_recover_path is None):
        print("Decoding ... ----------------------- nothing to recover -------------------------")
        log_txt_content.append("Decoding ... ----------------------- nothing to recover -------------------------")
        logger.info("*****Decoding ...  Nothing to recover *****")
        _state_dict = {}
        model = BertForWebqaDecoder.from_pretrained(args.bert_model,
            max_position_embeddings=args.max_position_embeddings, config_path=args.config_path,
            state_dict=_state_dict, num_labels=cls_num_labels, type_vocab_size=type_vocab_size, 
            task_idx=3, mask_word_id=mask_word_id, search_beam_size=args.beam_size, 
            length_penalty=args.length_penalty, eos_id=eos_word_ids, 
            forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set, 
            ngram_size=args.ngram_size, min_len=args.min_len, max_len_img_cxt=args.max_len_img_cxt)
    else:
        if recover_step:
            print("Decoding ... -------------------- recover from step {} -----------------------".format(recover_step))
            log_txt_content.append("Decoding ... -------------------- recover from step {} -----------------------".format(recover_step))
            logger.info("*****Decoding ...  Recover model: %d *****", recover_step)
            model_recover = torch.load(os.path.join(args.output_dir, "model.{0}.bin".format(recover_step)))
        elif args.model_recover_path:
            print("Decoding ... ------------------ recover from path ----------------------")
            log_txt_content.append("Decoding ... ------------------ recover from path ----------------------")
            logger.info("*****Decoding ...  Recover model: %s *****", args.model_recover_path)
            model_recover = torch.load(args.model_recover_path)

    #for model_recover_path in glob.glob(args.model_recover_path.strip()): # ori repo: can do infr on multiple ckpts at once
        model = BertForWebqaDecoder.from_pretrained(args.bert_model,
            max_position_embeddings=args.max_position_embeddings, config_path=args.config_path, 
            state_dict=model_recover, num_labels=cls_num_labels, type_vocab_size=type_vocab_size, 
            task_idx=3, mask_word_id=mask_word_id, search_beam_size=args.beam_size, 
            length_penalty=args.length_penalty, eos_id=eos_word_ids, 
            forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set, 
            ngram_size=args.ngram_size, min_len=args.min_len, max_len_img_cxt=args.max_len_img_cxt)
        del model_recover

        
    if args.fp16:
        model.half()
    model.to(device)
    #if n_gpu > 1:
        #model = torch.nn.DataParallel(model)

    torch.cuda.empty_cache()
    model.eval()
    print("-------------------- QA Decoding ------------------------")
    log_txt_content.append("-------------------- QA Decoding ------------------------")
    print(args.use_img_meta)
    print(args.use_img_content)
    log_txt_content.append("use_img_content = {}".format(args.use_img_content))
    log_txt_content.append("use_img_meta = {}".format(args.use_img_meta))
    log_txt_content.append("split = {}".format(args.split))
    log_txt_content.append("use_num_samples = {}".format(args.use_num_samples))
    
    output_lines = []
    output_confidence = []
    for infr_dataloader in infr_dataloaders:
        nbatches = len(infr_dataloader)
        
        iter_bar = tqdm(infr_dataloader, desc = 'Step = X')
        for step, batch in enumerate(iter_bar):
            #if step<834: continue
            with torch.no_grad():
                batch = [t.to(device) for t in batch]
                input_ids, segment_ids, position_ids, input_mask, task_idx, img, vis_pe, context_is_img = batch
                #print(tokenizer.convert_ids_to_tokens([i for i in list(input_ids.detach().cpu().numpy()[0][202:]) if i>0 and i!=102]))
                time.sleep(2)
                if args.fp16:
                    img = img.half()
                    vis_pe = vis_pe.half()

                conv_feats = img.data # Bx100x2048
                vis_pe = vis_pe.data

                traces = model(conv_feats, vis_pe, input_ids, segment_ids, position_ids, input_mask, context_is_img, task_idx=task_idx)
                    
                #if args.beam_size > 1:
                    #traces = {k: v.tolist() for k, v in traces.items()}
                    #output_ids = traces['pred_seq']
                #else:
                    #output_ids = traces[0].tolist()

                for i in range(input_ids.size(0)):
                    output_sequences = []
                    output_confidence.append(str(list(traces[i].keys())))
                    for w_ids in traces[i].values():
                        output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                        #output_buf = tokenizer.convert_ids_to_tokens([i for i in w_ids if i>0 and i!=102])
                        output_tokens = []
                        for t in output_buf:
                            if t in ("[SEP]", "[PAD]"):
                                break
                            output_tokens.append(t)
                        output_sequences.append(' '.join(detokenize(output_tokens)))
                    
                    output_lines.append(output_sequences)
                    #print("\noutput_sequence for cur batch = ", output_sequence)
                    # buf_id[i] = img_idx (global idx across chunks)
                iter_bar.set_description('Step = {}'.format(step))

        Q, A = infr_dataloader.dataset.get_QA_list()
        Q, A = [' '.join(detokenize(q)) for q in Q], [' '.join(detokenize(a)) for a in A]
        assert len(Q) == len(A) == len(output_lines) == len(output_confidence)
        
        predictions = zip(Q, A, output_lines)
        #for q, a, o in predictions:
            #print(q)
            #print(a)
            #print(o)
        eval_f = Evaluate()
        scores = eval_f.evaluate(cand=output_lines, ref=A, return_scores=True)

        # SQuAD style vqa eval: EM, F1
        F1_avg_scores = []
        F1_max_scores = []
        EM_scores = []
        for cands, a in zip(output_lines, A):
            assert len(cands)==args.beam_size
            F1_avg, F1_max, EM = compute_vqa_metrics(cands, a)
            F1_avg_scores.append(F1_avg)
            F1_max_scores.append(F1_max)
            EM_scores.append(EM)
        F1_avg = np.mean(F1_avg_scores)
        F1_max = np.mean(F1_max_scores)
        EM = np.mean(EM_scores)
        print("F1_avg = {}".format(F1_avg))
        print("F1_max = {}".format(F1_max))
        print("EM = {}".format(EM))

        with open(os.path.join(args.output_dir, 'qa_infr', args.split+"_qainfr_beam{}_{}_{}_{}.txt".format(args.beam_size, args.use_img_content, args.use_img_meta, args.output_suffix)), "w") as f:
            f.write(datetime.now(tz=timezone('US/Eastern')).strftime("%y-%m-%d %H:%M:%S") + '\n')
            f.write("\n".join(log_txt_content))
            f.write('\n --------------------- metrics -----------------------\n')
            f.write(str(scores))
            f.write('\n\n')
            f.write('\n'.join(["F1_avg = {}".format(F1_avg), "F1_max = {}".format(F1_max), "EM = {}".format(EM)]))
            f.write('\n\n')
            f.write('-----Starting writing results:-----')
            for q, a, oc, o in zip(Q, A, output_confidence, output_lines):
                f.write("\n\n")
                f.write("\n".join([q, a, oc, '\n'.join(o)]))
                

if __name__ == "__main__":
    main()
