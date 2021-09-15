"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '12355'
import sys
sys.path.append("/home/yingshac/CYS/WebQnA/VLP")
sys.path.append("/home/yingshan/CYS/WebQnA/VLP")
import logging
import glob
import math, time
import json, pickle
import argparse
from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import random
import copy

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForWebqa
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from vlp.loader_utils import batch_list_to_batch_tensors
import vlp.webqa_VinVL_loader as webqa_VinVL_loader
from misc.data_parallel import DataParallelImbalance
import matplotlib.pyplot as plt
from datetime import datetime
from pytz import timezone





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

def _get_loader_from_dataset(train_dataset, world_size, train_batch_size, num_workers, collate_fn):
    if world_size == 1:
        print("\nRandomSampler")
        train_sampler = RandomSampler(train_dataset, replacement=False)
        
    else:
        print("\nDistributedSampler")
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, sampler=train_sampler, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True)
    return train_dataloader, train_sampler

def save_loss_curve(loss, i_epoch, output_dir, task, all_tasks):
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(loss)+1), loss)
    plt.xlabel("iter")
    plt.ylabel("loss")
    title = "{}__epc={}__all_tasks={}".format(task, i_epoch, all_tasks)
    plt.title(title)
    plt.savefig(os.path.join(output_dir, "figs/"+title+".jpg"))

def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--ckpts_dir",
                        default='/data/yingshac/MMMHQA/ckpts/no_model_name_specified/',
                        type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--output_dir",
                        default='light_output/no_model_name_specified/',
                        type=str,
                        help="The output directory where the model predictions and loss curves.")
    parser.add_argument("--log_file",
                        default="training.log",
                        type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path",
                        #default=None,
                        default="/home/yingshac/CYS/WebQnA/cpts/cc_g8_lr1e-4_batch512_s0.75_b0.25/model.30.bin",
                        type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training. This should ALWAYS be set to True.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--finetune_decay",
                        action='store_true',
                        help="Weight decay to the original weights.")
    parser.add_argument("--num_train_epochs",
                        default=30,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--global_rank",
                        type=int,
                        default=-1,
                        help="global_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp32_embedding', action='store_true',
                        help="Whether to use 32-bit float precision instead of 32-bit for embeddings")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--from_scratch', action='store_true',
                        help="Initialize parameters with random values (i.e., training from scratch).")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--len_vis_input', type=int, default=100,
                        help="The length of visual token input")
    parser.add_argument('--max_len_b', type=int, default=109,
                        help="Truncate_config: maximum length of segment B.")
    parser.add_argument('--max_len_a', type=int, default=400,
                        help="Truncate_config: maximum length of segment A.")
    parser.add_argument('--max_len_img_cxt', type=int, default=200,
                        help="maximum length of segment image context.")
    parser.add_argument('--trunc_seg', default='b',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', action='store_true',
                        help="Truncate_config: Whether we should always truncate tail.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=3,
                        help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Number of workers for the data loader.")
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                        help="max position embeddings")

    # webqa dataset
    parser.add_argument('--txt_dataset_json_path', type=str, default="/home/yingshac/CYS/WebQnA/WebQnA_data_new/txt_dataset_0904_clean_fields.json")
    parser.add_argument('--img_dataset_json_path', type=str, default="/home/yingshac/CYS/WebQnA/WebQnA_data_new/img_dataset_0904_clean_fields.json")
    parser.add_argument('--use_num_samples', type=int, default=-1, help="how many samples should be loaded into memory")
    parser.add_argument('--answer_provided_by', type=str, default="img|txt")
    parser.add_argument('--use_x_distractors', action='store_true')
    parser.add_argument('--task_to_learn', type=str, default="filter|qa")

    parser.add_argument('--txt_filter_max_choices', type=int, default=16)
    parser.add_argument('--img_filter_max_choices', type=int, default=16)
    parser.add_argument('--filter_infr_log', type=str, default="filter_infr_log.txt")
    parser.add_argument("--recover_ori_ckpt", action='store_true',
                        help="Whether to load original VLP checkpoint.")
    parser.add_argument("--recover_step", type=int, default=None)
    parser.add_argument("--val_loss", action='store_true')

    parser.add_argument('--no_img_meta', action='store_true')
    parser.add_argument('--no_img_content', action='store_true')
    parser.add_argument('--no_txt_fact', action='store_true')
    parser.add_argument('--filter_infr_th', type=str, default="0.05|0.1|0.15|0.2|0.25|0.3|0.35|0.4|0.45|0.5|0.55|0.6|0.65|0.7|0.75|0.8|0.85|0.9|0.95")

    parser.add_argument("--output_suffix", default="", type=str)
    
    # Others for VLP
    parser.add_argument('--save_loss_curve', action='store_true')
    parser.add_argument('--split', type=str, default=['train', 'val', 'test'])

    # available Qcate in img data: {'YesNo': 8432, 'Others': 6748, 'choose': 5240, 'number': 2341, 'color': 2044, 'shape': 662}
    # available Qcate in txt data: ### TBD
    parser.add_argument('--Qcate', type=str, default=['all'])

    # Add VinVL feature support
    parser.add_argument('--gold_img_tsv', default="/data/yingshac/MMMHQA/VinVL_output/gold_0_22265/gold_vinvl.tsv", type=str)
    parser.add_argument('--neg_img_tsv', default="/data/yingshac/MMMHQA/VinVL_output/neg_imgs_0_338842/distractors_vinvl.tsv", type=str)
    parser.add_argument('--x_neg_img_tsv', default="/data/yingshac/MMMHQA/VinVL_output/x_neg_imgs_0_240661/x_distractors_vinvl.tsv", type=str)
    parser.add_argument('--vis_emb_ft_epc', default=0, type=int)
    
    parser.add_argument('--world_size', default = 1, type = int,
                        help = 'number of distributed processes')
    parser.add_argument('--dist_url', default='file://[PT_OUTPUT_DIR]/nonexistent_file', type = str,
                        help = 'url used to set up distributed training')
    parser.add_argument('--file_valid_jpgs', default='/mnt/dat/COCO/annotations/coco_valid_jpgs.json', type=str)
    parser.add_argument('--sche_mode', default='warmup_linear', type=str,
                        help="warmup_linear | warmup_constant | warmup_cosine")
    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--use_num_imgs', default=-1, type=int)
    parser.add_argument('--vis_mask_prob', default=0, type=float)
    parser.add_argument('--max_drop_worst_ratio', default=0, type=float)
    parser.add_argument('--drop_after', default=6, type=int)

    parser.add_argument('--enable_butd', action='store_true',
                        help='set to take in region features')

    parser.add_argument('--relax_projection',
                        action='store_true',
                        help="Use different projection layers for tasks.")
    parser.add_argument('--scst', action='store_true',
                        help='Self-critical sequence training')

    args = parser.parse_args()

    log_txt_content = []
    print('global_rank: {}, local rank: {}'.format(args.global_rank, args.local_rank))
    args.max_seq_length = args.max_len_b + args.max_len_a + 3 # +3 for 2x[SEP] and [CLS]
    args.dist_url = args.dist_url.replace('[PT_OUTPUT_DIR]', args.output_dir)
    args.use_img_meta = not args.no_img_meta
    args.use_img_content = not args.no_img_content
    args.use_txt_fact= not args.no_txt_fact
    assert args.len_vis_input == 100, "run main: only support 100 region features per image"
    assert args.output_dir.split('/')[-1] == args.ckpts_dir.split('/')[-1], "Warning: folder names for output & ckpts are not the same"
    # output config
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpts_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figs"), exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)
    
    logging.basicConfig(
        filename=os.path.join(args.output_dir, args.log_file),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        port = '12355'
        torch.distributed.init_process_group(backend='nccl', init_method = 'tcp://128.2.205.68:{}'.format(port),
            world_size=args.world_size, rank=args.global_rank)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case,
        cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.global_rank))
    if args.max_position_embeddings:
        tokenizer.max_len = args.max_position_embeddings
    # doesn't support WhitespaceTokenizer


    ImgDataTsv_dict = {}
    if args.gold_img_tsv is not None: ImgDataTsv_dict[0] = args.gold_img_tsv
    if args.neg_img_tsv is not None: ImgDataTsv_dict[1] = args.neg_img_tsv
    if args.x_neg_img_tsv is not None: ImgDataTsv_dict[2] = args.x_neg_img_tsv
    processor = webqa_VinVL_loader.Preprocess4webqa_VinVL(args.max_pred, args.mask_prob, \
        list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, seed=args.seed, max_len=args.max_seq_length, \
        len_vis_input=args.len_vis_input, max_len_a=args.max_len_a, max_len_b=args.max_len_b, \
        max_len_img_cxt=args.max_len_img_cxt, new_segment_ids=args.new_segment_ids, \
        truncate_config={'trunc_seg': args.trunc_seg, 'always_truncate_tail': args.always_truncate_tail}, \
        use_img_meta=args.use_img_meta, use_img_content=args.use_img_content, use_txt_fact=args.use_txt_fact, ImgDataTsv_dict = ImgDataTsv_dict)
    
    
    train_dataloaders = []
    train_samplers = []
    if "filter" in args.task_to_learn:
        if "txt" in args.answer_provided_by:
            if args.use_x_distractors:
                train_dataset = webqa_VinVL_loader.webqaDataset_filter_with_both(dataset_json_path=args.txt_dataset_json_path, \
                    split=args.split, Qcate=args.Qcate, \
                    batch_size=args.train_batch_size, tokenizer=tokenizer, use_num_samples=args.use_num_samples, \
                    processor=processor, answer_provided_by='txt', max_snippets=args.txt_filter_max_choices, max_imgs=args.img_filter_max_choices, device=device)
            else:
                train_dataset = webqa_VinVL_loader.webqaDataset_filter(dataset_json_path=args.txt_dataset_json_path, split=args.split, Qcate=args.Qcate, \
                    batch_size=args.train_batch_size, tokenizer=tokenizer, use_num_samples=args.use_num_samples, \
                    processor=processor, filter_max_choices=args.txt_filter_max_choices, device=device)

            train_dataloader, train_sampler = _get_loader_from_dataset(train_dataset, args.world_size, args.train_batch_size, args.num_workers, batch_list_to_batch_tensors)
            train_dataloaders.append(train_dataloader)
            train_samplers.append(train_sampler)
        
        if "img" in args.answer_provided_by:
            if args.use_x_distractors:
                train_dataset = webqa_VinVL_loader.webqaDataset_filter_with_both(dataset_json_path=args.img_dataset_json_path, \
                    split=args.split, Qcate=args.Qcate, \
                    batch_size=args.train_batch_size, tokenizer=tokenizer, use_num_samples=args.use_num_samples, \
                    processor=processor, answer_provided_by='img', max_snippets=args.txt_filter_max_choices, max_imgs=args.img_filter_max_choices, device=device)
            else:
                train_dataset = webqa_VinVL_loader.webqaDataset_filter_with_img(dataset_json_path=args.img_dataset_json_path, split=args.split, Qcate=args.Qcate, \
                    batch_size=args.train_batch_size, tokenizer=tokenizer, use_num_samples=args.use_num_samples, \
                    processor=processor, filter_max_choices=args.img_filter_max_choices, device=device)
            train_dataloader, train_sampler = _get_loader_from_dataset(train_dataset, args.world_size, args.train_batch_size, args.num_workers, batch_list_to_batch_tensors)
            train_dataloaders.append(train_dataloader)
            train_samplers.append(train_sampler)
    
    if "qa" in args.task_to_learn:
        if "txt" in args.answer_provided_by:
            train_dataset = webqa_VinVL_loader.webqaDataset_qa(dataset_json_path=args.txt_dataset_json_path, split=args.split, Qcate=args.Qcate, \
                    batch_size=args.train_batch_size, tokenizer=tokenizer, use_num_samples=args.use_num_samples, \
                    processor=processor, device=device)

            train_dataloader, train_sampler = _get_loader_from_dataset(train_dataset, args.world_size, args.train_batch_size, args.num_workers, batch_list_to_batch_tensors)
            train_dataloaders.append(train_dataloader)
            train_samplers.append(train_sampler)
        
        if "img" in args.answer_provided_by:
            train_dataset = webqa_VinVL_loader.webqaDataset_qa_with_img(dataset_json_path=args.img_dataset_json_path, split=args.split, Qcate=args.Qcate, \
                    batch_size=args.train_batch_size, tokenizer=tokenizer, use_num_samples=args.use_num_samples, \
                    processor=processor, device=device)
            train_dataloader, train_sampler = _get_loader_from_dataset(train_dataset, args.world_size, args.train_batch_size, args.num_workers, batch_list_to_batch_tensors)
            train_dataloaders.append(train_dataloader)
            train_samplers.append(train_sampler)

        
    loader_lengths = [len(l) for l in train_dataloaders]
    print("\nnbatches = ", sum(loader_lengths))
    train_dataloader_order = []
    for i in range(len(loader_lengths)):
        train_dataloader_order.extend([i] * loader_lengths[i])
    random.shuffle(train_dataloader_order)
    print("\ntrain_dataloader_order = ", train_dataloader_order)
    # The actual number of params updates
    t_total = int(sum(loader_lengths) * args.num_train_epochs * 1. / args.gradient_accumulation_steps)

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    recover_step = _get_max_epoch_model(args.ckpts_dir)
    if args.recover_step: recover_step = args.recover_step
    if args.recover_ori_ckpt or args.from_scratch: recover_step = None
    if args.from_scratch: args.model_recover_path = None


    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2
    relax_projection = 4 if args.relax_projection else 0
    task_idx_proj = 3 # harded to be 3 # if args.tasks == 'img2txt' else 0
    mask_word_id, eos_word_ids, pad_word_ids = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]", "[PAD]"]) # index in BERT vocab: 103, 102, 0

    # Recover model
    if (recover_step is None) and (args.model_recover_path is None):
        print("----------------------- nothing to recover -------------------------")
        log_txt_content.append("----------------------- nothing to recover -------------------------")
        assert args.scst == False, 'must init from maximum likelihood training'
        _state_dict = {} if args.from_scratch else None
        model = BertForWebqa.from_pretrained(
            args.bert_model, state_dict=_state_dict, num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, relax_projection=relax_projection,
            config_path=args.config_path, task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,
            fp32_embedding=args.fp32_embedding, cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.global_rank),
            drop_prob=args.drop_prob, max_len_img_cxt=args.max_len_img_cxt, use_vinvl=True)
        global_step = 0
    else:
        if recover_step:
            print("-------------------- recover from step {} -----------------------".format(recover_step))
            log_txt_content.append("-------------------- recover from step {} -----------------------".format(recover_step))
            logger.info("***** Recover model: %d *****", recover_step)
            model_recover = torch.load(os.path.join(
                args.ckpts_dir, "model.{0}.bin".format(recover_step)))
            # recover_step == number of epochs
            global_step = math.floor(
                recover_step * t_total * 1. / args.num_train_epochs)
        elif args.model_recover_path:
            print("------------------ recover from path ----------------------")
            log_txt_content.append("------------------ recover from path ----------------------")
            logger.info("***** Recover model: %s *****",
                        args.model_recover_path)
            model_recover = torch.load(args.model_recover_path)
            global_step = 0
        if not args.scst:
            model = BertForWebqa.from_pretrained(
                args.bert_model, state_dict=model_recover, num_labels=cls_num_labels,
                type_vocab_size=type_vocab_size, relax_projection=relax_projection,
                config_path=args.config_path, task_idx=task_idx_proj,
                max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,
                fp32_embedding=args.fp32_embedding, cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.global_rank),
                drop_prob=args.drop_prob, max_len_img_cxt=args.max_len_img_cxt, use_vinvl=True)
        else:
            raise NotImplementedError

        del model_recover
        torch.cuda.empty_cache()

    if args.fp16:
        model.half()
        if args.fp32_embedding:
            model.bert.embeddings.word_embeddings.float()
            model.bert.embeddings.position_embeddings.float()
            model.bert.embeddings.token_type_embeddings.float()
    print("model.to(device)")
    model.to(device)
    if args.local_rank != -1:
        try:
            # from apex.parallel import DistributedDataParallel as DDP
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model, device_ids = [args.local_rank], output_device = args.local_rank, find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=[1, 0])
        print("\nn_gpu = ", n_gpu)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            # from apex.optimizers import FP16_Optimizer
            from pytorch_pretrained_bert.optimization_fp16 import FP16_Optimizer_State
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer_State(
                optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer_State(
                optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        #optimizer = BertAdam(optimizer_grouped_parameters,
                             #lr=args.learning_rate,
                             #warmup=args.warmup_proportion,
                             #schedule=args.sche_mode,
                             #t_total=t_total,
                             #weight_decay = args.weight_decay)

    if recover_step:
        logger.info("***** Recover optimizer: %d *****", recover_step)
        optim_recover = torch.load(os.path.join(
            args.ckpts_dir, "optim.{0}.bin".format(recover_step)))
        if hasattr(optim_recover, 'state_dict'):
            optim_recover = optim_recover.state_dict()
        optimizer.load_state_dict(optim_recover)
        if args.loss_scale == 0:
            logger.info("***** Recover optimizer: dynamic_loss_scale *****")
            optimizer.dynamic_loss_scale = True

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    if args.do_train:
        print("start training")
        if "img" in args.answer_provided_by:
            print("use_img_meta = ", args.use_img_meta)
            print("use_img_content = ", args.use_img_content)
            print("\nimg Filter_max_choices: {}".format(args.img_filter_max_choices))
            if args.use_x_distractors: print("\ntxt Filter_max_choices: {}".format(args.txt_filter_max_choices))
        if "txt" in args.answer_provided_by:
            print("use_txt_fact = ", args.use_txt_fact)
            print("\ntxt Filter_max_choices: {}".format(args.txt_filter_max_choices))
            if args.use_x_distractors: print("\nimg Filter_max_choices: {}".format(args.img_filter_max_choices))

        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)
        logger.info("  Loader length = %d", len(train_dataloader))

        model.train()
        if recover_step:
            start_epoch = recover_step+1
        else:
            start_epoch = 1
        

        for i_epoch in trange(start_epoch, args.num_train_epochs+1, desc="Epoch"):

            print("\nstart epoch ", i_epoch)
            if i_epoch <= args.vis_emb_ft_epc:
                print("Train vis_embed and vis_pe_embed only, freeze all params elsewhere")
                for name, p in model.named_parameters():
                    if not name in ['vis_embed.0.weight', 'vis_embed.0.bias', 'vis_pe_embed.0.weight', 'vis_pe_embed.0.bias']:
                        p.requires_grad = False
                print("\n---------------- Start printing parameter names -----------")
                for name, p in model.named_parameters():
                    print(name, p.requires_grad)
            else:
                for name, p in model.named_parameters(): p.requires_grad = True
                #print("\n---------------- Start printing parameter names -----------")
                #for name, p in model.named_parameters():
                    #print(name, p.requires_grad)

            
            dataloader_iters = [iter(l) for l in train_dataloaders]
            if args.local_rank >= 0:
                for train_sampler in train_samplers:
                    train_sampler.set_epoch(i_epoch-1)
            iter_bar = tqdm(train_dataloader_order, desc='Iter (loss=X.XXX), loader_idx=X') 
            nbatches = sum(loader_lengths)
            
            qa_loss = []
            filter_loss = []
            loss_dict = [[],[],[],[]]
            scst_reward = []
            for step, loader_idx in enumerate(iter_bar):
                batch = next(dataloader_iters[loader_idx])
                for param_tensor in model.state_dict():
                    if torch.isnan(model.state_dict()[param_tensor]).any().item():
                        print("\n nan exists in ", param_tensor)
                batch = [t.to(device) if not isinstance(t, list) else t for t in batch ]

                input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next, do_filter_task, filter_label, logit_mask, ori_choices, task_idx, img, vis_pe, context, cxt_modality_label, example_ids = batch
                if args.fp16:
                    img = img.half()
                    vis_pe = vis_pe.half()

                conv_feats = img.data # Bx100x2048
                vis_pe = vis_pe.data
                # doesn't support scst training for not
                loss_tuple = model(vis_feats=conv_feats, vis_pe=vis_pe, input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, \
                        masked_lm_labels=masked_ids, do_filter_task=do_filter_task, filter_label=filter_label, logit_mask=logit_mask, context=context, \
                        cxt_modality_label=cxt_modality_label, next_sentence_label=is_next, masked_pos=masked_pos, masked_weights=masked_weights, \
                        task_idx=task_idx, drop_worst_ratio=args.max_drop_worst_ratio if i_epoch > args.drop_after else 0)
                mean_reward = loss_tuple[0].new(1).fill_(0)

                # disable pretext_loss_deprecated for now
                masked_lm_loss, cls_loss = loss_tuple
                if n_gpu > 1:    # mean() to average on multi-gpu. For dist, this is done through gradient addition.
                    masked_lm_loss = masked_lm_loss.mean()
                    cls_loss = cls_loss.mean()
                loss = masked_lm_loss + cls_loss
                # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
                iter_bar.set_description('Iter (loss={:.3f}) loader_idx={}'.format(loss.item(), loader_idx))
                qa_loss.append(masked_lm_loss.item())
                filter_loss.append(cls_loss.item())
                loss_dict[loader_idx].append(loss.item())
                scst_reward.append(mean_reward.item())
                
                if step%100 == 0:
                    logger.info("Epoch {}, Iter {}, Loss {:.2f}, Filter {:.2f}, Mean R {:.3f}\n".format(i_epoch, step, np.mean(qa_loss), np.mean(filter_loss), np.mean(scst_reward)))


                # ensure that accumlated gradients are normalized
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                    if amp_handle:
                        amp_handle._clear_cache()
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    lr_this_step = args.learning_rate * \
                        warmup_linear(global_step/t_total,
                                      args.warmup_proportion)
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                

            print(qa_loss)
            print(filter_loss)
            print(loss_dict)
            
            # Save a trained model
            logger.info(
                "** ** * Saving fine-tuned model and optimizer ** ** * ")
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                args.ckpts_dir, "model.{0}.bin".format(i_epoch))
            output_optim_file = os.path.join(
                args.ckpts_dir, "optim.{0}.bin".format(i_epoch))
            if args.global_rank in (-1, 0): # save model if the first device or no dist
                torch.save(copy.deepcopy(model_to_save).cpu().state_dict(), output_model_file)
                torch.save(optimizer.state_dict(), output_optim_file) # disable for now, need to sanitize state and ship everthing back to cpu
            # Save loss curve
            if args.save_loss_curve:
                loss_idx = 0
                if "filter" in args.task_to_learn and "txt" in args.answer_provided_by:
                    save_loss_curve(loss_dict[loss_idx], i_epoch, args.output_dir, "filter-txt", "-".join([args.task_to_learn, args.answer_provided_by]))
                    loss_idx += 1
                if "filter" in args.task_to_learn and "img" in args.answer_provided_by:
                    save_loss_curve(loss_dict[loss_idx], i_epoch, args.output_dir, "filter-img", "-".join([args.task_to_learn, args.answer_provided_by]))
                    loss_idx += 1
                if "qa" in args.task_to_learn and "txt" in args.answer_provided_by:
                    save_loss_curve(loss_dict[loss_idx], i_epoch, args.output_dir, "qa-txt", "-".join([args.task_to_learn, args.answer_provided_by]))
                    loss_idx += 1
                if "qa" in args.task_to_learn and "img" in args.answer_provided_by:
                    save_loss_curve(loss_dict[loss_idx], i_epoch, args.output_dir, "qa-img", "-".join([args.task_to_learn, args.answer_provided_by]))
                    loss_idx += 1
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

            if args.world_size > 1:
                torch.distributed.barrier()
        # cleanup
        if args.local_rank == -1 or args.no_cuda: pass
        else: torch.distributed.destroy_process_group()
    elif args.val_loss:
        print("--------------- Compute loss without grad ------------------")
        assert args.split == "val"
        if "img" in args.answer_provided_by:
            print("use_img_meta = ", args.use_img_meta)
            print("use_img_content = ", args.use_img_content)
            print("\nimg Filter_max_choices: {}".format(args.img_filter_max_choices))
        if "txt" in args.answer_provided_by:
            print("use_txt_fact = ", args.use_txt_fact)
            print("\ntxt Filter_max_choices: {}".format(args.txt_filter_max_choices))

        logger.info("***** Compute loss without grad *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)
        logger.info("  Loader length = %d", len(train_dataloader))

        model.eval()
        
        with torch.no_grad():
            dataloader_iters = [iter(l) for l in train_dataloaders]
            
            iter_bar = tqdm(train_dataloader_order, desc='Iter (loss=X.XXX), loader_idx=X') 
            nbatches = sum(loader_lengths)
            
            qa_loss = []
            filter_loss = []
            loss_dict = [[],[],[],[]]
            scst_reward = []
            for step, loader_idx in enumerate(iter_bar):
                
                batch = next(dataloader_iters[loader_idx])

                for param_tensor in model.state_dict():
                    if torch.isnan(model.state_dict()[param_tensor]).any().item():
                        print("\n nan exists in ", param_tensor)
                batch = [t.to(device) if not isinstance(t, list) else t for t in batch ]
                input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next, do_filter_task, filter_label, logit_mask, ori_choices, task_idx, img, vis_pe, context, cxt_modality_label, example_ids = batch
                if args.fp16:
                    img = img.half()
                    vis_pe = vis_pe.half()
                
                conv_feats = img.data # Bx100x2048
                vis_pe = vis_pe.data
                # doesn't support scst training for not
                loss_tuple = model(vis_feats=conv_feats, vis_pe=vis_pe, input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, \
                    masked_lm_labels=masked_ids, do_filter_task=do_filter_task, filter_label=filter_label, logit_mask=logit_mask, context=context, \
                        cxt_modality_label=cxt_modality_label, next_sentence_label=is_next, masked_pos=masked_pos, masked_weights=masked_weights, \
                        task_idx=task_idx, drop_worst_ratio=0)
                mean_reward = loss_tuple[0].new(1).fill_(0)

                # disable pretext_loss_deprecated for now
                masked_lm_loss, cls_loss = loss_tuple
                if n_gpu > 1:    # mean() to average on multi-gpu. For dist, this is done through gradient addition.
                    masked_lm_loss = masked_lm_loss.mean()
                    cls_loss = cls_loss.mean()
                loss = masked_lm_loss + cls_loss
                # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
                iter_bar.set_description('Iter (loss={:.3f}) loader_idx={}'.format(loss.item(), loader_idx))
                qa_loss.append(masked_lm_loss.item())
                filter_loss.append(cls_loss.item())
                loss_dict[loader_idx].append(loss.item())
                scst_reward.append(mean_reward.item())
                
                if step%100 == 0:
                    logger.info("Iter {}, Loss {:.2f}, Filter {:.2f}, Mean R {:.3f}\n".format(step, np.mean(qa_loss), np.mean(filter_loss), np.mean(scst_reward)))

                # ensure that accumlated gradients are normalized
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                
            print(qa_loss)
            print(filter_loss)
            print(loss_dict)
            print("Mean loss = ", np.mean([l for L in loss_dict for l in L]))
            
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

            with open(os.path.join(args.output_dir, "val_loss.txt"), "a") as f:
                f.write("\nrecover_step = {}, use_num_samples = {}, answer_provided_by = {}, task = {}\n".format(recover_step, args.use_num_samples, args.answer_provided_by, args.task_to_learn))
                f.write(str(np.mean([l for L in loss_dict for l in L])))

            
    else: # inference mode
        if "img" in args.answer_provided_by:
            print(args.use_img_meta)
            print(args.use_img_content)
            log_txt_content.append("use_img_content = {}".format(args.use_img_content))
            log_txt_content.append("use_img_meta = {}".format(args.use_img_meta))
            log_txt_content.append("\nimg Filter_max_choices: {}".format(args.img_filter_max_choices)) ## when txt is included, modify here!
            print("\nimg Filter_max_choices: {}".format(args.img_filter_max_choices))
        if "txt" in args.answer_provided_by:
            print(args.use_txt_fact)
            log_txt_content.append("use_txt_fact = {}".format(args.use_txt_fact))
            log_txt_content.append("\ntxt Filter_max_choices: {}".format(args.txt_filter_max_choices)) ## when txt is included, modify here!
            print("\ntxt Filter_max_choices: {}".format(args.txt_filter_max_choices))
        print("-------------------- Filter Inference mode ------------------------")
        log_txt_content.append("-------------------- Filter Inference mode ------------------------")
        
        log_txt_content.append("split = {}".format(args.split))
        log_txt_content.append("use_num_samples = {}".format(args.use_num_samples))
        
        th_list = [float(i) for i in args.filter_infr_th.split("|")]
        if not 0.7 in th_list: th_list.append(0.7)
        print("\nThresholds: ", str(th_list))
        log_txt_content.append("\nThresholds: {}".format(str(th_list)))
        model.eval()

        score_dict = dict([(th, {'pr':[], 're':[], 'f1':[]}) for th in th_list])
        #for th in th_list:
        dataloader_iters = [iter(l) for l in train_dataloaders]
        total_samples = sum([len(l.dataset) for l in train_dataloaders])
        print("\ntotal_samples = ", total_samples)
        iter_bar = tqdm(train_dataloader_order, desc='Iter (loss=X.XXX), loader_idx=X') 
        nbatches = sum(loader_lengths)
                
        Pred = []
        Choices = []
        Example_ids = []
        Filter_labels = []
        with torch.no_grad():
            for step, loader_idx in enumerate(iter_bar):
                batch = next(dataloader_iters[loader_idx])
                for param_tensor in model.state_dict():
                    if torch.isnan(model.state_dict()[param_tensor]).any().item():
                        print("\n nan exists in ", param_tensor)
                batch = [t.to(device) if not isinstance(t, list) else t for t in batch ]
                input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next, do_filter_task, filter_label, logit_mask, ori_choices, task_idx, img, vis_pe, context, cxt_modality_label, example_ids = batch
                if args.fp16:
                    img = img.half()
                    vis_pe = vis_pe.half()
                        
                conv_feats = img.data # Bx100x2048
                vis_pe = vis_pe.data

                # doesn't support scst training for not
                cur_batch_score, pred = model(vis_feats=conv_feats, vis_pe=vis_pe, input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, \
                        masked_lm_labels=masked_ids, do_filter_task=do_filter_task, filter_label=filter_label, logit_mask=logit_mask, context=context, \
                        cxt_modality_label=cxt_modality_label, next_sentence_label=is_next, masked_pos=masked_pos, masked_weights=masked_weights, \
                        task_idx=task_idx, drop_worst_ratio=0, filter_infr_th=th_list)
                assert len(cur_batch_score) == len(th_list)
                Pred.append(pred)
                Filter_labels.append(filter_label.detach().cpu())
                Choices.extend(ori_choices)
                Example_ids.extend(example_ids)
                if "filter" in args.task_to_learn:
                    for th in cur_batch_score:
                        score_dict[th]['pr'].append(cur_batch_score[th][0])
                        score_dict[th]['re'].append(cur_batch_score[th][1])
                        score_dict[th]['f1'].append(cur_batch_score[th][2])
                    iter_bar.set_description('Iter={} loader_idx={} '.format(step, loader_idx))
                else:
                    raise ValueError("Currently don't support qa task in inference mode")
            
            Pred = torch.cat(Pred, dim=0)
            Pred = Pred.numpy()
            Pred = [["{0:.4f}".format(s) for s in p] for p in Pred]
            Filter_labels = torch.cat(Filter_labels, dim=0)
            Filter_labels = Filter_labels.numpy()
            Filter_labels = [[int(i[0]) for i in b] for b in Filter_labels]
            for th in th_list:
                score_dict[th]['pr'] = np.sum(score_dict[th]['pr']) / float(total_samples)
                score_dict[th]['re'] = np.sum(score_dict[th]['re']) / float(total_samples)
                score_dict[th]['f1'] = np.sum(score_dict[th]['f1']) / float(total_samples)
                print("\nth = {}".format(th))
                print("pr.mean = ", score_dict[th]['pr'])
                print("re.mean = ", score_dict[th]['re'])
                print("f1.mean = ", score_dict[th]['f1'])
                log_txt_content.append("\nth = {}".format(th))
                log_txt_content.append("pr.mean = {}".format(score_dict[th]['pr']))
                log_txt_content.append("re.mean = {}".format(score_dict[th]['re']))
                log_txt_content.append("f1.mean = {}".format(score_dict[th]['f1']))
        output_pkl = {}
        for e, c, l, p in zip(Example_ids, Choices, Filter_labels, Pred):
            output_pkl[e] = {"choices": c, "labels": l, "pred_scores": p}
        pkl_filename = "{}_{}_step{}".format(str(args.split), args.use_num_samples, recover_step)
        if "img" in args.answer_provided_by:
            args.output_suffix = args.img_dataset_json_path.split('/')[-1].replace(".json", "") + args.output_suffix
            pkl_filename += "_{}_{}_{}_{}_".format("img", args.img_filter_max_choices, args.use_img_content, args.use_img_meta)
        if "txt" in args.answer_provided_by:
            args.output_suffix =  args.txt_dataset_json_path.split('/')[-1].replace(".json", "") + args.output_suffix
            pkl_filename += "_{}_{}_{}_".format("txt", args.txt_filter_max_choices, args.use_txt_fact)
        pkl_filename += args.output_suffix
        if args.use_x_distractors: pkl_filename += "_UNknown_modality"
        with open(os.path.join(args.output_dir, "{}.json".format(pkl_filename)), "w") as f:
            json.dump(output_pkl, f, indent=4)
        with open(os.path.join(args.output_dir, "{}.txt".format(pkl_filename)), "w") as f:
            f.write("\n")
            f.write(datetime.now(tz=timezone('US/Eastern')).strftime("%y-%m-%d %H:%M:%S") + '\n')
            f.write(pkl_filename)
            f.write("\n")
            f.write("\n".join(log_txt_content))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

