"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append("/home/yingshac/CYS/WebQnA/VLP")
import logging
import glob
import math
import json
import argparse
from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import random
import copy

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForWebqa, BertForPreTrainingLossMask, BertForSeq2SeqDecoder
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from vlp.loader_utils import batch_list_to_batch_tensors
import vlp.webqa_loader as webqa_loader
from vlp.scst_utils import *
from misc.data_parallel import DataParallelImbalance




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
    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--output_dir",
                        default='tmp',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_file",
                        default="training.log",
                        type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path",
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
                        default=4,
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
                        help="Truncate_config: maximum length of segment B.")
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
    parser.add_argument('--dataset_json_path', type=str, default="/home/yingshac/CYS/WebQnA/WebQnA_data/dataset_J0501-Copy1.json")
    parser.add_argument('--gold_feature_folder', type=str, default="/data/yingshac/MMMHQA/imgFeatures_upd/gold")
    parser.add_argument('--distractor_feature_folder', type=str, default="/data/yingshac/MMMHQA/imgFeatures_upd/distractors")
    parser.add_argument('--img_metadata_path', type=int, default=-1, help="how many samples should be loaded into memory")
    parser.add_argument('--use_num_samples', type=int, default=-1, help="how many samples should be loaded into memory")
    
    # Others for VLP
    parser.add_argument("--src_file", default=['/mnt/dat/COCO/annotations/dataset_coco.json'],
                        type=str, nargs='+',
                        help="The input data file name.")
    parser.add_argument('--enable_visdom', action='store_true')
    parser.add_argument('--visdom_port', type=int, default=8888)
    # parser.add_argument('--resnet_model', type=str, default='imagenet_weights/resnet101.pth')
    parser.add_argument('--image_root', type=str, default='/mnt/dat/COCO/images')
    parser.add_argument('--dataset', default='coco', type=str,
                        help='coco | flickr30k | cc')
    parser.add_argument('--split', type=str, nargs='+', default=['train', 'restval'])

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

    parser.add_argument('--s2s_prob', default=1, type=float,
                        help="Percentage of examples that are bi-uni-directional LM (seq2seq).")
    parser.add_argument('--bi_prob', default=0, type=float,
                        help="Percentage of examples that are bidirectional LM.")
    parser.add_argument('--enable_butd', action='store_true',
                        help='set to take in region features')
    parser.add_argument('--region_bbox_file', default='coco_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5', type=str)
    parser.add_argument('--region_det_file_prefix', default='feat_cls_1000/coco_detection_vg_100dets_gvd_checkpoint_trainval', type=str)
    parser.add_argument('--tasks', default='img2txt',
                        help='img2txt | vqa2')
    parser.add_argument('--relax_projection',
                        action='store_true',
                        help="Use different projection layers for tasks.")
    parser.add_argument('--scst', action='store_true',
                        help='Self-critical sequence training')

    args = parser.parse_args()

    print('global_rank: {}, local rank: {}'.format(args.global_rank, args.local_rank))
    args.max_seq_length = args.max_len_b + args.max_len_a + 3 # +3 for 2x[SEP] and [CLS]
    args.dist_url = args.dist_url.replace('[PT_OUTPUT_DIR]', args.output_dir)
    assert args.len_vis_input == 100, "run main: only support 100 region features per image"
    # output config
    os.makedirs(args.output_dir, exist_ok=True)
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
        torch.distributed.init_process_group(backend='nccl', init_method = args.dist_url,
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

    # plotting loss, optional
    if args.enable_visdom:
        import visdom
        vis = visdom.Visdom(port=args.visdom_port, env=args.output_dir)
        vis_window={'iter': None, 'score':None}

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case,
        cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.global_rank))
    if args.max_position_embeddings:
        tokenizer.max_len = args.max_position_embeddings
    # doesn't support WhitespaceTokenizer

    processor = webqa_loader.Preprocess4webqa(args.max_pred, args.mask_prob, \
            list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, max_len=args.max_seq_length, \
            len_vis_input=args.len_vis_input, max_len_a=args.max_len_a, max_len_b=args.max_len_b, \
            max_len_img_cxt=args.max_len_img_cxt, new_segment_ids=args.new_segment_ids, \
            truncate_config={'trunc_seg': args.trunc_seg, 'always_truncate_tail': args.always_truncate_tail}, \
            local_rank=args.local_rank)
    
    train_dataset = webqa_loader.webqaDataset(dataset_json_path=args.dataset_json_path, img_metadata_path=args.img_metadata_path, split=args.split, \
            batch_size=args.train_batch_size, tokenizer=tokenizer, gold_feature_folder=args.gold_feature_folder, \
            distractor_feature_folder=args.distractor_feature_folder, use_num_samples=args.use_num_samples, \
            processor=processor, device=device)

    if args.world_size == 1:
            train_sampler = RandomSampler(train_dataset, replacement=False)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.train_batch_size, sampler=train_sampler, num_workers=args.num_workers,
        collate_fn=batch_list_to_batch_tensors, pin_memory=True)

    train_dataloaders = [train_dataloader]
    loader_lengths = [len(l) for l in train_dataloaders]
    train_dataloader_order = []
    for i in range(len(loader_lengths)):
        train_dataloader_order.extend([i] * loader_lengths[i])
    train_dataloader_order = random.shuffle(train_dataloader_order)
    # The actual number of params updates
    t_total = int(sum(loader_lengths) * args.num_train_epochs * 1. / args.gradient_accumulation_steps)

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    recover_step = _get_max_epoch_model(args.output_dir)
    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2
    relax_projection = 4 if args.relax_projection else 0
    task_idx_proj = 3 # harded to be 3 # if args.tasks == 'img2txt' else 0
    mask_word_id, eos_word_ids, pad_word_ids = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]", "[PAD]"]) # index in BERT vocab: 103, 102, 0

    # Recover model
    if (recover_step is None) and (args.model_recover_path is None):
        print("nothing to recover")
        # if _state_dict == {}, the parameters are randomly initialized
        # if _state_dict == None, the parameters are initialized with bert-init
        assert args.scst == False, 'must init from maximum likelihood training'
        _state_dict = {} if args.from_scratch else None
        model = BertForWebqa.from_pretrained(
            args.bert_model, state_dict=_state_dict, num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, relax_projection=relax_projection,
            config_path=args.config_path, task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,
            fp32_embedding=args.fp32_embedding, cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.global_rank),
            drop_prob=args.drop_prob, max_len_a=args.max_len_a)
        global_step = 0
    else:
        if recover_step:
            logger.info("***** Recover model: %d *****", recover_step)
            model_recover = torch.load(os.path.join(
                args.output_dir, "model.{0}.bin".format(recover_step)))
            # recover_step == number of epochs
            global_step = math.floor(
                recover_step * t_total * 1. / args.num_train_epochs)
        elif args.model_recover_path:
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
                drop_prob=args.drop_prob, max_len_a=args.max_len_a)
        else:
            model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model,
                max_position_embeddings=args.max_position_embeddings, config_path=args.config_path,
                state_dict=model_recover, num_labels=cls_num_labels, type_vocab_size=type_vocab_size,
                task_idx=task_idx_proj, mask_word_id=mask_word_id, search_beam_size=1,
                eos_id=eos_word_ids, enable_butd=args.enable_butd,
                len_vis_input=args.len_vis_input)

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
        # model = torch.nn.DataParallel(model)
        pass
        #model = DataParallelImbalance(model)

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
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             schedule=args.sche_mode,
                             t_total=t_total)

    if recover_step:
        logger.info("***** Recover optimizer: %d *****", recover_step)
        optim_recover = torch.load(os.path.join(
            args.output_dir, "optim.{0}.bin".format(recover_step)))
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
            print(i_epoch)
            dataloader_iters = [iter(l) for l in train_dataloaders]
            if args.local_rank >= 0:
                train_sampler.set_epoch(i_epoch-1)
            iter_bar = tqdm(train_dataloader_order, desc='Iter (loss=X.XXX), loader_idx=X') 
            nbatches = sum(loader_lengths)
            train_loss = []
            filter_loss = []
            scst_reward = []
            for step, loader_idx in enumerate(iter_bar):
                batch = next(dataloader_iters[loader_idx])
                batch = [t.to(device) for t in batch]
                input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next, do_filter_task, filter_label, task_idx, img, vis_pe, context_is_img = batch
                if args.fp16:
                    img = img.half()
                    vis_pe = vis_pe.half()
                
                conv_feats = img.data # Bx100x2048
                vis_pe = vis_pe.data

                # doesn't support scst training for not
                loss_tuple = model(conv_feats, vis_pe, input_ids, segment_ids, input_mask, masked_ids, \
                        do_filter_task=do_filter_task, filter_label=filter_label, context_is_img=context_is_img, \
                        next_sentence_label=is_next, masked_pos=masked_pos, masked_weights=masked_weights, task_idx=task_idx,\
                        drop_worst_ratio=args.max_drop_worst_ratio if i_epoch > args.drop_after else 0)
                mean_reward = loss_tuple[0].new(1).fill_(0)

                # disable pretext_loss_deprecated for now
                masked_lm_loss, cls_loss = loss_tuple
                if n_gpu > 1:    # mean() to average on multi-gpu. For dist, this is done through gradient addition.
                    masked_lm_loss = masked_lm_loss.mean()
                    cls_loss = cls_loss.mean()
                loss = masked_lm_loss + cls_loss

                # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
                iter_bar.set_description('Iter (loss={:.3f}) loader_idx='.format(loss.item(), loader_idx))
                train_loss.append(loss.item())
                filter_loss.append(cls_loss.item())
                scst_reward.append(mean_reward.item())
                if step%100 == 0:
                    logger.info("Epoch {}, Iter {}, Loss {:.2f}, Filter {:.2f}, Mean R {:.3f}\n".format(i_epoch, step, np.mean(train_loss), np.mean(filter_loss), np.mean(scst_reward)))

                if args.enable_visdom:
                    if vis_window['iter'] is None:
                        vis_window['iter'] = vis.line(
                            X=np.tile(np.arange((i_epoch-1)*nbatches+step,
                                      (i_epoch-1)*nbatches+step+1), (1,1)).T,
                            Y=np.column_stack((np.asarray([np.mean(train_loss)]),)),
                            opts=dict(title='Training Loss',
                                      xlabel='Training Iteration',
                                      ylabel='Loss',
                                      legend=['total'])
                        )
                    else:
                        vis.line(
                            X=np.tile(np.arange((i_epoch-1)*nbatches+step,
                                      (i_epoch-1)*nbatches+step+1), (1,1)).T,
                            Y=np.column_stack((np.asarray([np.mean(train_loss)]),)),
                            opts=dict(title='Training Loss',
                                      xlabel='Training Iteration',
                                      ylabel='Loss',
                                      legend=['total']),
                            win=vis_window['iter'],
                            update='append'
                        )

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

            # Save a trained model
            logger.info(
                "** ** * Saving fine-tuned model and optimizer ** ** * ")
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                args.output_dir, "model.{0}.bin".format(i_epoch))
            output_optim_file = os.path.join(
                args.output_dir, "optim.{0}.bin".format(i_epoch))
            if args.global_rank in (-1, 0): # save model if the first device or no dist
                torch.save(copy.deepcopy(model_to_save).cpu().state_dict(), output_model_file)
                # torch.save(optimizer.state_dict(), output_optim_file) # disable for now, need to sanitize state and ship everthing back to cpu

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

            if args.world_size > 1:
                torch.distributed.barrier()


if __name__ == "__main__":
    main()
