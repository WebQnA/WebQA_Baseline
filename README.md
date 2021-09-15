This repo hosts the source code for the baseline models described in [WebQA: Multihop and Multimodal QA](https://arxiv.org/abs/2109.00590).

All models were initialized from the released [VLP checkpoints](https://github.com/LuoweiZhou/VLP#-misc).

We release checkpoints fine-tuned on WebQA [here](https://tiger.lti.cs.cmu.edu/yingshac/WebQA_data_first_release/WebQA_baseline_ckpts.7z).


## Environment
```
cd VLP
conda env create -f misc/vlp.yml --prefix /home/<username>/miniconda3/envs/vlp
conda activate vlp
```

Clone repo https://github.com/NVIDIA/apex
```
cd apex
git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8
python setup.py install --cuda_ext --cpp_ext
```

```
pip install datasets==1.7.0
pip install opencv-python==3.4.2.17 
```

## Visual Features Extraction

- X101fpn

The detectron2-based feature extraction code is available under this [repo](https://github.com/zdxdsw/WebQA_x101fpn/blob/main/featureExtraction.py). Part of the code is based on [LuoweiZhou/detectron-vlp](https://github.com/LuoweiZhou/detectron-vlp) and [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)

[Download checkpoint](https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212014&authkey=AAHgqN3Y-LXcBvU)

- VinVL

Please refer to [pzzhang/VinVL](https://github.com/pzzhang/VinVL) and [microsoft/scene_graph_benchmark](https://github.com/microsoft/scene_graph_benchmark)

[Download checkpoint](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth)


## Commands

```
cd vlp
```

Retrieval training
```
python run_webqa.py --new_segment_ids --train_batch_size 128 --split train --answer_provided_by 'img|txt' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 128 --save_loss_curve --output_dir light_output/filter_debug --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_debug --use_x_distractors --do_train --num_train_epochs 6
```

Retrieval inference
```
python run_webqa.py --new_segment_ids --train_batch_size 16 --split val --answer_provided_by 'img|txt' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --output_dir light_output/filter_debug --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_debug --recover_step 4 --use_x_distractors
```

QA training
```
python run_webqa.py --new_segment_ids --do_train --train_batch_size 128 --split train --answer_provided_by 'img|txt' --task_to_learn 'qa' --num_workers 4 --max_pred 50 --mask_prob 0.5 --learning_rate 1e-4 --gradient_accumulation_steps 64 --save_loss_curve --num_train_epochs 16 --output_dir light_output/qa_debug --ckpts_dir /data/yingshac/MMMHQA/ckpts/qa_debug
```

QA decode
```
python decode_webqa.py --new_segment_ids --batch_size 32 --answer_provided_by "img|txt" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/qa_debug --ckpts_dir /data/yingshac/MMMHQA/ckpts/qa_debug --no_eval --recover_step 11
```

With VinVL features, run `run_webqa_vinvl.py` or `decode_webqa_vinvl.py` instead.

## Reference
Please acknowledge the following paper if you use the code:
```
@inproceedings{WebQA21,
 title ={{WebQA: Multihop and Multimodal QA}},
 author={Yinghsan Chang and Mridu Narang and
         Hisami Suzuki and Guihong Cao and
         Jianfeng Gao and Yonatan Bisk},
 journal = {ArXiv},
 year = {2021},
 url  = {https://arxiv.org/abs/2109.00590}
}
```


## Related Projects/Codebase

- VinVL Visual Representations: https://github.com/pzzhang/VinVL
- Scene Graph Benchmark: https://github.com/microsoft/scene_graph_benchmark
- Detectron2: https://github.com/facebookresearch/detectron2

## Acknowledgement
Our code is mainly based on [Zhou](https://arxiv.org/pdf/1909.11059.pdf) et al.'s [VLP](https://github.com/LuoweiZhou/VLP) repo. We thank the authors for their valuable work.

