This repo hosts the source code for the baseline models described in [WebQA: Multihop and Multimodal QA](https://arxiv.org/abs/2109.00590).

All models were initialized from the released [VLP checkpoints](https://github.com/LuoweiZhou/VLP#-misc).

We release checkpoints fine-tuned on WebQA [here](https://tiger.lti.cs.cmu.edu/yingshac/WebQA_data_first_release/WebQA_baseline_ckpts.7z).


## News

**Update (8 Apr, 2022)**:

We are releasing all image features (stored in pickle files) pre-extracted by x101fpn! Given the sheer size (500GB) and the bandwidth limits of google drive, we have setup [this request form](https://forms.gle/5oR7PxXYJH1qF7ZT8) to grant access to individual users instead of making the files publicly accessible. Upon approval, you will get a link that expires in **90 days**. To request for extension, please submit the form again or contact the authors. Thanks!

**`/dev`** contains all pickles for reproducing results *only* on the **dev** set. (This small set of pickles can also be downloaded from [here](https://tiger.lti.cs.cmu.edu/yingshac/dev_7z.tar))
**`/test`** contains extra pickles for the **test** set which are NOT included in **/dev**.
**`/train`** contains extra pickles for the **train** set which are NOT included in the two folders above.

This file structure is to facilitate the most common demands for replicating results on the dev set and less common demands for replicating training. However, if you want to redo training with the released image features, you need to download all three directories (/dev, /test, /train).

The pickle files are stored by chunks of size 1000. The file structure should be the following:

```
<feature_folder>
│── dev
│   ├── 0
│   │   ├── 40000000.pkl
│   │   ├── ...
│   │   └── 40000999.pkl
│   │── 1
│   │   ├── 40001000.pkl
│   │   ├── ...
│   │   └── 40001999.pkl
│   ├── ...
│   └──64 ...
│
├── test
│   ├── 64 ...
│   ├── 65 ...
│   ├── ...
│   └── 140 ...
│
└── train
    ├── 140 ...
    ├── 141 ...
    ├── ...
    └── 389 ...
```

Codes (`run_webqa.py` and `webqa_loader.py`) have been updated to correctly load features from this file structure.

`<feature_folder>` is the directory path you should provide to the `--feature_folder` argument.


Also note that the pickle files are named by a new set of image ids (starting with 40,000,000). This is only for the ease of sorting pickle files by the order of dev/test/train. Image ids in the released `dataset.json` start with 30,000,000. Please use this [map](https://github.com/WebQnA/WebQA_Baseline/blob/main/misc/image_id_map_0328.pkl) for image id conversion.

**Update (6 Oct, 2021)**:

In our baseline code, we separate the data loading of image- and text-based queries for the sake of performance breakdown. Thus, the two arguments, `txt_dataset_json_path` and `img_dataset_json_path`, correspond to the two folds,  according to the `Qcate` field. 

**Update (29 Sep, 2021)**:

*(Forget about this if you are following the feature file structures in the Mar28 release)* We clarify here what are arguments `--gold_feature_folder`, `--distractor_feature_folder`, and `--x_distractor_feature_folder`. Basically, during implementation we divide the images into 3 buckets: positive images for image-based queries (`gold`), negative images for image-based queries (`distractors`) and negative images for text-based queries (`x_distractors`), where the 'x' stands for 'cross-modality'. Image- and text-based queries can be disinguished via the "Qcate" field in the dataset file. Text-based queries all have `Qcate == 'text'`, while the rest are image-based ones.

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

[Download checkpoint](https://tiger.lti.cs.cmu.edu/yingshac/WebQA_data_first_release/e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp-427.pkl)

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
python run_webqa.py --new_segment_ids --train_batch_size 16 --split val --answer_provided_by 'img|txt' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --output_dir light_output/filter_debug --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_debug --recover_step 3 --use_x_distractors
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

## TODO List

- Release x101fpn image features for the replicating results on **dev** set :white_check_mark:
- Release additional x101fpn image features for replicating results on **test** and **train** set :white_check_mark:
- Provide detailed documentations for VLP
- Release code for BM25 full-scale retrieval (over 544k text sources and 390k image sources across the entire dataset) :white_check_mark:
- Release code for CLIP (zero-shot) full-scale retrieval

