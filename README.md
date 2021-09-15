# VLP
This repo hosts the source code for the baseline models described in [WebQA: Multihop and Multimodal QA](https://arxiv.org/abs/2109.00590).
All models were initialized from the released [VLP checkpoints](https://github.com/LuoweiZhou/VLP#-misc).
We will release checkpoints fine-tuned on WebQA.


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
Our code is mainly based on [Zhou](https://arxiv.org/pdf/1909.11059.pdf) et al.'s [VLP](https://github.com/LuoweiZhou/VLP) repo. We thank the authors for their wonderful open-source efforts.

