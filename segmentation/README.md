## Installation

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

pip install regex
cd ./segmentation/
pip install -v -e .

sudo apt-get install ffmpeg
```





# Pretrained Weights

| Dataset        | Small ViT | Large ViT | Train Iter | Weights                                                      |
| -------------- | --------- | --------- | ---------- | ------------------------------------------------------------ |
| ADE20K         | DeiT3-S   | DeiT3-L   | 160K       | [huggingface](https://huggingface.co/ziplab/snnetv2_deit3_s_l_224_ade20k_setr_naive/blob/main/setr_naive_512x512_160k_b16_ade20k_snnetv2_deit3_s_l_lora_16_iter_160000.pth) |
| ADE20K         | DeiT3-B   | DeiT3-L   | 160K       | [huggingface](https://huggingface.co/ziplab/snnetv2_deit3_b_l_224_ade20k_setr_naive/blob/main/setr_naive_512x512_160k_b16_ade20k_snnetv2_deit3_b_l_lora_4_iter_160000.pth) |
| COCO-Stuff-10K | DeiT3-S   | DeiT3-L   | 80K        | [huggingface](https://huggingface.co/ziplab/snnetv2_deit3_s_l_224_coco_stuff_10k_setr_naive/blob/main/setr_naive_512x512_80k_b16_coco_stuff10k_deit_3_s_l_224_snnetv2_lora_r_16_iter_80000.pth) |
| COCO-Stuff-10K | DeiT3-B   | DeiT3-L   | 80K        | [huggingface](https://huggingface.co/ziplab/snnetv2_deit3_b_l_224_coco_stuff_10k_setr_naive/blob/main/setr_naive_512x512_80k_b16_coco_stuff10k_deit_3_b_l_224_snnetv2_lora_r_4_iter_80000.pth) |



## Training

1. First, analysing FLOPs for all stitches.

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/analysis_tools/get_flops_snnet.py [path/to/config]
```

For example, if you want to train `configs/snnet/setr_naive_512x512_80k_b16_coco_stuff10k_deit_3_s_l_224_snnetv2.py`, then run

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/analysis_tools/get_flops_snnet.py configs/snnet/setr_naive_512x512_80k_b16_coco_stuff10k_deit_3_s_l_224_snnetv2.py
```

The above command will generate a json file at `./model_flops`.



2. Train your model

```bash
bash tools/dist_train.sh configs/snnet/setr_naive_512x512_80k_b16_coco_stuff10k_deit_3_s_l_224_snnetv2.py 8 --no-validate
```



## Evaluation

```bash
bash tools/dist_test.sh [path/to/config] [path/to/checkpoint] [num of GPUs]
```

For example,

```bash
bash tools/dist_test.sh configs/setr/setr_naive_512x512_160k_b16_ade20k_deit_3_s_l_224_snnetv2.py setr_naive_512x512_160k_b16_ade20k_snnetv2_deit3_s_l_lora_16_iter_160000.pth 1

```

The above command will evaluate all stitches and generate a json file at `./results`



## Video Demo

You can run a video demo based on any stitches by

```bash
python demo/video_demo.py 0 [path/to/config] [path/to/checkpoint] --stitch-id 0 --show 
```

> 0 means using webcam, you can also pass a path to a video. see official demo from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/demo/video_demo.py).



## Acknowledgement

This code is built upon [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).
