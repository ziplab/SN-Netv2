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

TODO





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



## Gradio Demo

First, install gradio by 

```bash
pip install gradio
```

Next, run the gradio demo by 

```bash
python demo/video_demo_gradio.py
```



![gradio_demo](/data2/github/SN-Netv2/segmentation/demo/gradio_demo.png)

## Acknowledgement

This code is built upon [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).
