# Depth Estimation Code for SN-Netv2

## Installation

Follow [here](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/main/docs/get_started.md#installation) to prepare the environment.



## Pretrained Weights

TODO


## Training

1. First, analysing FLOPs for all stitches.

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/analysis_tools/get_flops_snnet.py [path/to/config]
```

For example, if you want to train `configs/snnet/snnetv2_dpt_deit3_s_l_nyu.py`, then run

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/get_flops.py configs/snnet/snnetv2_dpt_deit3_s_l_nyu.py
```

The above command will generate a json file at `./model_flops`.



2. Train your model

```bash
bash tools/dist_train.sh [path/to/config] 8 --no-validate
```



## Evaluation

```bash
bash tools/dist_test.sh [path/to/config] [path/to/checkpoint] [num of GPUs] --eval mIoU --out [json_output_file]
```

For example,

```bash
bash tools/dist_test.sh configs/snnet/snnetv2_dpt_deit3_s_l_nyu.py \
   ./ckpt/snnetv2_dpt_deit3_s_l_nyu/latest.pth 8 --eval mIoU --out ./snnetv2_dpt_deit3_s_l_nyu.json
```


## Acknowledgement

This code is built upon [Monocular-Depth-Estimation-Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox).
