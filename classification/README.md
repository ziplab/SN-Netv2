## ImageNet Classification Code for SN-Netv2

## Installation
Prepare a Python environment as below

```bash
conda create -n snnetv2 python=3.9 -y
conda activate snnetv2

# install PyTorch and CUDA Toolkit
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# install additional packages
pip install timm==0.6.12
pip install fvcore
pip install ninja

# install apex, adopted from https://github.com/NVIDIA/apex
cd ../
git clone https://github.com/NVIDIA/apex && cd apex
git checkout 23.05 # note that the latest apex has some issues when using fusedlamb optimizer
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../classification
```

Prepare pretrained checkpoints from DeiT-3
```bash
wget -P ../pretrained_weights/ https://dl.fbaipublicfiles.com/deit/deit_3_small_224_21k.pth 
wget -P ../pretrained_weights/ https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth 
wget -P ../pretrained_weights/ https://dl.fbaipublicfiles.com/deit/deit_3_large_224_21k.pth 
```

## Train

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
      --master_port 12345 \
      --use_env main.py \
      --config config/deit_stitching_snnetv2_s_l.json --dist-eval --get_flops
```

Make sure you update the ImageNet path in `config/deit_stitching_snnetv2_s_l.json`. By default, we train DeiT-based SN-Netv2 with 50 epochs.


## Evaluation

```
python -m torch.distributed.launch --nproc_per_node=8 \
      --master_port 1234 \
      --use_env main.py \
      --config config/deit_stitching_snnetv2_s_l.json \
      --dist-eval --get_flops --eval --resume [path/to/snnet_deit.pth]
```

## Acknowledgement

This implementation is mainly based on DeiT. We thank the authors for their released code.

