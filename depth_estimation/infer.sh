PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/test.py configs/dpt/dpt_deit3_s_stitch_l_nyu.py /data2/release_weights/nyu2/dpt_deit3_s_stitch_l_nyu/latest.pth --show-dir nfs/saves/visualization