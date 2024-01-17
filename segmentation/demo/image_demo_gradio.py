# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import cv2
from mmengine.model.utils import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model
from mmseg.apis.inference import show_result_pyplot
import torch
import time
import gradio as gr

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default='configs/setr/setr_naive_512x512_160k_b16_ade20k_deit_3_s_l_224_snnetv2.py', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file', default='/data2/release_weights/ade20k/setr_naive_512x512_160k_b16_ade20k_snnetv2_deit3_s_l_lora_16_iter_160000.pth')
    # parser.add_argument('--video', help='Video file or webcam id')

    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--show', action='store_true', help='Whether to show draw result')
    parser.add_argument(
        '--show-wait-time', default=1, type=int, help='Wait time after imshow')
    parser.add_argument(
        '--output-file', default=None, type=str, help='Output video file path')
    parser.add_argument(
        '--output-fourcc',
        default='MJPG',
        type=str,
        help='Fourcc of the output video')
    parser.add_argument(
        '--output-fps', default=30, type=int, help='FPS of the output video')
    parser.add_argument(
        '--output-height',
        default=-1,
        type=int,
        help='Frame height of the output video')
    parser.add_argument(
        '--output-width',
        default=-1,
        type=int,
        help='Frame width of the output video')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    def segment_video(image):
        # test a single image
        result = inference_model(model, image)

        # blend raw image and prediction
        draw_img = show_result_pyplot(model, image, result,
                                      show=False,
                                      with_labels=False,
                                      )
        return draw_img



    demo = gr.Interface(segment_video,
                        inputs=gr.Image(sources='webcam', streaming=True),
                        outputs=gr.Image(),
                        live=True
                        )

    demo.launch()


if __name__ == '__main__':
    main()
