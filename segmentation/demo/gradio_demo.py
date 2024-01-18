# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import cv2
from mmengine.model.utils import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model
from mmseg.apis.inference import show_result_pyplot
import torch
import time
import gradio as gr
import plotly.express as px
import json

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default='configs/snnet/setr_naive_512x512_160k_b16_ade20k_deit_3_s_l_224_snnetv2.py', help='Config file')
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

    from mmseg.models.backbones.snnet import get_stitch_configs_bidirection
    stitch_configs_info, _, _, anchor_ids, sl_ids, ls_ids, lsl_ids, sls_ids = get_stitch_configs_bidirection([12, 24])

    stitch_configs_info = {i: cfg for i, cfg in enumerate(stitch_configs_info)}


    with open('./model_flops/snnet_flops_setr_naive_512x512_160k_b16_ade20k_deit_3_s_l_224_snnetv2.json', 'r') as f:
        flops_params = json.load(f)

    with open('./results/eval_single_scale_20230507_235400.json', 'r') as f:
        results = json.load(f)

    config_ids = list(results.keys())
    flops_res = {}
    eval_res = {}
    total_data = {}
    for i, cfg_id in enumerate(config_ids):
        flops = flops_params[cfg_id]
        miou_res = results[cfg_id]['metric']['mIoU'] * 100
        eval_res[int(cfg_id)] = miou_res
        flops_res[int(cfg_id)] = flops / 1e9
        total_data[int(cfg_id)] = [flops // 1e9, miou_res]


    def visualize_stitch_pos(stitch_id):
        if stitch_id == 13:
            # 13 is equivalent to 0
            stitch_id = 0

        names = [f'ID {key}' for key in flops_res.keys()]

        fig = px.scatter(x=flops_res.values(), y=eval_res.values(), hover_name=names)
        fig.update_layout(
            title=f"SN-Netv2 - Stitch ID - {stitch_id}",
            title_x=0.5,
            xaxis_title="GFLOPs",
            yaxis_title="mIoU",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            ),
            legend=dict(
                yanchor="bottom",
                y=0.99,
                xanchor="left",
                x=0.01),
        )
        # continent, DarkSlateGrey
        fig.update_traces(marker=dict(size=10,
                                      line=dict(width=2)),
                          selector=dict(mode='markers'))

        fig.add_scatter(x=[flops_res[stitch_id]], y=[eval_res[stitch_id]], mode='markers', marker=dict(size=15), name='Current Stitch')
        return fig


    def segment_video(video, stitch_id):

        if stitch_id == 13:
            # 13 is equivalent to 0
            stitch_id = 0

        model.backbone.reset_stitch_id(stitch_id)
        output_video_path = './temp_video.avi'
        cap = cv2.VideoCapture(video)
        assert (cap.isOpened())
        input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        input_fps = cap.get(cv2.CAP_PROP_FPS)


        fourcc = cv2.VideoWriter_fourcc(*args.output_fourcc)
        output_fps = args.output_fps if args.output_fps > 0 else input_fps
        output_height = args.output_height if args.output_height > 0 else int(
            input_height)
        output_width = args.output_width if args.output_width > 0 else int(
            input_width)
        writer = cv2.VideoWriter(output_video_path, fourcc, output_fps,
                                 (output_width, output_height), True)

        try:
            while True:
                start_time = time.time()
                flag, frame = cap.read()
                if not flag:
                    break

                # test a single image
                result = inference_model(model, frame)

                # blend raw image and prediction
                draw_img = show_result_pyplot(model, frame, result,
                                              show=False,
                                              with_labels=False,
                                              )

                if draw_img.shape[0] != output_height or draw_img.shape[
                    1] != output_width:
                    draw_img = cv2.resize(draw_img,
                                          (output_width, output_height))
                writer.write(draw_img)
        finally:
            if writer:
                writer.release()
            cap.release()

        fig = visualize_stitch_pos(stitch_id)
        
        return output_video_path, fig

    def segment_image(image, stitch_id):
        if stitch_id == 13:
            # 13 is equivalent to 0
            stitch_id = 0

        model.backbone.reset_stitch_id(stitch_id)
        result = inference_model(model, image)
        draw_img = show_result_pyplot(model, image, result,
                                      show=False,
                                      with_labels=True,
                                      )
        fig = visualize_stitch_pos(stitch_id)
        return draw_img, fig



    with gr.Blocks() as image_demo:
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label='Input Image')
                stitch_slider = gr.Slider(minimum=0, maximum=134, step=1, label="Stitch ID")
                with gr.Row():
                    clear_button = gr.ClearButton()
                    submit_button = gr.Button()

            with gr.Column():
                image_output = gr.Image(label='Segmentation Results')
                stitch_plot = gr.Plot(label='Stitch Position')

            submit_button.click(
                fn=segment_image,
                inputs=[image_input, stitch_slider],
                outputs=[image_output, stitch_plot],
            )

            stitch_slider.change(
                fn=visualize_stitch_pos,
                inputs=[stitch_slider],
                outputs=[stitch_plot],
                show_progress=False
            )

            clear_button.click(
                lambda: [None, 0, None, None],
                outputs=[image_input, stitch_slider, image_output, stitch_plot],
            )

        gr.Examples(
            [
                ['./demo/demo_1.jpg', 0],
                ['./demo/demo_2.jpg', 1],
                ['./demo/demo_3.jpg', 93],
                ['./demo/demo_4.jpg', 3],
            ],
            inputs=[
                image_input,
                stitch_slider
            ],
            outputs=[
                image_input,
                stitch_plot
            ],
        )


    with gr.Blocks() as video_demo:
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label='Input Video')
                stitch_slider = gr.Slider(minimum=0, maximum=134, step=1, label="Stitch ID")
                with gr.Row():
                    clear_button = gr.ClearButton()
                    submit_button = gr.Button()

            with gr.Column():
                video_output = gr.Video(label='Segmentation Results')
                stitch_plot = gr.Plot(label='Stitch Position')

        submit_button.click(
            fn=segment_video,
            inputs=[video_input, stitch_slider],
            outputs=[video_output, stitch_plot],
        )

        stitch_slider.change(
            fn=visualize_stitch_pos,
            inputs=[stitch_slider],
            outputs=[stitch_plot],
            show_progress=False
        )

        clear_button.click(
            lambda: [None, 0, None, None],
            outputs=[video_input, stitch_slider, video_output, stitch_plot],
        )

        gr.Examples(
            [
                ['./demo/video_demo_input.mp4', 0],
            ],
            inputs=[
                video_input,
                stitch_slider
            ],
            outputs=[
                video_output,
                stitch_plot
            ],
        )



    with gr.Blocks() as demo:
        with gr.Column():
            gr.HTML("""
                <h1 align="center" style=" display: flex; flex-direction: row; justify-content: center; font-size: 25pt; ">Stitched ViTs are Flexible Vision Backbones</h1>
                <div align="center"> <img align="center" src='file/demo/gradio_banner.png' width="70%"> </div>
                <h3 align="center" >This is the segmentation demo page of SN-Netv2, an flexible vision backbone that allows for 100+ runtime speed and performance trade-offs.</h3>
                <h3 align="center" >You can also run this gradio demo on your local GPUs at https://github.com/ziplab/SN-Netv2</h3>
                """)
        tabbed_page = gr.TabbedInterface([image_demo, video_demo], ['Image', 'Video'])


    demo.launch(allowed_paths=['./'])


if __name__ == '__main__':
    main()
