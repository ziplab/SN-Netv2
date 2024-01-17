import gradio as gr
import os


def video_identity(video, stitch_id):
    return video, None

with gr.Blocks() as demo:
    with gr.Column():
        gr.HTML("""
            <h1 align="center" style=" display: flex; flex-direction: row; justify-content: center; font-size: 25pt; ">Stitched ViTs are Flexible Vision Backbones</h1>
            <h2 align="center" >This is the segmentation demo page of SN-Netv2, an flexible vision backbone that allows for 100+ runtime speed and performance trade-offs.</h3>
            """)
        with gr.Row():
            with gr.Column():
                video_input = gr.Video()
                stitch_slider = gr.Slider(minimum=0, maximum=134, step=1, label="All 100+ Stitches")
                submit_button = gr.Button()

            with gr.Column():
                video_output = gr.Video()
                stitch_plot = gr.Plot()

        submit_button.click(
            video_identity,
            inputs=[video_input, stitch_slider],
            outputs=[video_output, stitch_plot]
        )


if __name__ == "__main__":
    demo.launch()
