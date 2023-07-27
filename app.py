
import os
import json
import torch
import random

import gradio as gr
from glob import glob
from omegaconf import OmegaConf
from datetime import datetime
from safetensors import safe_open

from diffusers import AutoencoderKL
from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from animatediff.utils.convert_lora_with_backup import load_loras


sample_idx     = 0
max_LoRAs      = 5
scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
}

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

class ProjectConfigs:
    def __init__(self):
        self.date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.prompt = ""
        self.n_prompt = ""
        self.sampler = ""
        self.num_inference_steps = 20
        self.guidance_scale = 7.5
        self.width = 512
        self.height = 512
        self.video_length = 20
        self.seed = -1
        self.temporal_context = 20
        self.overlap = 20
        self.strides = 0
        self.fp16 = True
        self.loras = [{'path' : 'none', 'alpha': 0.8}]*5

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def save_configs(self):
        with open(os.path.join(self.savedir, f"temp{self.date_created}.json"), "a") as f:
            f.write(self.toJSON())
            f.write("\n\n")

class AnimateController:
    def __init__(self):
        
        # config dirs
        self.basedir                = os.getcwd()
        self.stable_diffusion_dir   = os.path.join(self.basedir, "models", "StableDiffusion")
        self.motion_module_dir      = os.path.join(self.basedir, "models", "Motion_Module")
        self.checkpoints_dir = os.path.join(self.basedir, "models", "checkpoints")
        self.init_images_dir        = os.path.join(self.basedir, "init_images")
        self.savedir                = os.path.join(self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample         = os.path.join(self.savedir, "sample")
        os.makedirs(self.savedir, exist_ok=True)

        self.loras_dir = os.path.join(self.basedir, "models", "loras")
        self.lora_list = []
        self.project = ProjectConfigs()

        self.stable_diffusion_list   = []
        self.motion_module_list      = []
        self.checkpoints_list = []
        self.init_image_list = []
        
        self.refresh_stable_diffusion()
        self.refresh_motion_module()
        self.refresh_personalized_model()
        
        # config models
        self.tokenizer             = None
        self.text_encoder          = None
        self.vae                   = None
        self.unet                  = None
        self.pipeline              = None
        self.lora_model_state_dict = {}
        
        self.inference_config      = OmegaConf.load("configs/inference/inference.yaml")

    def refresh_stable_diffusion(self):
        self.stable_diffusion_list = glob(os.path.join(self.stable_diffusion_dir, "*/"))

    def refresh_init_images(self):
        self.init_image_list = glob(os.path.join(self.init_images_dir, "*"))

    def refresh_motion_module(self):
        motion_module_list = glob(os.path.join(self.motion_module_dir, "*.ckpt"))
        self.motion_module_list = [os.path.basename(p) for p in motion_module_list]

    def refresh_personalized_model(self):
        personalized_model_list = glob(os.path.join(self.checkpoints_dir, "*.safetensors"))
        self.checkpoints_list = [os.path.basename(p) for p in personalized_model_list]

    def refresh_lora_models(self):
        lora_list = glob(os.path.join(self.loras_dir, "*.safetensors"))
        self.lora_list = [os.path.basename(p) for p in lora_list]

    def update_stable_diffusion(self, stable_diffusion_dropdown):
        self.tokenizer = CLIPTokenizer.from_pretrained(stable_diffusion_dropdown, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(stable_diffusion_dropdown, subfolder="text_encoder").cuda()
        self.vae = AutoencoderKL.from_pretrained(stable_diffusion_dropdown, subfolder="vae").cuda()
        self.unet = UNet3DConditionModel.from_pretrained_2d(stable_diffusion_dropdown, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs)).cuda()
        return gr.Dropdown.update()

    def update_motion_module(self, motion_module_dropdown):
        if self.unet is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.Dropdown.update(value=None)
        else:
            motion_module_dropdown = os.path.join(self.motion_module_dir, motion_module_dropdown)
            motion_module_state_dict = torch.load(motion_module_dropdown, map_location="cpu")
            missing, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0
            return gr.Dropdown.update()

    def update_base_model(self, base_model_dropdown):
        if self.unet is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.Dropdown.update(value=None)
        else:
            base_model_dropdown = os.path.join(self.checkpoints_dir, base_model_dropdown)
            base_model_state_dict = {}
            with safe_open(base_model_dropdown, framework="pt", device="cpu") as f:
                for key in f.keys():
                    base_model_state_dict[key] = f.get_tensor(key)
                    
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_model_state_dict, self.vae.config)
            self.vae.load_state_dict(converted_vae_checkpoint)

            converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_model_state_dict, self.unet.config)
            self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

            self.text_encoder = convert_ldm_clip_checkpoint(base_model_state_dict)
            return gr.Dropdown.update()

    # Load loras
    def load_lora(self, pipeline):
        pipeline = load_loras(pipeline, self.project.loras)
        return pipeline


    def animate(
        self,
        stable_diffusion_dropdown,
        motion_module_dropdown,
        base_model_dropdown,
        lora_alpha_slider,
        init_image,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        width_slider, 
        length_slider, 
        height_slider, 
        cfg_scale_slider, 
        seed_textbox,
        context_length,
        context_stride,
        context_overlap,
        fp16
    ):    
        if self.unet is None:
            raise gr.Error(f"Please select a pretrained model path.")
        if motion_module_dropdown == "": 
            raise gr.Error(f"Please select a motion module.")
        if base_model_dropdown == "":
            raise gr.Error(f"Please select a base DreamBooth model.")

        if is_xformers_available(): self.unet.enable_xformers_memory_efficient_attention()


        pipeline = AnimationPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            scheduler=scheduler_dict[sampler_dropdown](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        ).to("cuda")
        
        # if self.lora_model_state_dict != {}:
        #     pipeline = convert_lora(pipeline, self.lora_model_state_dict, alpha=lora_alpha_slider)

        pipeline = self.load_lora(pipeline)

        pipeline.to("cuda")

        if seed_textbox != -1 and seed_textbox != "": torch.manual_seed(int(seed_textbox))
        else: torch.seed()
        seed = torch.initial_seed()
        
        # Handle none init image
        if init_image == "none": init_image = None

        sample = pipeline(
            prompt              = prompt_textbox,
            init_image          = init_image,
            negative_prompt     = negative_prompt_textbox,
            num_inference_steps = sample_step_slider,
            guidance_scale      = cfg_scale_slider,
            width               = width_slider,
            height              = height_slider,
            video_length        = length_slider,
            temporal_context    = context_length,
            strides             = context_stride + 1,
            overlap             = context_overlap,
            fp16                = fp16,
        ).videos

        save_sample_path = os.path.join(self.savedir_sample, f"{sample_idx}.mp4")
        save_videos_grid(sample, save_sample_path)
    
        sample_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "sampler": sampler_dropdown,
            "num_inference_steps": sample_step_slider,
            "guidance_scale": cfg_scale_slider,
            "width": width_slider,
            "height": height_slider,
            "video_length": length_slider,
            "seed": seed,
            "temporal_context": context_length,
            "strides": context_stride,
            "overlap": context_overlap,
            "fp16": fp16,
        }
        json_str = json.dumps(sample_config, indent=4)
        with open(os.path.join(self.savedir, "logs.json"), "a") as f:
            f.write(json_str)
            f.write("\n\n")
            
        return save_sample_path
        

controller = AnimateController()


def ui():
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725)
            Yuwei Guo, Ceyuan Yang*, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, Bo Dai (*Corresponding Author)<br>
            [Arxiv Report](https://arxiv.org/abs/2307.04725) | [Project Page](https://animatediff.github.io/) | [Github](https://github.com/guoyww/animatediff/)
            """
        )
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 1. Model checkpoints (select pretrained model path first).
                """
            )
            with gr.Row():
                stable_diffusion_dropdown = gr.Dropdown(
                    label="Pretrained Model Path",
                    choices=controller.stable_diffusion_list,
                    interactive=True,
                    value=controller.stable_diffusion_list[0]
                )
                stable_diffusion_dropdown.change(fn=controller.update_stable_diffusion, inputs=[stable_diffusion_dropdown], outputs=[stable_diffusion_dropdown])
                
                stable_diffusion_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def update_stable_diffusion():
                    controller.refresh_stable_diffusion()
                    return gr.Dropdown.update(choices=controller.stable_diffusion_list)
                stable_diffusion_refresh_button.click(fn=update_stable_diffusion, inputs=[], outputs=[stable_diffusion_dropdown])

            with gr.Row():
                motion_module_dropdown = gr.Dropdown(
                    label="Select motion module",
                    choices=controller.motion_module_list,
                    interactive=True,
                    value=controller.motion_module_list[0]
                )
                motion_module_dropdown.change(fn=controller.update_motion_module, inputs=[motion_module_dropdown], outputs=[motion_module_dropdown])
                
                motion_module_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def update_motion_module():
                    controller.refresh_motion_module()
                    return gr.Dropdown.update(choices=controller.motion_module_list)
                motion_module_refresh_button.click(fn=update_motion_module, inputs=[], outputs=[motion_module_dropdown])
                
                base_model_dropdown = gr.Dropdown(
                    label="Select base Dreambooth model (required)",
                    choices=controller.checkpoints_list,
                    value=controller.checkpoints_list[0],
                    interactive=True,
                )

                base_model_dropdown.change(fn=controller.update_base_model, inputs=[base_model_dropdown], outputs=[base_model_dropdown])

                
                personalized_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def update_personalized_model():
                    controller.refresh_personalized_model()
                    return [gr.Dropdown.update(choices=controller.checkpoints_list)]
                personalized_refresh_button.click(fn=update_personalized_model, inputs=[], outputs=[base_model_dropdown])

                # Load default models
                controller.update_stable_diffusion(stable_diffusion_dropdown.value)
                controller.update_motion_module(motion_module_dropdown.value)
                controller.update_base_model(base_model_dropdown.value)

        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. Configs for AnimateDiff.
                """
            )

            lora_ui_rows = []
            lora_dropdown_list = []

            with gr.Row():
                number_of_LoRAs = gr.Slider(0, max_LoRAs, value=0, step=1, label="How many LoRAs to show:")
                lora_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
            
            for i in range(max_LoRAs):
                with gr.Row(visible=False) as test:

                    # Change to use gr.State() instead of gr.Textbox()
                    lora_index = gr.Textbox(value=i, visible=False)

                    lora_model_dropdown = gr.Dropdown(
                            label=f"Select LoRA model {i} (optional)",
                            choices=["none"] + controller.lora_list,
                            value="none",
                            interactive=True,
                            elem_id=f"lora_model_dropdown-{i}",
                        )
                    
                    
                    lora_alpha_slider = gr.Slider(label="LoRA alpha", value=0.8, minimum=0, maximum=2, interactive=True)

                    def update_lora(lora_index, lora_model_dropdown, lora_alpha_slider):
                        index = int(lora_index)
                        if lora_model_dropdown == "none":
                            lora_path = "none"
                        else:
                            lora_path = os.path.exists(os.path.join(controller.loras_dir, lora_model_dropdown))
                        controller.project.loras[index] = {
                            "path": lora_path,
                            "alpha": lora_alpha_slider
                        }
                        print(controller.project.loras)
                        return

                    lora_model_dropdown.change(fn=update_lora, inputs=[lora_index, lora_model_dropdown, lora_alpha_slider])

                    lora_alpha_slider.change(fn=update_lora, inputs=[lora_index, lora_model_dropdown, lora_alpha_slider])

                    lora_dropdown_list.append(lora_model_dropdown)

                lora_ui_rows.append(test)

            def update_number_of_LoRAs(number):
                return [gr.Row.update(visible=True)]*number + [gr.Row.update(visible=False)]*(max_LoRAs-number)

            number_of_LoRAs.change(fn=update_number_of_LoRAs, inputs=number_of_LoRAs, outputs=lora_ui_rows)

            def update_lora_list():
                controller.refresh_lora_models()
                return [gr.Dropdown.update(choices=["none"] + controller.lora_list)]*max_LoRAs

            lora_refresh_button.click(fn=update_lora_list, inputs=[], outputs=lora_dropdown_list)

            with gr.Row():
                init_image_dropdown = gr.Dropdown(
                    label="Select init image",
                    choices=["none"] + controller.init_image_list,
                    value="none",
                    interactive=True,
                )

                init_image_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def update_init_image():
                    controller.refresh_init_images()
                    return gr.Dropdown.update(choices=["none"] + controller.init_image_list)
                init_image_refresh_button.click(fn=update_init_image, inputs=[], outputs=[init_image_dropdown])

            # init_image = gr.Textbox(label="Init image", value="/content/AnimateDiff/configs/prompts/yoimiya-init.jpg")
            prompt_textbox = gr.Textbox(label="Prompt", lines=2, value="1girl, yoimiya (genshin impact), origen, line, comet, wink, Masterpiece ，BestQuality ，UltraDetailed")
            negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2, value="NSFW, lr, nsfw,(sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt_v2, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, ((2girl)), (deformed fingers:1.2), (long fingers:1.2),succubus wings,horn,succubus horn,succubus hairstyle, (bad-artist-anime), bad-artist, bad hand, grayscale, skin spots, acnes, skin blemishes")
                
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    with gr.Row():
                        sampler_dropdown   = gr.Dropdown(label="Sampling method", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        sample_step_slider = gr.Slider(label="Sampling steps", value=25, minimum=10, maximum=100, step=1)
                        
                    width_slider     = gr.Slider(label="Width",            value=512, minimum=256, maximum=1024, step=64)
                    height_slider    = gr.Slider(label="Height",           value=512, minimum=256, maximum=1024, step=64)
                    length_slider    = gr.Slider(label="Animation length", value=16,  minimum=8,   maximum=24,   step=1)
                    cfg_scale_slider = gr.Slider(label="CFG Scale",        value=7.5, minimum=0,   maximum=20)
                    context_length  = gr.Slider(label="Context length",        value=20, minimum=10,   maximum=40, step=1)
                    context_overlap = gr.Slider(label="Context overlap",        value=20, minimum=10,   maximum=40, step=1)
                    context_stride = gr.Slider(label="Context stride",        value=0, minimum=0,   maximum=20, step=1)
                    fp16 = gr.Checkbox(label="FP16", value=True)
                    
                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=-1)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(fn=lambda: gr.Textbox.update(value=random.randint(1, 1e8)), inputs=[], outputs=[seed_textbox])
            
                    generate_button = gr.Button(value="Generate", variant='primary')
                    
                result_video = gr.Video(label="Generated Animation", interactive=False)

            generate_button.click(
                fn=controller.animate,
                inputs=[
                    stable_diffusion_dropdown,
                    motion_module_dropdown,
                    base_model_dropdown,
                    lora_alpha_slider,
                    init_image_dropdown,
                    prompt_textbox, 
                    negative_prompt_textbox, 
                    sampler_dropdown, 
                    sample_step_slider, 
                    width_slider, 
                    length_slider, 
                    height_slider, 
                    cfg_scale_slider, 
                    seed_textbox,
                    context_length,
                    context_stride,
                    context_overlap,
                    fp16
                ],
                outputs=[result_video]
            )
            
    return demo


if __name__ == "__main__":
    demo = ui()
    demo.queue(concurrency_count=3)
    demo.launch(share=True, debug=True)
