import os
import comfy
import torch
import torchvision
import sys
import folder_paths
import numpy
import pygit2
from PIL import Image, ImageOps

this_path = os.path.dirname(__file__)
repo_dir = os.path.join(this_path,"InfiniteYou")
requirements_txt = os.path.join(repo_dir,"requirements.txt")

if not os.path.exists(requirements_txt):
    pygit2.clone_repository("https://github.com/azmenak/InfiniteYou", repo_dir, depth=1)
    if not os.path.exists(requirements_txt):
        print("*** Could not get InfiniteYou repository.  Please install git.")

sys.path.insert(0, this_path)
from InfiniteYou.pipelines.pipeline_infu_flux import InfUFluxPipeline
sys.path.remove(this_path)

class InfiniteYouSampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id_image": ("IMAGE",),
                "base_model": (['auto'] + folder_paths.get_filename_list("diffusion_models"),),
                "seed": ("INT", {
                    "default": 42
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "a robot",
                    "lazy": True
                }),
                "steps": ("INT", {
                    "default": 30
                }),
            },
            "optional": {
                "guidance": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                    "lazy": True
                }),
                "infusenet_conditioning_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                    "lazy": True
                }),
                "infusenet_guidance_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                    "lazy": True
                }),
                "infusenet_guidance_end": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                    "lazy": True
                }),
                "model_version": (['aes_stage2', 'sim_stage1'], {
                    "default": 'aes_stage2'
                }),
                "control_image": ("IMAGE",),
                "enable_realism": ("BOOLEAN", {
                    "default": False
                }),
                "enable_anti_blur": ("BOOLEAN", {
                    "default": False
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sampler"
    CATEGORY = "sampling"

    def sampler(
        self,
        id_image,
        base_model,  # TODO
        seed,
        prompt,
        steps=30,
        guidance=3.5,
        infusenet_conditioning_scale=1.0,
        infusenet_guidance_start=0.0,
        infusenet_guidance_end=1.0,
        model_version='aes_stage2',
        control_image=None,
        enable_realism=False,
        enable_anti_blur=False,
    ):
        device = comfy.model_management.get_torch_device()
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        elif device.type == 'mps':
            torch.set_default_device("mps:0")

        infu_flux_version = 'v1.0'
        model_dir = 'ByteDance/InfiniteYou'
        if base_model == 'auto':
            base_model_path = 'black-forest-labs/FLUX.1-dev'
        else:
            base_model_path = folder_paths.get_full_path_or_raise("diffusion_models", base_model)

        infu_model_path = os.path.join(
            model_dir, f'infu_flux_{infu_flux_version}',
            model_version
        )
        insightface_root_path = os.path.join(
            model_dir, 'supports', 'insightface'
            )
        pipe = InfUFluxPipeline(
            base_model_path=base_model_path,
            infu_model_path=infu_model_path,
            insightface_root_path=insightface_root_path,
            infu_flux_version=infu_flux_version,
            model_version=model_version,
        )

        # Load LoRAs if enabled
        lora_dir = os.path.join(model_dir, 'supports', 'optional_loras')
        if not os.path.exists(lora_dir): 
            lora_dir = "./models/InfiniteYou/supports/optional_loras"
        loras = []
        if enable_realism:
            loras.append([os.path.join(lora_dir, 'flux_realism_lora.safetensors'), 'realism', 1.0])
        if enable_anti_blur:
            loras.append([os.path.join(lora_dir, 'flux_anti_blur_lora.safetensors'), 'anti_blur', 1.0])
        if loras:
            pipe.load_loras(loras)

        control_image_pil = None
        if control_image is not None:
            control_i = 255. * control_image[0].cpu().numpy()
            control_image_pil = Image.fromarray(numpy.clip(i, 0, 255).astype(numpy.uint8))
        i = 255. * id_image[0].cpu().numpy()
        img = Image.fromarray(numpy.clip(i, 0, 255).astype(numpy.uint8))
        image = pipe(
            id_image=img,
            prompt=prompt,
            control_image=control_image_pil,
            seed=seed,
            guidance_scale=guidance,
            num_steps=steps,
            infusenet_conditioning_scale=infusenet_conditioning_scale,
            infusenet_guidance_start=infusenet_guidance_start,
            infusenet_guidance_end=infusenet_guidance_end,
        )

        image = image.convert("RGB")
        image = numpy.array(image).astype(numpy.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        images = []
        images.append(image)

        if len(images) > 1:
            images = torch.cat(images, dim=0)
        else:
            images = images[0]

        return (images,)
