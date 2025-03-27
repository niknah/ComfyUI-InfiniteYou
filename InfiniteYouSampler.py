import os
import comfy
import torch
import sys
import folder_paths
import numpy
import pygit2
import hashlib
from PIL import Image

this_path = os.path.dirname(__file__)
repo_dir = os.path.join(this_path, "InfiniteYou")
requirements_txt = os.path.join(repo_dir, "requirements.txt")

if not os.path.exists(requirements_txt):
    pygit2.clone_repository(
        "https://github.com/bytedance/InfiniteYou",
        repo_dir,
        depth=1
        )
    if not os.path.exists(requirements_txt):
        print("*** Could not get InfiniteYou repository.  Please install git.")

sys.path.insert(0, this_path)
from InfiniteYou.pipelines.pipeline_infu_flux import (  # noqa: E402
    InfUFluxPipeline
)
sys.path.remove(this_path)


class InfiniteYouSampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        base_models = (
            ['auto'] +
            folder_paths.get_filename_list("diffusion_models")
        )
        return {
            "required": {
                "id_image": ("IMAGE",),
                "base_model": (base_models, ),
                "seed": ("INT", {
                    "default": 0
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "a man on a horse",
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
                "width": ("INT", {
                    "default": 864,
                }),
                "height": ("INT", {
                    "default": 1152,
                }),
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
        base_model,
        seed,
        prompt,
        steps=30,
        guidance=3.5,
        infusenet_conditioning_scale=1.0,
        infusenet_guidance_start=0.0,
        infusenet_guidance_end=1.0,
        model_version='aes_stage2',
        control_image=None,
        width=864,
        height=1152,
        enable_realism=False,
        enable_anti_blur=False,
    ):
        # torch.cuda.set_device(comfy.model_management.get_torch_device())

        infu_flux_version = 'v1.0'
        model_dir = 'ByteDance/InfiniteYou'
        if base_model == 'auto':
            base_model_path = 'black-forest-labs/FLUX.1-dev'
        else:
            # TODO
            base_model_path = folder_paths.get_full_path_or_raise(
                "diffusion_models", base_model
                )

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

        images = []
        i_upto = 0
        for id_image in id_image:
            control_image_pil = None
            if control_image is not None:
                c_upto = i_upto % len(control_image)
                control_i = 255. * control_image[c_upto].cpu().numpy()
                control_image_pil = Image.fromarray(
                    numpy.clip(control_i, 0, 255).astype(numpy.uint8)
                )

            i = 255. * id_image.cpu().numpy()
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
                width=width,
                height=height,
            )
            image = image.convert("RGB")
            image = numpy.array(image).astype(numpy.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            images.append(image)
            i_upto += 1

        if len(images) > 1:
            images = torch.cat(images, dim=0)
        else:
            images = images[0]

        return (images,)

    @classmethod
    def IS_CHANGED(
        self,
        id_image,
        base_model,
        seed,
        prompt,
        steps,
        guidance,
        infusenet_conditioning_scale,
        infusenet_guidance_start,
        infusenet_guidance_end,
        model_version,
        control_image,
        width,
        height,
        enable_realism,
        enable_anti_blur,
    ):
        m = hashlib.sha256()
        m.update(id_image)
        m.update(base_model)
        m.update(seed)
        m.update(prompt)
        m.update(steps)
        m.update(guidance)
        m.update(infusenet_conditioning_scale)
        m.update(infusenet_guidance_start)
        m.update(infusenet_guidance_end)
        m.update(model_version)
        m.update(control_image)
        m.update(width)
        m.update(height)
        m.update(enable_realism)
        m.update(enable_anti_blur)
        return m.digest().hex()
