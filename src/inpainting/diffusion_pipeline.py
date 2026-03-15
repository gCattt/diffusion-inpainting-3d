from typing import Optional
from PIL import Image
import torch
import os

try:
    from diffusers import StableDiffusionInpaintPipeline
except Exception as e:
    raise ImportError("diffusers not available. Install `diffusers` and relevant deps.") from e


class DiffusionInpaint:
    """
    Minimal wrapper over Hugging Face diffusers StableDiffusionInpaintPipeline.

    Usage:
        p = DiffusionInpaint(model_id="runwayml/stable-diffusion-inpainting", device="cuda")
        out = p.inpaint(image_pil, mask_pil, prompt="...", num_inference_steps=30)
    """
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        use_auth_token: Optional[str] = None,
    ):
        # select device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # prefer fp16 on cuda if not explicitly set
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # load pipeline
        load_kwargs = {}
        if use_auth_token is not None:
            load_kwargs["use_auth_token"] = use_auth_token

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            **load_kwargs,
        )

        self.pipe.safety_checker = None

        # move to device
        self.pipe = self.pipe.to(self.device)

        # speed / memory options
        try:
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_attention_slicing("max")
        except Exception:
            pass

        # optional: xformers if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Image.Image:
        """
        Run a single inpaint call. `image` is the original RGB render (PIL), `mask` is the mask image:
        mask white (255) = area to inpaint by SD inpaint API (the HF pipeline expects white regions to be replaced).
        Returns a PIL image.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        if mask.mode != "L":
            mask = mask.convert("L")

        # pipeline returns a dict-like object with 'images'
        result = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
        )

        out_img = result.images[0]
        return out_img