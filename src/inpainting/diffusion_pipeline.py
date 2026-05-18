from typing import Optional
from PIL import Image
import torch
import numpy as np

try:
    from diffusers import (
        StableDiffusionInpaintPipeline,
        StableDiffusionControlNetInpaintPipeline,
        ControlNetModel,
        UniPCMultistepScheduler,
    )
except Exception as e:
    raise ImportError("diffusers not available. Install `diffusers[torch]` and relevant deps.") from e


class DiffusionInpaint:
    def __init__(
        self,
        base_model_id: str,
        controlnet_model_id: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        use_auth_token: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.torch_dtype = torch_dtype

        load_kwargs = {}
        if use_auth_token is not None:
            load_kwargs["use_auth_token"] = use_auth_token

        # self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     base_model_id,
        #     torch_dtype=self.torch_dtype,
        #     **load_kwargs,
        # )

        controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=self.torch_dtype, **load_kwargs)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
           base_model_id,
           controlnet=controlnet,
           torch_dtype=self.torch_dtype,
           **load_kwargs,
        )

        try:
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        except Exception:
            pass

        try:
            self.pipe.safety_checker = None
        except Exception:
            pass

        self.pipe = self.pipe.to(self.device)

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # try:
            #     self.pipe.enable_model_cpu_offload()
            # except Exception:
            #     pass

            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                self.pipe.enable_attention_slicing()

            try:
                self.pipe.vae.enable_slicing()
                self.pipe.vae.enable_tiling()
            except Exception:
                pass

        self.pipe.unet.to(memory_format=torch.channels_last)
        self.pipe.unet.eval()
        self.pipe.vae.eval()

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        control_image: Optional[Image.Image] = None,
        num_inference_steps: int = 40,
        strength: float = 0.85,
        guidance_scale: float = 5.0,
        padding_mask_crop: Optional[int] = None,
        controlnet_conditioning_scale: float = 1.0,
        negative_prompt: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        if mask.mode != "L":
            mask = mask.convert("L")
        mask = np.array(mask)
        mask = (mask > 127).astype(np.uint8) * 255
        mask = Image.fromarray(mask)

        if control_image is not None and control_image.mode != "RGB":
           control_image = control_image.convert("RGB")

        with torch.inference_mode():
            result = self.pipe(
                image=image,
                mask_image=mask,
                prompt=prompt,
                control_image=control_image,
                num_inference_steps=num_inference_steps,
                strength=strength,
                guidance_scale=guidance_scale,
                padding_mask_crop=padding_mask_crop,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                negative_prompt=negative_prompt,
                generator=generator,
            )

        out_img = result.images[0]
        return out_img

class DiffusionRefine:
    def __init__(
        self,
        base_model_id: str,
        ip_adapter_model_id: str = "h94/IP-Adapter",
        ip_adapter_weight: str = "ip-adapter_sd15.bin",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        use_auth_token: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.torch_dtype = torch_dtype

        load_kwargs = {}
        if use_auth_token is not None:
            load_kwargs["use_auth_token"] = use_auth_token

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            base_model_id,
            torch_dtype=self.torch_dtype,
            **load_kwargs,
        )

        self.pipe = self.pipe.to(self.device)

        self.pipe.load_ip_adapter(
            ip_adapter_model_id,
            subfolder="models",
            weight_name=ip_adapter_weight,
        )
        self.pipe.set_ip_adapter_scale(0.0)

        try:
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        except Exception:
            pass

        try:
            self.pipe.safety_checker = None
        except Exception:
            pass

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # try:
            #     self.pipe.enable_model_cpu_offload()
            # except Exception:
            #     pass

            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                self.pipe.enable_attention_slicing()

            try:
                self.pipe.vae.enable_slicing()
            except Exception:
                pass

        self.pipe.unet.to(memory_format=torch.channels_last)
        self.pipe.unet.eval()
        self.pipe.vae.eval()

    def refine(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        ip_adapter_image: Optional[Image.Image] = None,
        refinement_steps: int = 20,
        refinement_strength: float = 0.2,
        refinement_guidance_scale: float = 5.0,
        ip_adapter_scale: float = 0.05,
        negative_prompt: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        if mask.mode != "L":
            mask = mask.convert("L")
        mask = np.array(mask)
        mask = (mask > 127).astype(np.uint8) * 255
        mask = Image.fromarray(mask)

        if ip_adapter_image is not None:
            if ip_adapter_image.mode != "RGB":
                ip_adapter_image = ip_adapter_image.convert("RGB")
        else:
            ip_adapter_image = image

        self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        with torch.inference_mode():
            image_embeds_data = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=[ip_adapter_image],
                ip_adapter_image_embeds=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            # resolve AttributeError
            sanitized_embeds = []
            for item in image_embeds_data:
                if isinstance(item, (tuple, list)):
                    concatenated = torch.cat([item[0], item[1]], dim=0)
                    sanitized_embeds.append(concatenated)
                else:
                    sanitized_embeds.append(item)

            result = self.pipe(
                image=image,
                mask_image=mask,
                prompt=prompt,
                # ip_adapter_image= ip_adapter_image,
                ip_adapter_image_embeds=sanitized_embeds,
                num_inference_steps=refinement_steps,
                strength=refinement_strength,
                guidance_scale=refinement_guidance_scale,
                negative_prompt=negative_prompt,
                generator=generator,
            )

        out_img = result.images[0]
        return out_img

def depth_tensor_to_control_pil(depth_tensor, target_size: Optional[int] = None, valid_mask: Optional[np.ndarray] = None):
    if isinstance(depth_tensor, torch.Tensor):
        d = depth_tensor.detach().cpu().float().numpy()
    else:
        d = np.array(depth_tensor).astype("float32")

    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    if valid_mask is not None:
            d = d.copy()
            d[valid_mask == 0] = np.nan

    mn = np.nanpercentile(d, 2)
    mx = np.nanpercentile(d, 98)
    if mx - mn > 1e-8:
        dn = (d - mn) / (mx - mn)
    else:
        dn = np.zeros_like(d)

    dn = np.clip(dn, 0.0, 1.0)
    dn = np.nan_to_num(dn, nan=0.0)

    if valid_mask is not None:
        dn[valid_mask == 0] = 0.0

    arr = (dn * 255.0).astype("uint8")
    img = Image.fromarray(arr, mode="L").convert("RGB")
    if target_size is not None:
        img = img.resize((target_size, target_size), Image.BILINEAR)
    return img