from typing import Optional
from PIL import Image
import torch
import numpy as np

try:
    from diffusers import (
        StableDiffusionInpaintPipeline,
        ControlNetModel,
        StableDiffusionControlNetInpaintPipeline,
        UniPCMultistepScheduler,
    )
except Exception as e:
    raise ImportError("diffusers not available. Install `diffusers[torch]` and relevant deps.") from e


class DiffusionInpaint:
    """
    Minimal wrapper over Hugging Face diffusers.
    """
    def __init__(
        self,
        base_model_id: str,
        controlnet_model_id: str,
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
        self.torch_dtype = torch_dtype

        # load pipeline
        load_kwargs = {}
        if use_auth_token is not None:
            load_kwargs["use_auth_token"] = use_auth_token

        controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=self.torch_dtype, **load_kwargs)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
           base_model_id,
           controlnet=controlnet,
           torch_dtype=self.torch_dtype,
           **load_kwargs,
        )

        # self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     base_model_id,
        #     torch_dtype=self.torch_dtype,
        #     **load_kwargs,
        # )
        
        # scheduler improvement (optional)
        try:
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        except Exception:
            pass

        # disable safety checker for debugging (careful if publishing outputs)
        try:
            self.pipe.safety_checker = None
        except Exception:
            pass

        # move to device and enable memory-saving features (best-effort)
        self.pipe = self.pipe.to(self.device)

        # try:
        #     # accelerate-backed offload if available
        #     self.pipe.enable_model_cpu_offload()
        # except Exception:
        #     pass

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                self.pipe.enable_attention_slicing()

            try:
                self.pipe.enable_vae_slicing()
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
        """
        Run a single inpaint call:
        image: original RGB render (PIL)
        mask: L image (white = areas to inpaint)
        control_image: PIL (depth->RGB) or None

        Returns a PIL image.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        if mask.mode != "L":
            mask = mask.convert("L")
        if control_image is not None and control_image.mode != "RGB":
           control_image = control_image.convert("RGB")

        with torch.inference_mode():
            # pipeline returns a dict-like object with 'images'
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

# helper: convert depth tensor (torch) or numpy array to PIL RGB depth image
def depth_tensor_to_control_pil(depth_tensor, target_size: Optional[int] = None, valid_mask: Optional[np.ndarray] = None):
    """
    depth_tensor: torch.Tensor (H,W) or numpy array.
    Normalizes depth to [0,255] and returns RGB PIL.
    """
    if isinstance(depth_tensor, torch.Tensor):
        d = depth_tensor.detach().cpu().float().numpy()
    else:
        d = np.array(depth_tensor).astype("float32")

    # handle inf/nan
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    if valid_mask is not None:
            # valid_mask is 1 where surface exists
            d = d.copy()
            d[valid_mask == 0] = np.nan  # mark background as nan to normalize out, will set to 0 later
            
    # mn = float(np.nanmin(d)) if np.isfinite(np.nanmin(d)) else 0.0
    # mx = float(np.nanmax(d)) if np.isfinite(np.nanmax(d)) else mn + 1.0

    mn = np.nanpercentile(d, 2)
    mx = np.nanpercentile(d, 98)
    if mx - mn > 1e-8:
        # dn = (np.nan_to_num(d, nan=mn) - mn) / (mx - mn)
        dn = (d - mn) / (mx - mn)
    else:
        dn = np.zeros_like(d)

    dn = np.clip(dn, 0.0, 1.0)
    dn = np.nan_to_num(dn, nan=0.0)

    # ensure background = 0
    if valid_mask is not None:
        dn[valid_mask == 0] = 0.0

    arr = (dn * 255.0).astype("uint8")
    img = Image.fromarray(arr, mode="L").convert("RGB")
    if target_size is not None:
        img = img.resize((target_size, target_size), Image.BILINEAR)
    return img