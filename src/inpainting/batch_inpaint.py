from pathlib import Path
from PIL import Image, ImageFilter
import torch
import numpy as np
import yaml
from tqdm import tqdm
from .diffusion_pipeline import DiffusionInpaint, depth_tensor_to_control_pil


def load_config(path="configs/inpainting_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def inpaint_main(cfg_path="configs/inpainting_config.yaml"):
    cfg = load_config(cfg_path)

    rgb_inpaint_dir = Path(cfg["rgb_inpaint_dir"])
    mask_dir = Path(cfg["mask_dir"])
    depth_dir = Path(cfg["depth_dir"])
    face_idx_dir = Path(cfg.get("face_idx_dir"))
    inpainted_dir = Path(cfg["inpainted_dir"])
    inpainted_dir.mkdir(parents=True, exist_ok=True)

    base_model_id = cfg.get("base_model_id", "sd2-community/stable-diffusion-2-inpainting")
    controlnet_model_id = cfg.get("controlnet_model_id", "thibaud/controlnet-sd21-depth-diffusers")
    device = cfg.get("device", None)
    use_auth = cfg.get("use_auth_token", None)
    seed = cfg.get("seed", None)
    steps = cfg.get("num_inference_steps", 20)
    guidance = cfg.get("guidance_scale", 7.5)
    control_scale = cfg.get("controlnet_conditioning_scale", 1.0)
    negative_prompt = cfg.get("negative_prompt", None)
    target_size = cfg.get("resize_to", None)

    # init pipeline
    pipe = DiffusionInpaint(
        base_model_id=base_model_id,
        controlnet_model_id=controlnet_model_id, 
        device=device, 
        use_auth_token=use_auth
    )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # iterate RGB renders and corresponding mask files
    rgb_files = sorted(rgb_inpaint_dir.glob("*.png"))
    for rgb_path in tqdm(rgb_files, desc="inpainting"):
        name = rgb_path.stem
        mask_path = mask_dir / f"{name}.png"
        if not mask_path.exists():
            print(f"mask missing for {name}, skipping")
            continue

        image = Image.open(rgb_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # optionally resize to model expected size if specified
        if target_size:
            image = image.resize((target_size, target_size), Image.LANCZOS)
            mask = mask.resize((target_size, target_size), Image.NEAREST)

        # binary mask force
        mask_np = np.array(mask)
        mask_np = np.where(mask_np > 127, 255, 0).astype(np.uint8)
        mask = Image.fromarray(mask_np)

        mask = mask.filter(ImageFilter.MaxFilter(3))

        control_image = None
        depth_path = depth_dir / f"{name}.pt"
        face_idx_path = face_idx_dir / f"{name}.pt"
        if not depth_path.exists():
            print(f"depth missing for {name}, skipping control image")
        else:
            depth_t = torch.load(depth_path)
            valid_mask = None
            if face_idx_path.exists():
                pix_to_face = torch.load(face_idx_path)
                if pix_to_face.ndim == 3 and pix_to_face.shape[-1] == 1:
                    pix_to_face = pix_to_face.squeeze(-1)
                valid_mask = (pix_to_face >= 0).cpu().numpy().astype(np.uint8)  # 1 visible, 0 background
            # normalize/resample in helper
            control_image = depth_tensor_to_control_pil(depth_t, target_size, valid_mask=valid_mask)

        out_img = pipe.inpaint(
            image=image,
            mask=mask,
            prompt=cfg.get("prompt", ""),
            control_image=control_image,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=control_scale,
            negative_prompt=negative_prompt,
        )

        out_path = inpainted_dir / f"{name}_inpainted.png"
        out_img.save(out_path)

if __name__ == "__main__":
    inpaint_main()