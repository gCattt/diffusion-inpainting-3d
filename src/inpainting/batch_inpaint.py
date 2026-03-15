from pathlib import Path
from PIL import Image
import torch
import yaml
import os
from tqdm import tqdm
from .diffusion_pipeline import DiffusionInpaint


def load_config(path="configs/inpainting_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def inpaint_main(cfg_path="configs/inpainting_config.yaml"):
    cfg = load_config(cfg_path)

    rgb_dir = Path(cfg["rgb_dir"])
    mask_dir = Path(cfg["mask_dir"])
    inpainted_dir = Path(cfg["inpainted_dir"])
    inpainted_dir.mkdir(parents=True, exist_ok=True)

    model_id = cfg.get("model_id", "runwayml/stable-diffusion-inpainting")
    device = cfg.get("device", None)
    steps = cfg.get("num_inference_steps", 30)
    guidance = cfg.get("guidance_scale", 7.5)
    negative_prompt = cfg.get("negative_prompt", None)
    use_auth = cfg.get("use_auth_token", None)

    # init pipeline
    print(f"Loading inpaint model {model_id} on {device or 'auto'} ...")
    pipe = DiffusionInpaint(model_id=model_id, device=device, use_auth_token=use_auth)

    # iterate RGB renders and corresponding mask files
    rgb_files = sorted(rgb_dir.glob("*.png"))
    for rgb_path in tqdm(rgb_files, desc="inpainting"):
        name = rgb_path.stem
        # expected mask names: same stem (or you can define mapping rules in cfg)
        mask_path = mask_dir / f"{name}.png"
        if not mask_path.exists():
            print(f"mask missing for {name}, skipping")
            continue

        image = Image.open(rgb_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # optionally resize to model expected size if specified
        target_size = cfg.get("resize_to", None)
        if target_size:
            image = image.resize((target_size, target_size), Image.LANCZOS)
            mask = mask.resize((target_size, target_size), Image.NEAREST)

        out = pipe.inpaint(
            image=image,
            mask=mask,
            prompt=cfg.get("prompt", ""),
            num_inference_steps=steps,
            guidance_scale=guidance,
            negative_prompt=negative_prompt,
        )

        out_path = inpainted_dir / f"{name}_inpainted.png"
        out.save(out_path)

    print("Batch inpainting complete. Results in:", inpainted_dir)


if __name__ == "__main__":
    inpaint_main()