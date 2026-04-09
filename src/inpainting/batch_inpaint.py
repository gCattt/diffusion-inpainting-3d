from pathlib import Path
from PIL import Image, ImageFilter
import torch
import numpy as np
import yaml
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from .diffusion_pipeline import DiffusionInpaint, depth_tensor_to_control_pil


def load_config(path="configs/inpainting_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def sorted_views_by_mask_coverage(rgb_dir: Path, mask_dir: Path):
    scored = []
    for rgb_path in sorted(rgb_dir.glob("*.png")):
        name = rgb_path.stem
        mask_path = mask_dir / f"{name}.png"
        if not mask_path.exists():
            continue
        m = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        score = int(m.sum())
        scored.append((score, rgb_path))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]

def inpaint_main(cfg_path="configs/inpainting_config.yaml"):
    cfg = load_config(cfg_path)

    rgb_inpaint_dir = Path(cfg["rgb_inpaint_dir"])
    mask_dir = Path(cfg["mask_dir"])
    #depth_dir = Path(cfg["depth_dir"])
    face_idx_dir = cfg.get("face_idx_dir")
    face_idx_dir = Path(face_idx_dir) if face_idx_dir is not None else None
    inpainted_dir = Path(cfg["inpainted_dir"])
    inpainted_dir.mkdir(parents=True, exist_ok=True)

    base_model_id = cfg.get("base_model_id", "runwayml/stable-diffusion-inpainting")
    #controlnet_model_id = cfg.get("controlnet_model_id", "lllyasviel/sd-controlnet-depth")
    device = cfg.get("device", None)
    use_auth = cfg.get("use_auth_token", None)
    seed = cfg.get("seed", None)
    steps = cfg.get("num_inference_steps", 40)
    strength = float(cfg.get("strength", 0.85))
    guidance = cfg.get("guidance_scale", 5.0)
    padding_mask_crop = cfg.get("padding_mask_crop", 32)
    #control_scale = cfg.get("controlnet_conditioning_scale", 1.0)
    negative_prompt = cfg.get("negative_prompt", None)
    target_size = cfg.get("resize_to", None)

    max_views = cfg.get("max_views", 1)
    rank_by_coverage = bool(cfg.get("rank_views_by_mask_coverage", True))
    mask_dilate = int(cfg.get("mask_dilate", 0))
    same_noise_across_views = bool(cfg.get("same_noise_across_views", False))

    pipe = DiffusionInpaint(
        base_model_id=base_model_id,
        #controlnet_model_id=controlnet_model_id, 
        device=device, 
        use_auth_token=use_auth
    )

    if rank_by_coverage:
        rgb_files = sorted_views_by_mask_coverage(rgb_inpaint_dir, mask_dir)
    else:
        rgb_files = sorted(rgb_inpaint_dir.glob("*.png"))

    if max_views is not None:
        rgb_files = rgb_files[:max_views]

    # iterate RGB renders and corresponding mask files
    for rgb_path in tqdm(rgb_files, desc="inpainting"):
        name = rgb_path.stem

        mask_path = mask_dir / f"{name}.png"
        face_idx_path = face_idx_dir / f"{name}.pt" if face_idx_dir else None

        if not mask_path.exists() or face_idx_path is None or not face_idx_path.exists():
            print(f"missing data for {name}, skipping")
            continue

        image = Image.open(rgb_path).convert("RGB")
        raw_mask = Image.open(mask_path).convert("L")

        pix_to_face = torch.load(face_idx_path, map_location="cpu")
        if pix_to_face.ndim == 3 and pix_to_face.shape[-1] == 1:
            pix_to_face = pix_to_face.squeeze(-1)

        mesh_mask_bool = (pix_to_face >= 0).numpy()
        corruption_mask_bool = (np.array(raw_mask)) > 127

        # if resize is needed, resize masks first (NEAREST), then combine
        if target_size:
            image = image.resize((target_size, target_size), Image.LANCZOS)

            corruption_mask_img = Image.fromarray(
                (corruption_mask_bool.astype(np.uint8) * 255)
            ).resize((target_size, target_size), Image.NEAREST)
            corruption_mask_bool = (np.array(corruption_mask_img) > 127)

            mesh_mask_img = Image.fromarray(
                (mesh_mask_bool.astype(np.uint8) * 255)
            ).resize((target_size, target_size), Image.NEAREST)
            mesh_mask_bool = (np.array(mesh_mask_img) > 127)

        final_mask_bool = corruption_mask_bool.copy()

        # optional dilation on the final binary mask
        if mask_dilate and mask_dilate > 1:
            k = mask_dilate + 1 if mask_dilate % 2 == 0 else mask_dilate
            final_mask_bool = np.array(
                Image.fromarray((final_mask_bool.astype(np.uint8) * 255))
                .filter(ImageFilter.MaxFilter(k))
            ) > 127

        final_mask = Image.fromarray((final_mask_bool.astype(np.uint8) * 255))

        # neutralize background outside mesh
        image_np = np.array(image)
        image_np[~mesh_mask_bool] = 127
        image = Image.fromarray(image_np)

        # control_image = None
        # depth_path = depth_dir / f"{name}.pt"
        # if depth_path.exists():
        #     depth_t = torch.load(depth_path, map_location="cpu")
        #     valid_mask = None

        #     if face_idx_dir is not None:
        #         face_idx_path = face_idx_dir / f"{name}.pt"
        #         if face_idx_path.exists():
        #             pix_to_face = torch.load(face_idx_path, map_location="cpu")
        #             if pix_to_face.ndim == 3 and pix_to_face.shape[-1] == 1:
        #                 pix_to_face = pix_to_face.squeeze(-1)
        #             valid_mask = (pix_to_face >= 0).numpy().astype(np.uint8)

        #     control_image = depth_tensor_to_control_pil(
        #         depth_t,
        #         target_size=target_size,
        #         valid_mask=valid_mask,
        #     )
        # else:
        #     print(f"depth missing for {name}, continuing without control image")

        view_idx = 0
        if "_view" in name:
            try:
                view_idx = int(name.rsplit("_view", 1)[-1])
            except Exception:
                pass

        generator = None
        if seed is not None:
            local_seed = seed if same_noise_across_views else seed + view_idx
            generator = torch.Generator(device=pipe.device).manual_seed(local_seed)
 
        # inpaint
        out_img = pipe.inpaint(
            image=image,
            mask=final_mask,
            prompt=cfg.get("prompt", ""),
            #control_image=control_image,
            generator=generator,
            num_inference_steps=steps,
            strength=strength,
            guidance_scale=guidance,
            padding_mask_crop=padding_mask_crop,
            #controlnet_conditioning_scale=control_scale,
            negative_prompt=negative_prompt,
        )

        # blend the inpainted output with the original image using a soft mask to avoid hard edges
        orig = np.array(image)
        gen = np.array(out_img)
        m = (final_mask_bool.astype(np.float32))

        m_soft = gaussian_filter(m, sigma=1.0)
        m_soft = np.clip(m_soft, 0.0, 1.0)[..., None]

        blended = np.clip(gen * m_soft + orig * (1.0 - m_soft), 0, 255).astype(np.uint8)

        out_img = Image.fromarray(blended)
        out_path = inpainted_dir / f"{name}_inpainted.png"
        out_img.save(out_path)


if __name__ == "__main__":
    inpaint_main()