from pathlib import Path
from PIL import Image
import torch
import numpy as np
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import re

from .diffusion_pipeline import DiffusionInpaint, DiffusionRefine, depth_tensor_to_control_pil
from src.utils.config_utils import load_yaml_config


def camera_similarity(cam1, cam2):
    R1, T1 = cam1["R"].squeeze(0), cam1["T"].squeeze(0)
    R2, T2 = cam2["R"].squeeze(0), cam2["T"].squeeze(0)

    C1 = -R1.T @ T1
    C2 = -R2.T @ T2

    C1 = C1 / (torch.norm(C1) + 1e-8)
    C2 = C2 / (torch.norm(C2) + 1e-8)

    cos_sim = torch.clamp(torch.dot(C1, C2), -1.0, 1.0)
    angle = torch.acos(cos_sim)

    return angle.item()

def compute_overlap(curr_faces, curr_mask, ref_faces):
    valid_curr = (curr_faces >= 0)
    target_mask = curr_mask & valid_curr
    if target_mask.sum() == 0:
        return 0.0
    
    faces_to_fill = curr_faces[target_mask]

    valid_ref = (ref_faces >= 0)
    faces_in_ref = np.unique(ref_faces[valid_ref])
    if faces_in_ref.size == 0:
        return 0.0

    covered_pixels = np.isin(
        faces_to_fill,
        faces_in_ref,
        assume_unique=False
    )

    overlap_score = covered_pixels.sum() / target_mask.sum()
    
    return float(overlap_score)

def precompute_views_data(rgb_dir, mask_dir, face_idx_dir, cameras_dir):
    views_data = {}

    for rgb_path in sorted(rgb_dir.rglob("*.png")):
        name = rgb_path.stem
        mesh_name = rgb_path.parent.name
        
        mask_path = mask_dir / mesh_name / f"{name}.png"
        face_idx_path = face_idx_dir / mesh_name / f"{name}.pt"
        cam_path = cameras_dir / mesh_name / f"{name}.pt"

        if not (mask_path.exists() and face_idx_path.exists() and cam_path.exists()):
            continue

        try:
            with Image.open(mask_path) as im:
                corruption_mask = (np.array(im.convert("L")) > 127)

            pix_to_face = torch.load(face_idx_path, map_location="cpu")
            if pix_to_face.ndim == 3 and pix_to_face.shape[-1] == 1:
                pix_to_face = pix_to_face.squeeze(-1)

            faces = pix_to_face
            if isinstance(faces, torch.Tensor):
                faces = faces.detach().cpu().numpy()

            cam = torch.load(cam_path, map_location="cpu")
        except Exception as e:
            print(f"Error precomputing view data (mask/faces/camera) for {name}: {e}")
            continue

        mesh_mask = (faces >= 0)
        valid_context = ((~corruption_mask) & mesh_mask).sum()

        views_data[name] = {
            "rgb_path": rgb_path,
            "mask_path": mask_path,
            "face_idx_path": face_idx_path,
            "cam_path": cam_path,
            "faces": faces,
            "cam": cam,
            "corruption_mask": corruption_mask,
            "coverage": valid_context,
            "mesh_name": mesh_name,
        }

    return views_data

def order_views_greedy(views_data, cfg):
    views = list(views_data.items())

    views.sort(key=lambda x: x[1]["coverage"], reverse=True)
    if len(views) == 0:
        return []

    best_coverage = views[0][1]["coverage"]
    ordered = [views.pop(0)]
    remaining = views

    max_angle = cfg.get("max_angle", 1.0)
    while len(remaining) > 0:
        best_score = -1
        best_idx = -1

        for i, (name, v) in enumerate(remaining):
            scores = []
            curr_cam = v["cam"]
            curr_faces = v["faces"]
            curr_mask = v["corruption_mask"]

            for _, ref in ordered:
                ref_faces = ref["faces"]
                ref_cam = ref["cam"]

                angle = camera_similarity(curr_cam, ref_cam)
                if angle > max_angle:
                    continue

                overlap = compute_overlap(curr_faces, curr_mask, ref_faces)
                if overlap < cfg.get("min_overlap", 0.1):
                    continue

                angle_score = 1 - angle / max_angle
                score = overlap * 0.8 + angle_score * 0.2
                scores.append(score)

            if len(scores) > 0:
                best_local_score = max(scores)
            else:
                best_local_score = 0.0

            score_total = best_local_score + 0.05 * (v["coverage"] / (best_coverage + 1e-6))

            if score_total > best_score:
                best_score = score_total
                best_idx = i

        if best_idx == -1:
            best_idx = 0

        ordered.append(remaining.pop(best_idx))

    return [v["rgb_path"] for _, v in ordered]

def inpaint_main():
    cfg = load_yaml_config("configs/inpainting_config.yaml")

    rgb_inpaint_dir = Path(cfg["rgb_inpaint_dir"])
    mask_dir = Path(cfg["mask_dir"])
    depth_dir = Path(cfg["depth_dir"])
    face_idx_dir = cfg.get("face_idx_dir")
    face_idx_dir = Path(face_idx_dir) if face_idx_dir is not None else None
    cameras_dir = Path(cfg["cameras_dir"])
    inpainted_dir = Path(cfg["inpainted_dir"])
    inpainted_dir.mkdir(parents=True, exist_ok=True)

    base_model_id = cfg.get("base_model_id", "runwayml/stable-diffusion-inpainting")
    device = cfg["device"]
    use_auth = cfg.get("use_auth_token")

    mask_dilate = int(cfg.get("mask_dilate", 0))
    same_noise_across_views = bool(cfg.get("same_noise_across_views", False))

    seed = cfg.get("seed")
    padding_mask_crop = cfg.get("padding_mask_crop", None)
    if padding_mask_crop is not None:
        padding_mask_crop = int(padding_mask_crop)
    negative_prompt = cfg.get("negative_prompt", "")
    target_size = cfg.get("resize_to")
    prompt = cfg.get("prompt", "")

    inpaint_pipe = DiffusionInpaint(
        base_model_id=base_model_id,
        controlnet_model_id=cfg.get("controlnet_model_id", "lllyasviel/sd-controlnet-depth"), 
        device=device, 
        use_auth_token=use_auth
    )

    refine_pipe = DiffusionRefine(
        base_model_id=base_model_id,
        ip_adapter_model_id=cfg.get("ip_adapter_model", "h94/IP-Adapter"),
        ip_adapter_weight=cfg.get("ip_adapter_weight", "ip-adapter_sd15.bin"),
        device=device,
        use_auth_token=use_auth,
    )
  
    if face_idx_dir is None:
        raise ValueError("face_idx_dir is required for multiview ordering")

    views_data = precompute_views_data(
        rgb_inpaint_dir,
        mask_dir,
        face_idx_dir,
        cameras_dir
    )

    rgb_files = order_views_greedy(views_data, cfg)

    max_views = cfg.get("max_views", 1)
    if isinstance(max_views, int) and max_views > 0:
        rgb_files = rgb_files[:max_views]

    inpainted_cache = {}
    ordered_names = [p.stem for p in rgb_files]
    for name in tqdm(ordered_names):
        parts = name.split('_')
        exclude = r'^(1k|2k|4k|8k|16k|view\d+|png)$'
        keywords = [p for p in parts if not p.isdigit() and not re.match(exclude, p.lower())]
        object_context = " ".join(keywords[:5])
        current_prompt = f"{object_context} {prompt}" 

        curr_data = views_data[name]

        rgb_path = curr_data["rgb_path"]
        with Image.open(rgb_path) as im:
            image = im.convert("RGB").copy()

        mesh_mask_bool = (curr_data["faces"] >= 0)
        corruption_mask_bool = curr_data["corruption_mask"]

        if target_size is not None:
            image = image.resize((target_size, target_size), Image.LANCZOS)

            corruption_mask_img = Image.fromarray(
                (corruption_mask_bool.astype(np.uint8) * 255)
            ).resize((target_size, target_size), Image.NEAREST)
            corruption_mask_bool = (np.array(corruption_mask_img) > 127)

            mesh_mask_img = Image.fromarray(
                (mesh_mask_bool.astype(np.uint8) * 255)
            ).resize((target_size, target_size), Image.NEAREST)
            mesh_mask_bool = (np.array(mesh_mask_img) > 127)

        final_mask_bool = corruption_mask_bool & mesh_mask_bool
        if final_mask_bool.sum() == 0:
            print(f"Skipping {name}: inpaint region empty (no overlap between corruption and mesh masks).")
            continue

        if mask_dilate and mask_dilate > 0:
            k = mask_dilate
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            final_mask_bool = cv2.dilate(final_mask_bool.astype(np.uint8), kernel).astype(bool)
            final_mask_bool &= mesh_mask_bool

        final_mask_for_sd = Image.fromarray((final_mask_bool.astype(np.uint8) * 255))

        image_np = np.array(image)

        temp_img = image_np.copy()

        kernel_context = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        sampling_area = cv2.dilate(final_mask_bool.astype(np.uint8), kernel_context).astype(bool)
        local_context_mask = sampling_area & mesh_mask_bool & (~final_mask_bool)
        local_context_pixels = image_np[local_context_mask]

        if len(local_context_pixels) > 0:
            local_fill_color = np.median(local_context_pixels, axis=0).astype(np.uint8)
        else:
            valid_mesh_pixels = image_np[mesh_mask_bool & ~final_mask_bool]
            local_fill_color = np.median(valid_mesh_pixels, axis=0).astype(np.uint8) if len(valid_mesh_pixels) > 0 else [127,127,127]

        temp_img[final_mask_bool] = local_fill_color
        # background completion
        bg_mask_cv = (~mesh_mask_bool).astype(np.uint8) * 255
        image_padded_np = cv2.inpaint(temp_img, bg_mask_cv, inpaintRadius=20, flags=cv2.INPAINT_TELEA)
        # hole completion
        image_padded_np[final_mask_bool] = [0, 0, 0]
        holes_mask_cv = final_mask_bool.astype(np.uint8) * 255
        image_ready_for_sd = cv2.inpaint(image_padded_np, holes_mask_cv, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

        image_for_sd = Image.fromarray(image_ready_for_sd)

        control_image = None
        mesh_name = views_data[name]["mesh_name"]
        depth_path = depth_dir / mesh_name / f"{name}.pt"

        if depth_path.exists():
            depth_t = torch.load(depth_path, map_location="cpu")

            valid_mask = mesh_mask_bool.astype(np.uint8)

            control_image = depth_tensor_to_control_pil(
                depth_t,
                target_size=target_size,
                valid_mask=valid_mask,
            )
        else:
            print(f"Depth map not found at {depth_path}, proceeding without ControlNet guidance for {name}")

        view_idx = 0
        if "_view" in name:
            try:
                view_idx = int(name.rsplit("_view", 1)[-1])
            except Exception:
                pass

        generator = None
        if seed is not None:
            local_seed = seed if same_noise_across_views else seed + view_idx
            generator = torch.Generator(device=inpaint_pipe.device).manual_seed(local_seed)

        best_score = -1.0
        ip_adapter_image = None
        if len(inpainted_cache) > 0:
            curr_cam = curr_data["cam"]
            curr_faces = curr_data["faces"]
            curr_mask = curr_data["corruption_mask"]

            best_score = -1
            best_ref = None

            for _, ref_data in inpainted_cache.items():
                ref_cam = ref_data["cam"]
                ref_faces = ref_data["faces"]

                angle = camera_similarity(curr_cam, ref_cam)
                max_angle = cfg.get("max_angle", 1.0)
                if angle > max_angle:
                    continue

                overlap = compute_overlap(curr_faces, curr_mask, ref_faces)
                if overlap < cfg.get("min_overlap", 0.05):
                    continue

                angle_score = 1 - angle / max_angle
                angle_score = np.clip(angle_score, 0.0, 1.0)

                score = overlap * 0.85 + angle_score * 0.15

                if score > best_score:
                    best_score = score
                    with Image.open(ref_data["path"]) as ref_im:
                        best_ref = ref_im.convert("RGB").copy()

            ip_scale = float(cfg.get("ip_adapter_scale", 0.1))
            if best_ref is not None:
                ip_adapter_image = best_ref.resize(image_for_sd.size)

                max_target_scale = 0.3
                ip_scale = max_target_scale * (best_score ** 0.5)
                ip_scale = float(np.clip(ip_scale, 0.0, max_target_scale))
            else:
                print(f"\n[WARNING] No cached reference meets similarity criteria for {name}", flush=True)

        out_img = inpaint_pipe.inpaint(
            image=image_for_sd,
            mask=final_mask_for_sd,
            prompt=current_prompt,
            control_image=control_image,
            num_inference_steps=int(cfg.get("num_inference_steps", 25)),
            strength=float(cfg.get("strength", 0.85)),
            guidance_scale=float(cfg.get("guidance_scale", 5.0)),
            padding_mask_crop=padding_mask_crop,
            controlnet_conditioning_scale=float(cfg.get("controlnet_conditioning_scale", 0.4)),
            negative_prompt=negative_prompt,
            generator=generator,
        )

        refinement_thresh = float(cfg.get("refinement_thresh", 0.2))
        if ip_adapter_image is not None and best_score > refinement_thresh:
            out_img = refine_pipe.refine(
                image=out_img,
                mask=final_mask_for_sd,
                prompt=current_prompt,
                ip_adapter_image=ip_adapter_image,
                refinement_steps=int(cfg.get("refinement_steps", 10)),
                refinement_strength=float(cfg.get("refinement_strength", 0.10)),
                refinement_guidance_scale=float(cfg.get("refinement_guidance_scale", 4.5)),
                ip_adapter_scale=ip_scale,
                negative_prompt=negative_prompt,
                generator=generator,
            )

        orig = np.array(image)
        gen = np.array(out_img)

        m = final_mask_bool.astype(np.float32)
        if m.ndim == 2:
            m = m[..., None]

        m_soft = gaussian_filter(m, sigma=0.6)
        m_soft = np.clip(m_soft * 2.2, 0.0, 1.0)
        if mesh_mask_bool.ndim == 2:
            mmb = mesh_mask_bool[..., None]
        else:
            mmb = mesh_mask_bool
        m_soft = m_soft * mmb.astype(np.float32)
        blended = np.clip(gen.astype(np.float32) * m_soft + orig.astype(np.float32) * (1.0 - m_soft), 0, 255).astype(np.uint8)

        out_img = Image.fromarray(blended)

        mesh_name = views_data[name].get("mesh_name", "")
        if mesh_name:
            mesh_inpainted_dir = inpainted_dir / mesh_name
            mesh_inpainted_dir.mkdir(parents=True, exist_ok=True)
            out_path = mesh_inpainted_dir / f"{name}_inpainted.png"
        else:
            out_path = inpainted_dir / f"{name}_inpainted.png"
        
        out_img.save(out_path)

        inpainted_cache[name] = {
            "path": out_path,
            "faces": views_data[name]["faces"],
            "cam": views_data[name]["cam"],
        }

        del out_img
        del orig, gen, m, m_soft, blended
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    inpaint_main()