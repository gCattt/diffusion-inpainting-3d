from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
import yaml
from pytorch3d.io import load_obj
from src.utils.io_utils import group_inpainted_by_mesh, resolve_assets_for_mesh

def load_config(path="configs/texture_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_rgb(arr, path: Path):
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)

def load_mesh_uv(mesh_path: Path):
    verts, faces, aux = load_obj(str(mesh_path), load_textures=False)

    if aux.verts_uvs is None or faces.textures_idx is None:
        raise ValueError(f"{mesh_path} has no UV data")

    verts_uv = aux.verts_uvs.cpu().numpy().astype(np.float32)
    faces_uv = faces.textures_idx.cpu().numpy().astype(np.int64)
    return verts_uv, faces_uv

def backproject_single_view(
    image_np,
    pix_to_face,
    bary_coords,
    verts_uv,
    faces_uv,
    tex_h,
    tex_w,
    pixel_weight=None,
):
    """
    image_np: [H, W, 3] float32 in [0,1]
    pix_to_face: [H, W] int64
    bary_coords: [H, W, 3] float32
    pixel_weight: [H, W] float32 in [0,1], optional
    """
    H, W = pix_to_face.shape
    valid = pix_to_face >= 0
    if valid.sum() == 0:
        return None, None

    ys, xs = np.where(valid)
    face_ids = pix_to_face[valid].astype(np.int64)
    bary = bary_coords[valid].astype(np.float32)

    tri_uv_idx = faces_uv[face_ids]  # [N, 3]
    uv0 = verts_uv[tri_uv_idx[:, 0]]
    uv1 = verts_uv[tri_uv_idx[:, 1]]
    uv2 = verts_uv[tri_uv_idx[:, 2]]

    uv = (
        bary[:, 0:1] * uv0
        + bary[:, 1:2] * uv1
        + bary[:, 2:3] * uv2
    )

    u = np.clip(np.round(uv[:, 0] * (tex_w - 1)).astype(np.int64), 0, tex_w - 1)
    v = np.clip(np.round((1.0 - uv[:, 1]) * (tex_h - 1)).astype(np.int64), 0, tex_h - 1)

    if pixel_weight is None:
        w = np.ones((len(u),), dtype=np.float32)
    else:
        w = pixel_weight[ys, xs].astype(np.float32)

    tex_acc = np.zeros((tex_h * tex_w, 3), dtype=np.float32)
    w_acc = np.zeros((tex_h * tex_w,), dtype=np.float32)

    flat_idx = v * tex_w + u
    src = image_np[ys, xs].astype(np.float32)

    np.add.at(tex_acc, flat_idx, src * w[:, None])
    np.add.at(w_acc, flat_idx, w)

    tex_acc = tex_acc.reshape(tex_h, tex_w, 3)
    w_acc = w_acc.reshape(tex_h, tex_w)

    return tex_acc, w_acc

def reconstruct_texture_for_mesh(
    mesh_path: Path,
    corrupted_texture_path: Path,
    inpainted_paths,
    face_dir: Path,
    bary_dir: Path,
    mask_dir: Path,
    output_path: Path,
):
    base = np.array(Image.open(corrupted_texture_path).convert("RGB"), dtype=np.float32) / 255.0
    tex_h, tex_w = base.shape[:2]

    verts_uv, faces_uv = load_mesh_uv(mesh_path)

    tex_acc_total = np.zeros_like(base, dtype=np.float32)
    w_acc_total = np.zeros((tex_h, tex_w), dtype=np.float32)

    for img_path in sorted(inpainted_paths):
        stem = img_path.stem.replace("_inpainted", "")
        face_path = face_dir / f"{stem}.pt"
        bary_path = bary_dir / f"{stem}.pt"
        mask_path = mask_dir / f"{stem}.png"

        if not face_path.exists() or not bary_path.exists():
            continue

        image_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0

        pix_to_face = torch.load(face_path, map_location="cpu")
        if pix_to_face.ndim == 3 and pix_to_face.shape[-1] == 1:
            pix_to_face = pix_to_face.squeeze(-1)
        pix_to_face = pix_to_face.numpy().astype(np.int64)

        bary_coords = torch.load(bary_path, map_location="cpu")
        if bary_coords.ndim == 4 and bary_coords.shape[2] == 1:
            bary_coords = bary_coords.squeeze(2)
        bary_coords = bary_coords.numpy().astype(np.float32)

        pixel_weight = None
        if mask_path.exists():
            m = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
            pixel_weight = m

        tex_acc, w_acc = backproject_single_view(
            image_np=image_np,
            pix_to_face=pix_to_face,
            bary_coords=bary_coords,
            verts_uv=verts_uv,
            faces_uv=faces_uv,
            tex_h=tex_h,
            tex_w=tex_w,
            pixel_weight=pixel_weight,
        )

        if tex_acc is None:
            continue

        tex_acc_total += tex_acc
        w_acc_total += w_acc

    out = base.copy()
    valid = w_acc_total > 0
    out[valid] = tex_acc_total[valid] / w_acc_total[valid, None]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_rgb(out, output_path)
    return output_path

def backprojection_main():
    cfg = load_config()

    mesh_dir = Path(cfg["mesh_dir"])
    corrupted_dir = Path(cfg["corrupted_dir"])
    texture_images_dir = corrupted_dir / "images"
    inpainted_dir = Path(cfg["inpainted_dir"])
    render_dir = Path(cfg["render_dir"])
    face_dir = render_dir / "face_idx"
    bary_dir = render_dir / "barycentric"
    mask_dir = render_dir / "masks"

    out_dir = Path(cfg.get("reconstructed_texture_dir", "data/outputs/reconstructed_textures"))
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = group_inpainted_by_mesh(inpainted_dir)

    for mesh_name, paths in grouped.items():
        mesh_path, corrupted_texture_path = resolve_assets_for_mesh(
            mesh_name,
			mesh_dir,
			texture_images_dir,
		)		

        if mesh_path is None or corrupted_texture_path is None:
            print(f"missing mesh or corrupted texture for {mesh_name}, skipping")
            continue

        out_path = out_dir / f"{mesh_name}_reconstructed.png"

        reconstruct_texture_for_mesh(
            mesh_path=mesh_path,
            corrupted_texture_path=corrupted_texture_path,
            inpainted_paths=paths,
            face_dir=face_dir,
            bary_dir=bary_dir,
            mask_dir=mask_dir,
            output_path=out_path,
        )


if __name__ == "__main__":
    backprojection_main()