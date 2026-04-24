from pathlib import Path
from PIL import Image
import torch
import numpy as np
import yaml
import cv2
from pytorch3d.io import load_obj
from scipy.ndimage import distance_transform_edt

from src.utils.io_utils import group_inpainted_by_mesh, resolve_assets_for_mesh


def load_config(path="configs/texture_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_rgb(arr, path: Path):
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)

def load_mesh_uv_and_geometry(mesh_path: Path):
    verts, faces, aux = load_obj(str(mesh_path), load_textures=False)

    if aux.verts_uvs is None or faces.textures_idx is None:
        raise ValueError(f"{mesh_path} has no UV data")

    verts_uv = aux.verts_uvs.cpu().numpy().astype(np.float32)
    faces_uv = faces.textures_idx.cpu().numpy().astype(np.int64)

    verts_xyz = verts.cpu().numpy().astype(np.float32)
    faces_xyz = faces.verts_idx.cpu().numpy().astype(np.int64)

    tri = verts_xyz[faces_xyz]  # [F, 3, 3]
    face_centroids = tri.mean(axis=1)

    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    face_normals /= (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)

    return verts_uv, faces_uv, face_centroids, face_normals

def camera_center_from_RT(R, T):
    R = R[0].cpu().numpy()
    T = T[0].cpu().numpy()
    return -R.T @ T

def calculate_face_view_weights(face_centroids, face_normals, R, T, power=2.0):
    cam_center = camera_center_from_RT(R, T)  # [3]

    view_vec = cam_center[None, :] - face_centroids  # [F, 3]
    view_vec /= (np.linalg.norm(view_vec, axis=1, keepdims=True) + 1e-8)

    w = np.sum(face_normals * view_vec, axis=1)
    # w = np.clip(w, 0.0, 1.0)
    # w = 0.05 + 0.95 * (w ** power)
    w = np.clip(w, 0.0, 1.0) ** power

    return w.astype(np.float32)

def pixel_confidence_from_mask(mask_u8):
    m = mask_u8 > 127
    conf = np.ones_like(mask_u8, dtype=np.float32)

    if not m.any():
        return conf

    dist = distance_transform_edt(m).astype(np.float32)
    if dist.max() > 1e-8:
        inside = dist / dist.max()
        conf[m] = 0.35 + 0.65 * (1 - inside[m])
        # conf[m] = 0.1 + 0.9 * inside[m]
    else:
        conf[m] = 0.35

    return conf

# antialiasing and hole filling for small gaps in UV space
def fill_small_uv_holes(out, uv_mask, valid, max_radius=2):
    holes = uv_mask & (~valid)
    if not np.any(holes):
        return out

    # nearest valid pixel for every location
    dist = distance_transform_edt(~valid)
    _, idx = distance_transform_edt(~valid, return_indices=True)

    fill = holes & (dist <= max_radius)
    if np.any(fill):
        out[fill] = out[idx[0][fill], idx[1][fill]]

    return out

def repair_local_outliers(out, uv_mask, valid, dark_thresh=0.10, diff_thresh=0.18):
    repaired = out.copy()

    img8 = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
    med8 = cv2.medianBlur(img8, 3)
    med = med8.astype(np.float32) / 255.0

    gray = out.mean(axis=-1)
    diff = np.abs(out - med).mean(axis=-1)

    bad = uv_mask & valid & ((gray < dark_thresh) | (diff > diff_thresh))

    if np.any(bad):
        repaired[bad] = med[bad]

    return repaired

def backproject_single_view(
    image_np,
    pix_to_face,
    bary_coords,
    verts_uv,
    faces_uv,
    tex_h,
    tex_w,
    face_weights,
    pixel_weight=None,
):
    valid = pix_to_face >= 0
    if valid.sum() == 0:
        return None, None

    ys, xs = np.where(valid)
    face_ids = pix_to_face[valid].astype(np.int64)
    bary = bary_coords[valid].astype(np.float32)

    tri_uv_idx = faces_uv[face_ids]
    uv = (
        bary[:, 0:1] * verts_uv[tri_uv_idx[:, 0]] +
        bary[:, 1:2] * verts_uv[tri_uv_idx[:, 1]] +
        bary[:, 2:3] * verts_uv[tri_uv_idx[:, 2]]
    )

    # u = np.clip(np.round(uv[:, 0] * (tex_w - 1)).astype(np.int64), 0, tex_w - 1)
    # v = np.clip(np.round((1.0 - uv[:, 1]) * (tex_h - 1)).astype(np.int64), 0, tex_h - 1)
    # flat_idx = v * tex_w + u

    u = uv[:, 0] * (tex_w - 1)
    v = (1.0 - uv[:, 1]) * (tex_h - 1)

    u0 = np.clip(np.floor(u).astype(np.int64), 0, tex_w - 1)
    v0 = np.clip(np.floor(v).astype(np.int64), 0, tex_h - 1)
    u1 = np.clip(u0 + 1, 0, tex_w - 1)
    v1 = np.clip(v0 + 1, 0, tex_h - 1)

    du = np.clip(u - u0, 0.0, 1.0)
    dv = np.clip(v - v0, 0.0, 1.0)

    w00 = (1 - du) * (1 - dv)
    w10 = du * (1 - dv)
    w01 = (1 - du) * dv
    w11 = du * dv

    idx00 = v0 * tex_w + u0
    idx10 = v0 * tex_w + u1
    idx01 = v1 * tex_w + u0
    idx11 = v1 * tex_w + u1

    src = image_np[ys, xs].astype(np.float32)

    w = face_weights[face_ids].astype(np.float32)
    if pixel_weight is not None:
        w = w * pixel_weight[ys, xs].astype(np.float32)

    tex_acc = np.zeros((tex_h * tex_w, 3), dtype=np.float32)
    w_acc = np.zeros((tex_h * tex_w,), dtype=np.float32)

    # for c in range(3):
    #     tex_acc[:, c] = np.bincount(flat_idx, weights=src[:, c] * w, minlength=tex_h * tex_w)

    # w_acc = np.bincount(flat_idx, weights=w, minlength=tex_h * tex_w)

    minlength = tex_h * tex_w

    for c in range(3):
        tex_acc[:, c] += np.bincount(idx00, weights=src[:, c] * w * w00, minlength=minlength)
        tex_acc[:, c] += np.bincount(idx10, weights=src[:, c] * w * w10, minlength=minlength)
        tex_acc[:, c] += np.bincount(idx01, weights=src[:, c] * w * w01, minlength=minlength)
        tex_acc[:, c] += np.bincount(idx11, weights=src[:, c] * w * w11, minlength=minlength)

    w_acc += np.bincount(idx00, weights=w * w00, minlength=minlength)
    w_acc += np.bincount(idx10, weights=w * w10, minlength=minlength)
    w_acc += np.bincount(idx01, weights=w * w01, minlength=minlength)
    w_acc += np.bincount(idx11, weights=w * w11, minlength=minlength)

    tex_acc = tex_acc.reshape(tex_h, tex_w, 3)
    w_acc = w_acc.reshape(tex_h, tex_w)

    return tex_acc, w_acc

def reconstruct_texture_for_mesh(
    mesh_path: Path,
    corrupted_texture_path: Path,
    inpainted_paths,
    face_dir: Path,
    bary_dir: Path,
    cam_dir: Path,
    mask_dir: Path,
    output_path: Path,
):
    with Image.open(corrupted_texture_path) as im:
        base = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
    
    tex_h, tex_w = base.shape[:2]
    verts_uv, faces_uv, face_centroids, face_normals = load_mesh_uv_and_geometry(mesh_path)

    uv_mask = np.zeros((tex_h, tex_w), dtype=np.uint8)
    u_coords = np.clip(np.round(verts_uv[:, 0] * (tex_w - 1)), 0, tex_w - 1).astype(np.int32)
    v_coords = np.clip(np.round((1.0 - verts_uv[:, 1]) * (tex_h - 1)), 0, tex_h - 1).astype(np.int32)
    pts = np.stack([u_coords, v_coords], axis=1)

    for face in faces_uv:
        poly = pts[face]
        cv2.fillConvexPoly(uv_mask, poly, 1)

    uv_mask = uv_mask.astype(bool)

    tex_acc_total = np.zeros_like(base, dtype=np.float32)
    w_acc_total = np.zeros((tex_h, tex_w), dtype=np.float32)

    for img_path in sorted(inpainted_paths):
        stem = img_path.stem.replace("_inpainted", "")
        face_path = face_dir / f"{stem}.pt"
        bary_path = bary_dir / f"{stem}.pt"
        cam_path = cam_dir / f"{stem}.pt"
        mask_path = mask_dir / f"{stem}.png"

        if not (face_path.exists() and bary_path.exists() and cam_path.exists()):
            continue

        with Image.open(img_path) as im:
            image_np = np.array(im.convert("RGB"), dtype=np.float32) / 255.0

        pix_to_face = torch.load(face_path, map_location="cpu")
        if pix_to_face.ndim == 3 and pix_to_face.shape[-1] == 1:
            pix_to_face = pix_to_face.squeeze(-1)
        pix_to_face = pix_to_face.numpy().astype(np.int64)

        bary_coords = torch.load(bary_path, map_location="cpu")
        if bary_coords.ndim == 4 and bary_coords.shape[2] == 1:
            bary_coords = bary_coords.squeeze(2)
        bary_coords = bary_coords.numpy().astype(np.float32)

        cam_data = torch.load(cam_path, map_location="cpu")
        face_weights = calculate_face_view_weights(
            face_centroids=face_centroids,
            face_normals=face_normals,
            R=cam_data["R"],
            T=cam_data["T"],
            power=2.0,
        )

        pixel_weight = None
        if mask_path.exists():
            with Image.open(mask_path) as im:
                mask_u8 = np.array(im.convert("L"), dtype=np.uint8)
            pixel_weight = pixel_confidence_from_mask(mask_u8)

        tex_acc, w_acc = backproject_single_view(
            image_np=image_np,
            pix_to_face=pix_to_face,
            bary_coords=bary_coords,
            verts_uv=verts_uv,
            faces_uv=faces_uv,
            tex_h=tex_h,
            tex_w=tex_w,
            face_weights=face_weights,
            pixel_weight=pixel_weight,
        )

        if tex_acc is not None:
            tex_acc_total += tex_acc
            w_acc_total += w_acc

    out = base.copy()
    valid = w_acc_total > 1e-6
    out[valid] = tex_acc_total[valid] / (w_acc_total[valid, None] + 1e-8)

    out = fill_small_uv_holes(out, uv_mask=uv_mask, valid=valid, max_radius=1)

    new_valid = valid | (np.any(out != base, axis=-1))
    holes = (uv_mask & (~new_valid)).astype(np.uint8) * 255
    if np.any(holes):
        out_uint8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        out_refined = cv2.inpaint(out_uint8, holes, 3, cv2.INPAINT_TELEA)
        out = out_refined.astype(np.float32) / 255.0

    out = repair_local_outliers(out, uv_mask=uv_mask, valid=new_valid)

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
    cam_dir = render_dir / "cameras"
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
            cam_dir=cam_dir,
            mask_dir=mask_dir,
            output_path=out_path,
        )


if __name__ == "__main__":
    backprojection_main()