from pathlib import Path
from PIL import Image
import torch
import numpy as np
import cv2
from pytorch3d.io import load_obj

from src.utils.io_utils import group_inpainted_by_mesh, resolve_assets_for_mesh
from src.utils.config_utils import load_yaml_config


def save_rgb(arr, path: Path):
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255).round().astype(np.uint8)).save(path)

def load_mesh_uv_and_geometry(mesh_path: Path):
    verts, faces, aux = load_obj(str(mesh_path), load_textures=False)

    if aux.verts_uvs is None or faces.textures_idx is None:
        raise ValueError(f"{mesh_path} has no UV data")

    verts_uv = aux.verts_uvs.cpu().numpy().astype(np.float32)
    faces_uv = faces.textures_idx.cpu().numpy().astype(np.int64)

    verts_xyz = verts.cpu().numpy().astype(np.float32)
    faces_xyz = faces.verts_idx.cpu().numpy().astype(np.int64)

    tri = verts_xyz[faces_xyz]
    face_centroids = tri.mean(axis=1)

    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    face_normals /= (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)

    return verts_uv, faces_uv, face_centroids, face_normals

def camera_center_from_RT(R, T):
    R = R[0].cpu().numpy()
    T = T[0].cpu().numpy()
    return -R.T @ T

def calculate_face_view_weights(face_centroids, face_normals, R, T, power=2.0):
    cam_center = camera_center_from_RT(R, T)

    view_vec = cam_center[None, :] - face_centroids
    view_vec /= (np.linalg.norm(view_vec, axis=1, keepdims=True) + 1e-8)

    w = np.sum(face_normals * view_vec, axis=1)
    w = np.clip(w, 0.0, 1.0)

    w = 0.1 + 0.9 * (w ** power) 

    return w.astype(np.float32)

def backproject_single_view(
    image_np,
    view_mask,
    pix_to_face,
    bary_coords,
    verts_uv,
    faces_uv,
    tex_h,
    tex_w,
    face_weights,
):
    valid_mask = (view_mask > 0.05) & (pix_to_face >= 0)
    if valid_mask.sum() == 0:
        return None, None
    
    mask_u8 = (view_mask > 0.05).astype(np.uint8)
    if mask_u8.max() == 0: return None, None
    
    dist_map = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    if dist_map.max() > 1e-8:
        dist_map /= dist_map.max()

    edge_confidence = dist_map ** 0.7

    ys, xs = np.where(valid_mask)

    face_ids = pix_to_face[valid_mask].astype(np.int64)
    bary = bary_coords[valid_mask].astype(np.float32)

    tri_uv_idx = faces_uv[face_ids]
    uv = (
        bary[:, 0:1] * verts_uv[tri_uv_idx[:, 0]] +
        bary[:, 1:2] * verts_uv[tri_uv_idx[:, 1]] +
        bary[:, 2:3] * verts_uv[tri_uv_idx[:, 2]]
    )

    u = uv[:, 0] * (tex_w - 1)
    v = (1.0 - uv[:, 1]) * (tex_h - 1)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)

    u1 = np.clip(u0 + 1, 0, tex_w - 1)
    v1 = np.clip(v0 + 1, 0, tex_h - 1)

    u0 = np.clip(u0, 0, tex_w - 1)
    v0 = np.clip(v0, 0, tex_h - 1)

    du = (u - u0).astype(np.float32)
    dv = (v - v0).astype(np.float32)

    src = image_np[ys, xs].astype(np.float32)

    w = face_weights[face_ids] * edge_confidence[ys, xs]

    tex_acc = np.zeros((tex_h, tex_w, 3), dtype=np.float32)
    w_acc = np.zeros((tex_h, tex_w), dtype=np.float32)

    w00 = (1 - du) * (1 - dv) * w
    w10 = du * (1 - dv) * w
    w01 = (1 - du) * dv * w
    w11 = du * dv * w

    for c in range(3):
        np.add.at(tex_acc[..., c], (v0, u0), src[:, c] * w00)
        np.add.at(tex_acc[..., c], (v0, u1), src[:, c] * w10)
        np.add.at(tex_acc[..., c], (v1, u0), src[:, c] * w01)
        np.add.at(tex_acc[..., c], (v1, u1), src[:, c] * w11)

    np.add.at(w_acc, (v0, u0), w00)
    np.add.at(w_acc, (v0, u1), w10)
    np.add.at(w_acc, (v1, u0), w01)
    np.add.at(w_acc, (v1, u1), w11)

    uv_colors = np.zeros((tex_h, tex_w, 3), dtype=np.float32)
    uv_weights = np.zeros((tex_h, tex_w), dtype=np.float32)

    valid = w_acc > 1e-8

    uv_colors[valid] = tex_acc[valid] / w_acc[valid, None]
    uv_weights[valid] = w_acc[valid]

    return uv_colors, uv_weights

def reconstruct_texture_for_mesh(
    mesh_path: Path,
    corrupted_texture_path: Path,
    corruption_mask_path: Path,
    inpainted_paths,
    face_dir: Path,
    bary_dir: Path,
    cam_dir: Path,
    mask_dir: Path,
    output_path: Path,
):
    with Image.open(corrupted_texture_path) as im:
        base_uv = np.array(im.convert("RGB"), dtype=np.float32) / 255.0

    with Image.open(corruption_mask_path) as m:
        corruption_mask = (np.array(m.convert("L")) > 127)

    tex_h, tex_w = base_uv.shape[:2]
    verts_uv, faces_uv, face_centroids, face_normals = load_mesh_uv_and_geometry(mesh_path)

    uv_mask = np.zeros((tex_h, tex_w), dtype=np.uint8)
    u_coords = np.clip(np.round(verts_uv[:, 0] * (tex_w - 1)), 0, tex_w - 1).astype(np.int32)
    v_coords = np.clip(np.round((1.0 - verts_uv[:, 1]) * (tex_h - 1)), 0, tex_h - 1).astype(np.int32)
    pts = np.stack([u_coords, v_coords], axis=1)

    for face in faces_uv:
        poly = pts[face]
        cv2.fillConvexPoly(uv_mask, poly, 1)

    uv_mask = uv_mask.astype(bool)

    final_uv = base_uv.copy()
    best_weight_map = np.zeros((tex_h, tex_w), dtype=np.float32)

    for img_path in sorted(inpainted_paths):
        stem = img_path.stem.replace("_inpainted", "")
        mesh_name = img_path.parent.name

        try:
            face_file = face_dir / mesh_name / f"{stem}.pt"
            bary_file = bary_dir / mesh_name / f"{stem}.pt"
            cam_file = cam_dir / mesh_name / f"{stem}.pt"
            mask_file = mask_dir / mesh_name / f"{stem}.png"

            pix_to_face = torch.load(face_file, map_location="cpu").squeeze().numpy()
            bary_coords = torch.load(bary_file, map_location="cpu").squeeze().numpy()
            cam_data = torch.load(cam_file, map_location="cpu")

            with Image.open(mask_file) as m_img:
                view_mask = np.array(m_img.convert("L"), dtype=np.float32) / 255.0

            with Image.open(img_path) as im:
                image_np = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
        except Exception as e:
            print(f"Error loading backprojection data for {stem}: {e}")
            continue

        face_weights = calculate_face_view_weights(face_centroids, face_normals, cam_data["R"], cam_data["T"], power=2.0)

        uv_colors, uv_weights = backproject_single_view(
            image_np,
            view_mask,
            pix_to_face,
            bary_coords,
            verts_uv,
            faces_uv,
            tex_h,
            tex_w,
            face_weights
        )

        if uv_colors is None:
            continue

        uv_weights *= 1.0 / (np.percentile(uv_weights, 95) + 1e-8)

        improve_mask = uv_weights > (best_weight_map * 1.05)
        final_uv[improve_mask] = uv_colors[improve_mask]
        best_weight_map[improve_mask] = uv_weights[improve_mask]

    reconstructed_mask = best_weight_map > 1e-6

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_band = (
        cv2.dilate(reconstructed_mask.astype(np.uint8), kernel).astype(bool)
        ^ cv2.erode(reconstructed_mask.astype(np.uint8), kernel).astype(bool)
    ).astype(bool)

    padded_uv = final_uv.copy()
    for _ in range(4):
        dilated = cv2.dilate((padded_uv * 255).astype(np.uint8), kernel).astype(np.float32) / 255.0
        padded_uv[~uv_mask] = dilated[~uv_mask]

    blur_small = cv2.GaussianBlur(final_uv, (0,0), sigmaX=1.0)
    sharpen_mask = (reconstructed_mask & corruption_mask & uv_mask)[..., None]
    sharpened = np.clip(final_uv + 0.35 * (final_uv - blur_small), 0.0, 1.0 )
    final_uv = np.where(sharpen_mask, sharpened, final_uv)

    blurred = cv2.GaussianBlur(padded_uv, (0,0), sigmaX=2.0)
    final_uv[edge_band] = (
        0.4 * final_uv[edge_band] +
        0.6 * blurred[edge_band]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_rgb(final_uv, output_path)
    return output_path

def backprojection_main():
    cfg = load_yaml_config("configs/texture_config.yaml")

    mesh_dir = Path(cfg["mesh_dir"])
    corrupted_dir = Path(cfg["textures_corrupted_dir"])
    texture_images_dir = corrupted_dir / "images"
    texture_masks_dir = corrupted_dir / "masks"
    inpainted_dir = Path(cfg["inpainted_views_dir"])
    render_dir = Path(cfg["renders_dir"])
    face_dir = render_dir / "face_idx"
    bary_dir = render_dir / "barycentric"
    cam_dir = render_dir / "cameras"
    mask_dir = render_dir / "masks"

    out_dir = Path(cfg["reconstructed_textures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = group_inpainted_by_mesh(inpainted_dir)

    for mesh_name, paths in grouped.items():
        mesh_path, corrupted_texture_path, corruption_mask_path = resolve_assets_for_mesh(
            mesh_name,
			mesh_dir,
			texture_images_dir,
            texture_masks_dir,
		)		

        if mesh_path is None or corrupted_texture_path is None or corruption_mask_path is None:
            print(f"missing assets for {mesh_name}, skipping")
            continue

        out_path = out_dir / f"{mesh_name}_reconstructed.png"

        reconstruct_texture_for_mesh(
            mesh_path=mesh_path,
            corrupted_texture_path=corrupted_texture_path,
            corruption_mask_path=corruption_mask_path,
            inpainted_paths=paths,
            face_dir=face_dir,
            bary_dir=bary_dir,
            cam_dir=cam_dir,
            mask_dir=mask_dir,
            output_path=out_path,
        )


if __name__ == "__main__":
    backprojection_main()