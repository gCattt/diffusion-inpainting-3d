from pathlib import Path
import torch
import numpy as np
from PIL import Image
from pytorch3d.io import load_obj

from src.utils.io_utils import find_mesh_for_texture
from src.utils.config_utils import load_yaml_config


def load_uv_mask(mask_path, device):
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask).to(device)
    return mask

def project_mask(
    uv_mask,
    pix_to_face,
    bary_coords,
    faces_uv,
    verts_uv,
    batch_size=100000,
):
    device = uv_mask.device
    H, W = pix_to_face.shape
    view_mask = torch.zeros((H, W), device=device)

    valid = pix_to_face >= 0
    num_valid = int(valid.sum().item())
    if num_valid == 0:
        return view_mask

    face_ids = pix_to_face[valid].long()
    bary = bary_coords[valid].view(-1, 3).float()

    if face_ids.max().item() >= faces_uv.shape[0] or face_ids.min().item() < 0:
        raise RuntimeError("face_ids out of range: check pix_to_face / faces_uv")

    faces = faces_uv[face_ids]

    tex_h, tex_w = uv_mask.shape
    start = 0
    while start < num_valid:
        end = min(start + batch_size, num_valid)
        faces_b = faces[start:end]
        bary_b = bary[start:end]

        uv0 = verts_uv[faces_b[:, 0]]
        uv1 = verts_uv[faces_b[:, 1]]
        uv2 = verts_uv[faces_b[:, 2]]

        uv = (bary_b.unsqueeze(-1) * torch.stack([uv0, uv1, uv2], dim=1)).sum(dim=1)

        u = (uv[:, 0] * (tex_w - 1)).long()
        v = ((1 - uv[:, 1]) * (tex_h - 1)).long()

        u = torch.clamp(u, 0, tex_w - 1)
        v = torch.clamp(v, 0, tex_h - 1)

        sampled = uv_mask[v, u]

        valid_indices = valid.nonzero(as_tuple=False)
        batch_indices = valid_indices[start:end]
        view_mask[batch_indices[:, 0], batch_indices[:, 1]] = sampled

        start = end

    return view_mask

def save_mask(mask, path):
    mask = (mask.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask).save(path)

def mask_projection_main():
    cfg = load_yaml_config("configs/multiview_config.yaml")
    
    mesh_dir = Path(cfg["mesh_dir"])
    texture_images_dir = Path(cfg["texture_images_dir"])
    texture_masks_dir = Path(cfg["texture_masks_dir"])
    render_dir = Path(cfg["render_dir"])

    face_dir = render_dir / "face_idx"
    bary_dir = render_dir / "barycentric"
    mask_out_dir = render_dir / "masks"

    mask_out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for texture_path in texture_images_dir.glob("*_corrupted.png"):
        mesh_path = find_mesh_for_texture(texture_path, mesh_dir)
        if mesh_path is None:
            continue

        mesh_name = mesh_path.stem

        # Create subdirectory for this mesh
        mesh_mask_out_dir = mask_out_dir / mesh_name
        mesh_mask_out_dir.mkdir(parents=True, exist_ok=True)

        verts, faces, aux = load_obj(mesh_path, load_textures=False)

        verts_uv = aux.verts_uvs.to(device)
        faces_uv = faces.textures_idx.to(device)

        base_name = texture_path.stem.replace("_corrupted", "_mask")
        mask_path = texture_masks_dir / f"{base_name}.png"

        if not mask_path.exists():    continue

        uv_mask = load_uv_mask(mask_path, device)

        face_dir_mesh = face_dir / mesh_name
        bary_dir_mesh = bary_dir / mesh_name

        if not face_dir_mesh.exists() or not bary_dir_mesh.exists():
            print(f"missing mesh-specific render data for {mesh_name}, skipping")
            continue

        face_files = sorted(face_dir_mesh.glob(f"{mesh_name}_view*.pt"))
        if not face_files:
            print(f"no view files in {face_dir_mesh}, skipping {mesh_name}")
            continue

        for face_file in face_files:
            view_id = face_file.stem.split("_")[-1]

            bary_file = bary_dir_mesh / f"{mesh_name}_{view_id}.pt"

            pix_to_face = torch.load(face_file)
            if pix_to_face.ndim == 3 and pix_to_face.shape[-1] == 1:
                pix_to_face = pix_to_face.squeeze(-1)
            pix_to_face = pix_to_face.to(device).long()

            bary_coords = torch.load(bary_file)
            if bary_coords.ndim == 4 and bary_coords.shape[2] == 1:
                bary_coords = bary_coords.squeeze(2)
            bary_coords = bary_coords.to(device).float()

            view_mask = project_mask(
                uv_mask,
                pix_to_face,
                bary_coords,
                faces_uv,
                verts_uv,
            )

            out_path = mesh_mask_out_dir / f"{mesh_name}_{view_id}.png"
            save_mask(view_mask, out_path)


if __name__ == "__main__":
    mask_projection_main()