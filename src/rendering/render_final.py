from pathlib import Path
from PIL import Image
import yaml
import torch
import numpy as np
from torchvision.utils import save_image
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesUV,
    FoVPerspectiveCameras,
)
from src.rendering.renderer import create_renderer
from src.rendering.camera_utils import create_cameras
from src.utils.io_utils import normalize_name


def load_config(path="configs/multiview_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def find_mesh_for_reconstructed_texture(texture_path: Path, mesh_dir: Path):
    target = normalize_name(texture_path.stem)

    candidates = []
    for p in mesh_dir.glob("*.obj"):
        if normalize_name(p.stem) == target:
            candidates.append(p)

    return candidates[0] if candidates else None

def load_mesh(mesh_path, texture_path, device):
    verts, faces, aux = load_obj(str(mesh_path))

    texture = Image.open(texture_path).convert("RGB")
    texture_tensor = torch.from_numpy(np.array(texture)).float() / 255.0
    texture_tensor = texture_tensor.unsqueeze(0).to(device)

    if aux.verts_uvs is None or faces.textures_idx is None:
        raise ValueError(f"Mesh {mesh_path.name} has no UV coordinates.")

    textures = TexturesUV(
        maps=texture_tensor,
        faces_uvs=faces.textures_idx[None].to(device),
        verts_uvs=aux.verts_uvs[None].to(device),
    )

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.verts_idx.to(device)],
        textures=textures,
    )

    return mesh

def render_final_main():
    cfg = load_config()

    mesh_dir = Path(cfg["mesh_dir"])
    reconstructed_dir = Path(cfg.get("reconstructed_texture_dir", "data/outputs/reconstructed_textures"))
    final_renders_dir = Path(cfg.get("final_renders_dir", "data/outputs/final_renders"))
    final_renders_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cameras = create_cameras(cfg, device)
    renderer_shaded = create_renderer(cfg, device, flat=False)

    reconstructed_textures = sorted(reconstructed_dir.glob("*_reconstructed.png"))
    if not reconstructed_textures:
        print(f"no reconstructed textures found in {reconstructed_dir}")
        return

    for texture_path in reconstructed_textures:
        mesh_path = find_mesh_for_reconstructed_texture(texture_path, mesh_dir)
        if mesh_path is None:
            print(f"missing mesh for {texture_path.name}, skipping")
            continue

        mesh = load_mesh(mesh_path, texture_path, device)

        # scale mesh exactly as in the intermediate renderer
        scale = 1.0 / mesh.verts_packed().abs().max().item()
        mesh.scale_verts_(scale)

        mesh_name = mesh_path.stem
        out_dir = final_renders_dir / mesh_name
        out_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for i in range(cfg["num_views"]):
                cam = FoVPerspectiveCameras(
                    device=device,
                    R=cameras.R[i:i+1],
                    T=cameras.T[i:i+1],
                )

                images = renderer_shaded(mesh, cameras=cam)
                rgb = images[0, ..., :3]

                out_path = out_dir / f"{mesh_name}_view{i:02d}_final.png"
                save_image(rgb.permute(2, 0, 1), out_path)

        del mesh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    render_final_main()