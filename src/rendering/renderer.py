from pathlib import Path
from PIL import Image
import yaml
import torch
import json
import imageio
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesUV,
    FoVPerspectiveCameras
)
import torchvision.transforms as T
from torchvision.utils import save_image
from .camera_utils import create_cameras


def load_config(path="configs/multiview_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_mesh(mesh_path, texture_path, device):
    verts, faces, aux = load_obj(mesh_path)

    texture = Image.open(texture_path).convert("RGB")
    texture_tensor = torch.from_numpy(
        np.array(texture)
    ).float() / 255.0

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

def create_renderer(cfg, device):
    raster_settings = RasterizationSettings(
        image_size=cfg["image_size"],
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=None,
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=None,
            lights=lights,
        ),
    )

    return renderer

def render_views(mesh_path, texture_path, renderer, cameras, rgb_dir, depth_dir, face_dir, bary_dir, cam_dir, cfg, device):
    with torch.no_grad():
        mesh = load_mesh(mesh_path, texture_path, device)
        # normalize mesh scale
        scale = 1.0 / mesh.verts_packed().abs().max().item()
        mesh.scale_verts_(scale)

        mesh_name = mesh_path.stem

        for i in range(cfg["num_views"]):
            cam = FoVPerspectiveCameras(
                device=device,
                R=cameras.R[i:i+1],
                T=cameras.T[i:i+1],
            )

            images = renderer(mesh, cameras=cam)
            rgb = images[0, ..., :3]

            fragments = renderer.rasterizer(mesh, cameras=cam)
            pix_to_face = fragments.pix_to_face[0]
            bary_coords = fragments.bary_coords[0]
            depth = fragments.zbuf[0, ..., 0]

            rgb_path = rgb_dir / f"{mesh_name}_view{i:02d}.png"
            depth_path = depth_dir / f"{mesh_name}_view{i:02d}.pt"
            face_path = face_dir / f"{mesh_name}_view{i:02d}.pt"
            bary_path = bary_dir / f"{mesh_name}_view{i:02d}.pt"
            cam_path = cam_dir / f"{mesh_name}_view{i:02d}.pt"

            save_image(rgb.permute(2,0,1), rgb_path)
            torch.save(depth.cpu(), depth_path)
            torch.save(pix_to_face.cpu(), face_path)
            torch.save(bary_coords.cpu(), bary_path)
            torch.save(
                {
                    "R": cam.R.cpu(),
                    "T": cam.T.cpu(),
                },
                cam_path,
            )

            del images, fragments, depth
            torch.cuda.empty_cache()

    del mesh

def find_mesh_for_texture(texture_path, mesh_dir):
    name = texture_path.stem
    # remove diffusion-specific suffix
    base = name.replace("_diff", "").replace("_corrupted", "")

    candidates = list(mesh_dir.glob(f"{base}*.obj"))
    if not candidates:
        return None

    return candidates[0]

def renderer_main():
    cfg = load_config()

    mesh_dir = Path(cfg["mesh_dir"])
    texture_dir = Path(cfg["texture_dir"])

    rgb_dir = Path(cfg["render_dir"]) / "rgb"
    depth_dir = Path(cfg["render_dir"]) / "depth"
    face_dir = Path(cfg["render_dir"]) / "face_idx"
    bary_dir = Path(cfg["render_dir"]) / "barycentric"
    cam_dir = Path(cfg["render_dir"]) / "cameras"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)
    bary_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cameras = create_cameras(cfg, device)
    renderer = create_renderer(cfg, device)

    metadata = []

    for texture_path in texture_dir.glob("*_corrupted.png"):
        mesh_path = find_mesh_for_texture(texture_path, mesh_dir)
        if mesh_path is None:
            continue

        render_views(
            mesh_path,
            texture_path,
            renderer,
            cameras,
            rgb_dir,
            depth_dir,
            face_dir,
            bary_dir,
            cam_dir,
            cfg,
            device,
        )

        metadata.append(
            {
                "mesh": mesh_path.name,
                "texture": texture_path.name,
            }
        )

    metadata_path = Path(cfg["render_dir"]) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    renderer_main()