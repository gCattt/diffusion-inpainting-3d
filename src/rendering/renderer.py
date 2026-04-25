from pathlib import Path
from PIL import Image
import torch
import json
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardFlatShader,
    PointLights,
    AmbientLights,
    TexturesUV
)
from torchvision.utils import save_image

from .camera_utils import create_cameras
from src.utils.config_utils import load_yaml_config
from src.utils.io_utils import find_mesh_for_texture


def create_renderer(cfg, device, flat=False):
    raster_settings = RasterizationSettings(
        image_size=cfg["image_size"],
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    if flat:
        lights = AmbientLights(device=device, ambient_color=[[1.0, 1.0, 1.0]])
        shader = HardFlatShader(device=device, cameras=None, lights=lights)
    else:
        lights = PointLights(device=device, location=[[0.0, 5.0, 0.0]])
        shader = SoftPhongShader(device=device, cameras=None, lights=lights)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=None,
            raster_settings=raster_settings,
        ),
        shader=shader,
    )

    return renderer

def load_mesh(mesh_path, texture_path, device):
    verts, faces, aux = load_obj(str(mesh_path), load_textures=False)

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

def normalize_mesh(mesh):
    verts = mesh.verts_packed()

    center = (verts.min(0).values + verts.max(0).values) * 0.5
    mesh.offset_verts_(-center.expand_as(verts))

    radius = verts.norm(dim=1).max().item()
    scale = 1.0 / (radius + 1e-8)
    mesh.scale_verts_(scale)

    return mesh

def render_views(
    mesh_path,
    texture_path,
    renderer_shaded,
    renderer_flat,
    cameras,
    rgb_dir,
    rgb_inpaint_dir=None,
    depth_dir=None,
    face_dir=None,
    bary_dir=None,
    cam_dir=None,
    cfg=None,
    device=None,
    save_aux=True,
):
    mesh = load_mesh(mesh_path, texture_path, device)
    mesh = normalize_mesh(mesh)
    mesh_name = mesh_path.stem

    num_views = cfg["num_views"]
    with torch.no_grad():
        for i in range(num_views):
            cam = cameras[[i]]

            images_shaded = renderer_shaded(mesh, cameras=cam)
            rgb_shaded = images_shaded[0, ..., :3]
            save_image(rgb_shaded.permute(2, 0, 1), rgb_dir / f"{mesh_name}_view{i:02d}.png")

            if rgb_inpaint_dir is not None:
                images_flat = renderer_flat(mesh, cameras=cam)
                rgb_flat = images_flat[0, ..., :3]
                save_image(rgb_flat.permute(2, 0, 1), rgb_inpaint_dir / f"{mesh_name}_view{i:02d}.png")
                del images_flat

            if save_aux:
                fragments = renderer_shaded.rasterizer(mesh, cameras=cam)
                pix_to_face = fragments.pix_to_face[0]
                bary_coords = fragments.bary_coords[0]
                depth = fragments.zbuf[0, ..., 0]

                torch.save(depth.cpu(), depth_dir / f"{mesh_name}_view{i:02d}.pt")
                torch.save(pix_to_face.cpu(), face_dir / f"{mesh_name}_view{i:02d}.pt")
                torch.save(bary_coords.cpu(), bary_dir / f"{mesh_name}_view{i:02d}.pt")
                torch.save({"R": cam.R.cpu(), "T": cam.T.cpu()}, cam_dir / f"{mesh_name}_view{i:02d}.pt")

                del fragments, depth

            del images_shaded

    del mesh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def collect_texture_files(texture_dir: Path, extensions):
    files = []
    for ext in extensions:
        ext = ext.lower().lstrip(".")
        files.extend(texture_dir.glob(f"*.{ext}"))
        files.extend(texture_dir.glob(f"*.{ext.upper()}"))
    return sorted(set(files))

def render_dataset(
    mesh_dir,
    texture_dir,
    texture_glob,
    rgb_dir,
    rgb_inpaint_dir,
    depth_dir,
    face_dir,
    bary_dir,
    cam_dir,
    cfg,
    device,
    renderer_shaded,
    renderer_flat,
    save_aux,
    dataset_name,
    cameras,
):
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if rgb_inpaint_dir is not None:
        rgb_inpaint_dir.mkdir(parents=True, exist_ok=True)
    if save_aux:
        depth_dir.mkdir(parents=True, exist_ok=True)
        face_dir.mkdir(parents=True, exist_ok=True)
        bary_dir.mkdir(parents=True, exist_ok=True)
        cam_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    # texture_glob can be a string or a list of extensions
    if isinstance(texture_glob, (list, tuple)):
        texture_paths = collect_texture_files(texture_dir, texture_glob)
    else:
        texture_paths = sorted(texture_dir.glob(texture_glob))

    for texture_path in texture_paths:
        mesh_path = find_mesh_for_texture(texture_path, mesh_dir)
        if mesh_path is None:
            print(f"missing mesh for {texture_path.name}, skipping")
            continue

        render_views(
            mesh_path=mesh_path,
            texture_path=texture_path,
            renderer_shaded=renderer_shaded,
            renderer_flat=renderer_flat,
            cameras=cameras,
            rgb_dir=rgb_dir,
            rgb_inpaint_dir=rgb_inpaint_dir,
            depth_dir=depth_dir,
            face_dir=face_dir,
            bary_dir=bary_dir,
            cam_dir=cam_dir,
            cfg=cfg,
            device=device,
            save_aux=save_aux,
        )

        metadata.append(
            {
                "dataset": dataset_name,
                "mesh": mesh_path.name,
                "texture": texture_path.name,
            }
        )

    return metadata

def renderer_main():
    cfg = load_yaml_config("configs/multiview_config.yaml")

    mesh_dir = Path(cfg["mesh_dir"])
    corrupted_texture_dir = Path(cfg["texture_images_dir"])
    reference_texture_dir = Path(cfg["reference_texture_dir"])

    render_root = Path(cfg["render_dir"])
    rgb_dir = render_root / "rgb"
    rgb_inpaint_dir = render_root / "rgb_inpaint"
    reference_rgb_dir = render_root / "reference_rgb"
    depth_dir = render_root / "depth"
    face_dir = render_root / "face_idx"
    bary_dir = render_root / "barycentric"
    cam_dir = render_root / "cameras"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cameras = create_cameras(cfg, device)
    renderer_shaded = create_renderer(cfg, device, flat=False)
    renderer_flat = create_renderer(cfg, device, flat=True)

    metadata = []

    metadata += render_dataset(
        mesh_dir=mesh_dir,
        texture_dir=corrupted_texture_dir,
        texture_glob="*_corrupted.png",
        rgb_dir=rgb_dir,
        rgb_inpaint_dir=rgb_inpaint_dir,
        depth_dir=depth_dir,
        face_dir=face_dir,
        bary_dir=bary_dir,
        cam_dir=cam_dir,
        cfg=cfg,
        device=device,
        renderer_shaded=renderer_shaded,
        renderer_flat=renderer_flat,
        save_aux=True,
        dataset_name="corrupted",
        cameras=cameras,
    )

    metadata += render_dataset(
        mesh_dir=mesh_dir,
        texture_dir=reference_texture_dir,
        texture_glob=["jpg", "jpeg", "png"],
        rgb_dir=reference_rgb_dir,
        rgb_inpaint_dir=None,
        depth_dir=depth_dir,
        face_dir=face_dir,
        bary_dir=bary_dir,
        cam_dir=cam_dir,
        cfg=cfg,
        device=device,
        renderer_shaded=renderer_shaded,
        renderer_flat=renderer_flat,
        save_aux=False,
        dataset_name="reference",
        cameras=cameras,
    )

    metadata_path = render_root / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    renderer_main()