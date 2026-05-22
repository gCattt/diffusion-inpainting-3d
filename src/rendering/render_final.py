from pathlib import Path
import torch
from torchvision.utils import save_image
import gc

from src.rendering.camera_utils import create_cameras
from src.rendering.renderer import create_renderer, load_mesh, normalize_mesh
from src.utils.config_utils import load_yaml_config
from src.utils.io_utils import find_mesh_for_texture


def render_final_main():
    cfg = load_yaml_config("configs/multiview_config.yaml")

    mesh_dir = Path(cfg["mesh_dir"])
    reconstructed_textures_dir = Path(cfg["reconstructed_textures_dir"])
    final_renders_dir = Path(cfg["final_renders_dir"])
    final_renders_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cameras = create_cameras(cfg, device)
    renderer_shaded = create_renderer(cfg, device, flat=False)

    reconstructed_textures = sorted(reconstructed_textures_dir.glob("*_reconstructed.png"))
    if not reconstructed_textures:
        print(f"No reconstructed textures found in {reconstructed_textures_dir}")
        return

    for texture_path in reconstructed_textures:
        mesh_path = find_mesh_for_texture(texture_path, mesh_dir)
        if mesh_path is None:
            print(f"Missing mesh for {texture_path.name}, skipping")
            continue

        try:
            mesh = load_mesh(mesh_path, texture_path, device)
            mesh = normalize_mesh(mesh)
            mesh_name = mesh_path.stem

            out_dir = final_renders_dir / mesh_name
            out_dir.mkdir(parents=True, exist_ok=True)

            num_views = len(cameras.R)
            with torch.no_grad():
                for i in range(num_views):
                    try:
                        cam = cameras[[i]]

                        images_shaded = renderer_shaded(mesh, cameras=cam)
                        rgb_shaded = images_shaded[0, ..., :3]
                        save_image(rgb_shaded.permute(2, 0, 1), out_dir / f"{mesh_name}_view{i:02d}_final.png")

                        del images_shaded, rgb_shaded, cam

                    except Exception as e:
                        print(f"Error rendering view {i} for {mesh_name}: {e}")
                        continue

            del mesh
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error processing texture {texture_path.name}: {e}")
            continue


if __name__ == "__main__":
    render_final_main()