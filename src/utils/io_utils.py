from pathlib import Path

def find_mesh_for_texture(texture_path, mesh_dir):
    name = texture_path.stem
    base = name.replace("_diff", "").replace("_corrupted", "")

    candidates = list(mesh_dir.glob(f"{base}*.obj"))
    if not candidates:
        return None

    return candidates[0]