from pathlib import Path
import re


def normalize_name(name: str) -> str:
    parts = name.split("_")
    parts = [p for p in parts if p not in {"diff", "corrupted", "inpainted", "reconstructed", "final"}]
    parts = [p for p in parts if not re.fullmatch(r"view\d+", p)]
    return "_".join(parts)

def find_mesh(mesh_name: str, mesh_dir: Path):
    target = normalize_name(mesh_name)
    candidates = []
    for p in mesh_dir.glob("*.obj"):
        if normalize_name(p.stem) == target:
            candidates.append(p)
    return candidates[0] if candidates else None

def find_mesh_for_texture(texture_path, mesh_dir):
    target = normalize_name(texture_path.stem)
    candidates = []
    for p in mesh_dir.glob("*.obj"):
        if normalize_name(p.stem) == target:
            candidates.append(p)
    return candidates[0] if candidates else None

def find_corrupted_texture(mesh_name: str, texture_dir: Path):
    target = normalize_name(mesh_name)

    search_dirs = [texture_dir]
    if (texture_dir / "images").exists():
        search_dirs.insert(0, texture_dir / "images")

    for d in search_dirs:
        for p in d.glob("*corrupted.png"):
            if normalize_name(p.stem) == target:
                return p

    return None

def group_inpainted_by_mesh(inpainted_dir: Path):
    groups = {}
    for p in inpainted_dir.glob("*_inpainted.png"):
        base = normalize_name(p.stem)
        groups.setdefault(base, []).append(p)
    return groups

def resolve_assets_for_mesh(mesh_name: str, mesh_dir: Path, texture_dir: Path):
    mesh_path = find_mesh(mesh_name, mesh_dir)
    texture_path = find_corrupted_texture(mesh_name, texture_dir)
    return mesh_path, texture_path