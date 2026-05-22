from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw

from src.utils.config_utils import load_yaml_config

STAGE_SUFFIXES = (
    "_corrupted",
    "_inpainted",
    "_final",
    "_reconstructed",
)

def strip_stage_suffix(stem: str) -> str:
    changed = True
    while changed:
        changed = False
        for suf in STAGE_SUFFIXES:
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                changed = True
    return stem

def extract_key(path: Path) -> str:
    return strip_stage_suffix(path.stem)

def build_index(paths):
    idx = {}
    for p in paths:
        key = extract_key(p)
        idx[key] = p
    return idx

def collect_stage_paths(cfg):
    reference_dir = Path(cfg["reference_render_dir"])
    corrupted_dir = Path(cfg["corrupted_render_dir"])
    inpainted_dir = Path(cfg["inpainted_views_dir"])
    final_dir = Path(cfg["final_renders_dir"])

    reference_paths = sorted(reference_dir.rglob("*.png"))
    corrupted_paths = sorted(corrupted_dir.rglob("*.png"))
    inpainted_paths = sorted(inpainted_dir.rglob("*.png"))
    final_paths = sorted(final_dir.rglob("*_final.png"))

    return {
        "reference": build_index(reference_paths),
        "corrupted": build_index(corrupted_paths),
        "inpainted": build_index(inpainted_paths),
        "final": build_index(final_paths),
    }

def make_side_by_side(images, labels, target_height=None):
    assert len(images) == len(labels)

    if target_height is None:
        target_height = max(img.height for img in images)

    resized = []
    for img in images:
        if img.height != target_height:
            new_w = int(round(img.width * (target_height / img.height)))
            img = img.resize((new_w, target_height), Image.BILINEAR)
        resized.append(img)

    label_h = 28
    total_w = sum(img.width for img in resized)
    canvas = Image.new("RGB", (total_w, target_height + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for img, label in zip(resized, labels):
        canvas.paste(img, (x, label_h))
        draw.text((x + 6, 6), label, fill=(0, 0, 0))
        x += img.width

    return canvas

def compare_renders_main():
    cfg = load_yaml_config("configs/multiview_config.yaml")

    out_root = Path(cfg["evaluation_dir"])
    comp_dir = out_root / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    stages = collect_stage_paths(cfg)

    for key, final_path in stages["final"].items():
        reference_path = stages["reference"].get(key)
        if reference_path is None:
            continue

        mesh_name = final_path.parent.name
        corrupted_path = stages["corrupted"].get(key)
        inpainted_path = stages["inpainted"].get(key)

        ref_img = Image.open(reference_path).convert("RGB")
        final_check = Image.open(final_path).convert("RGB")

        if ref_img.size != final_check.size:
            continue

        # comparison image: reference | corrupted | inpainted | final
        pil_images = [ref_img]
        labels = ["reference"]

        if corrupted_path is not None:
            pil_images.append(Image.open(corrupted_path).convert("RGB"))
            labels.append("corrupted")

        if inpainted_path is not None:
            pil_images.append(Image.open(inpainted_path).convert("RGB"))
            labels.append("inpainted")

        pil_images.append(final_check)
        labels.append("final")

        comp = make_side_by_side(pil_images, labels)
        mesh_comp_dir = comp_dir / mesh_name
        mesh_comp_dir.mkdir(parents=True, exist_ok=True)
        comp.save(mesh_comp_dir / f"{key.replace('/', '_')}_comparison.png")


if __name__ == "__main__":
    compare_renders_main()