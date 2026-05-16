from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import csv
import json
from PIL import Image, ImageDraw
import yaml
from .metrics import (
    load_image,
    load_mask,
    compute_metrics,
    compute_masked_metrics,
)

STAGE_SUFFIXES = (
    "_corrupted",
    "_inpainted",
    "_final",
    "_reconstructed",
)


def load_config(path="configs/multiview_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

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
    """
    Normalize the filename so stages can be matched.

    Examples:
      food_kiwi_01_8k_view00_inpainted -> food_kiwi_01_8k_view00
      food_kiwi_01_8k_view00_final     -> food_kiwi_01_8k_view00
      food_kiwi_01_8k_view00           -> food_kiwi_01_8k_view00
    """
    return strip_stage_suffix(path.stem)

def build_index(paths):
    idx = {}
    for p in paths:
        key = extract_key(p)
        idx[key] = p
    return idx

def collect_stage_paths(cfg):
    renders_root = Path(cfg.get("render_dir", "data/renders"))
    outputs_root = Path(cfg.get("outputs_dir", "data/outputs"))

    reference_dir = Path(cfg.get("reference_render_dir", renders_root / "reference_rgb"))
    corrupted_dir = renders_root / "rgb"
    mask_dir = renders_root / "masks"
    inpainted_dir = outputs_root / "inpainted_views"
    final_dir = outputs_root / "final_renders"

    reference_paths = sorted(reference_dir.rglob("*.png"))
    corrupted_paths = sorted(corrupted_dir.rglob("*.png"))
    mask_paths = sorted(mask_dir.rglob("*.png"))
    inpainted_paths = sorted(inpainted_dir.rglob("*.png"))
    final_paths = sorted(final_dir.rglob("*_final.png"))

    return {
        "reference": build_index(reference_paths),
        "corrupted": build_index(corrupted_paths),
        "mask": build_index(mask_paths),
        "inpainted": build_index(inpainted_paths),
        "final": build_index(final_paths),
    }

def make_side_by_side(images, labels, target_height=None):
    """
    images: list of PIL.Image
    labels: list[str]
    """
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

def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _merge_dicts(*dicts):
    out = {}
    for d in dicts:
        if d:
            out.update(d)
    return out

def compare_renders_main(cfg_path="configs/multiview_config.yaml"):
    cfg = load_config(cfg_path)

    out_root = Path(cfg.get("evaluation_dir", "data/outputs/evaluation"))
    comp_dir = out_root / "comparisons"
    metrics_dir = out_root / "metrics"
    comp_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    stages = collect_stage_paths(cfg)

    rows = defaultdict(list)  # Group by mesh_name
    summary = defaultdict(list)

    for key, final_path in stages["final"].items():
        reference_path = stages["reference"].get(key)
        if reference_path is None:
            continue

        mesh_name = final_path.parent.name
        corrupted_path = stages["corrupted"].get(key)
        inpainted_path = stages["inpainted"].get(key)
        mask_path = stages["mask"].get(key)

        ref = load_image(reference_path)
        final_img = load_image(final_path)

        if ref.shape != final_img.shape:
            continue

        final_metrics = _merge_dicts(
            compute_metrics(ref, final_img),
            compute_masked_metrics(ref, final_img, load_mask(mask_path)) if mask_path is not None else {},
        )
        final_metrics["key"] = key
        final_metrics["stage"] = "final"
        rows[mesh_name].append(final_metrics)
        summary["final"].append(final_metrics)

        if corrupted_path is not None:
            corrupted_img = load_image(corrupted_path)
            if corrupted_img.shape == ref.shape:
                corrupted_metrics = compute_metrics(ref, corrupted_img)
                corrupted_metrics["key"] = key
                corrupted_metrics["stage"] = "corrupted"
                rows[mesh_name].append(corrupted_metrics)
                summary["corrupted"].append(corrupted_metrics)

        if inpainted_path is not None:
            inpainted_img = load_image(inpainted_path)
            if inpainted_img.shape == ref.shape:
                inpainted_metrics = compute_metrics(ref, inpainted_img)
                inpainted_metrics["key"] = key
                inpainted_metrics["stage"] = "inpainted"
                rows[mesh_name].append(inpainted_metrics)
                summary["inpainted"].append(inpainted_metrics)

        # Comparison image: reference | corrupted | inpainted | final
        pil_images = [Image.open(reference_path).convert("RGB")]
        labels = ["reference"]

        if corrupted_path is not None:
            pil_images.append(Image.open(corrupted_path).convert("RGB"))
            labels.append("corrupted")

        if inpainted_path is not None:
            pil_images.append(Image.open(inpainted_path).convert("RGB"))
            labels.append("inpainted")

        pil_images.append(Image.open(final_path).convert("RGB"))
        labels.append("final")

        comp = make_side_by_side(pil_images, labels)
        mesh_comp_dir = comp_dir / mesh_name
        mesh_comp_dir.mkdir(parents=True, exist_ok=True)
        comp.save(mesh_comp_dir / f"{key.replace('/', '_')}_comparison.png")

    # Write per-image metrics CSV for each mesh
    for mesh_name, mesh_rows in rows.items():
        mesh_metrics_dir = metrics_dir / mesh_name
        mesh_metrics_dir.mkdir(parents=True, exist_ok=True)
        write_csv(mesh_rows, mesh_metrics_dir / "per_image_metrics.csv")

    summary_out = {}
    for stage, vals in summary.items():
        if not vals:
            continue

        summary_out[stage] = {
            "count": len(vals),
            "mse_mean": float(sum(v["mse"] for v in vals) / len(vals)) if all("mse" in v for v in vals) else None,
            "psnr_mean": float(sum(v["psnr"] for v in vals) / len(vals)) if all("psnr" in v for v in vals) else None,
        }

        if all("ssim" in v for v in vals):
            summary_out[stage]["ssim_mean"] = float(sum(v["ssim"] for v in vals) / len(vals))

        if all("masked_mse" in v for v in vals):
            summary_out[stage]["masked_mse_mean"] = float(sum(v["masked_mse"] for v in vals) / len(vals))
        if all("masked_psnr" in v for v in vals):
            summary_out[stage]["masked_psnr_mean"] = float(sum(v["masked_psnr"] for v in vals) / len(vals))
        if all("masked_ssim" in v for v in vals):
            summary_out[stage]["masked_ssim_mean"] = float(sum(v["masked_ssim"] for v in vals) / len(vals))

    with open(metrics_dir / "summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    print(f"Saved comparisons to: {comp_dir}")
    print(f"Saved metrics to: {metrics_dir}")
    return summary_out


if __name__ == "__main__":
    compare_renders_main()