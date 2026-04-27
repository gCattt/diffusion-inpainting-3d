import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from src.utils.config_utils import load_yaml_config


def generate_random_mask(width, height, num_shapes, thickness_min=4, thickness_max=30):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for _ in range(num_shapes):
        shape_type = random.choices(
            ["rectangle", "ellipse", "line"],
            weights=[0.5, 0.4, 0.1]
        )[0]

        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)
        x2 = random.randint(x1, width)
        y2 = random.randint(y1, height)

        if shape_type == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=255)
        elif shape_type == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=255)
        elif shape_type == "line":
            thickness = random.randint(thickness_min, thickness_max)
            draw.line([x1, y1, x2, y2], fill=255, width=thickness)

    return mask

def apply_corruption(texture, mask):
    tex_np = np.array(texture)
    mask_np = np.array(mask)

    corrupted = tex_np.copy()
    corrupted[mask_np == 255] = 0

    return Image.fromarray(corrupted)

def corrupt_single_texture(input_path, images_path, masks_path, num_shapes, thickness_min=4, thickness_max=30):
    texture = Image.open(input_path).convert("RGB")
    w, h = texture.size

    mask = generate_random_mask(w, h, num_shapes, thickness_min, thickness_max)
    corrupted = apply_corruption(texture, mask)

    corrupted.save(images_path)
    mask.save(masks_path)

def corrupt_main():
    cfg = load_yaml_config("configs/texture_config.yaml")
    corrupt_cfg = cfg["corruption"]

    input_dir = Path(cfg["textures_resized_dir"])
    output_dir = Path(cfg["textures_corrupted_dir"])

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    num_shapes = corrupt_cfg["num_shapes"]
    thickness_min = corrupt_cfg["line_thickness_min"]
    thickness_max = corrupt_cfg["line_thickness_max"]

    extensions = [
        ext.lower().lstrip(".")
        for ext in cfg.get("extensions", [])
    ]

    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f"*.{ext}"))
        files.extend(input_dir.glob(f"*.{ext.upper()}"))

    for img_path in sorted(files):
        name = img_path.stem

        images_path = images_dir / f"{name}_corrupted.png"
        masks_path = masks_dir / f"{name}_mask.png"

        corrupt_single_texture(img_path, images_path, masks_path, num_shapes, thickness_min, thickness_max)


if __name__ == "__main__":
    corrupt_main()