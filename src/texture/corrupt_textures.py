import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from src.utils.config_utils import load_yaml_config


def generate_random_mask(width, height, num_shapes, thickness_min, thickness_max, min_shape_size, shape_max_ratio, line_max_ratio):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for _ in range(num_shapes):
        shape_type = random.choices(
            ["rectangle", "ellipse", "line"],
            weights=[0.3, 0.2, 0.5]
        )[0]

        max_w = max(min_shape_size, int(width * shape_max_ratio))
        max_h = max(min_shape_size, int(height * shape_max_ratio))

        w_shape = random.randint(min_shape_size, max_w)
        h_shape = random.randint(min_shape_size, max_h)

        x1 = random.randint(0, max(0, width - w_shape))
        y1 = random.randint(0, max(0, height - h_shape))
        x2 = x1 + w_shape
        y2 = y1 + h_shape

        if shape_type == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=255)
        elif shape_type == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=255)
        elif shape_type == "line":
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)

            max_len = int(min(width, height) * line_max_ratio)
            dx = random.randint(-max_len, max_len)
            dy = random.randint(-max_len, max_len)

            x2 = np.clip(x1 + dx, 0, width - 1)
            y2 = np.clip(y1 + dy, 0, height - 1)

            thickness = random.randint(thickness_min, thickness_max)

            draw.line([x1, y1, x2, y2], fill=255, width=thickness)

    return mask

def apply_corruption(texture, mask):
    tex_np = np.array(texture)
    mask_np = np.array(mask)

    corrupted = tex_np.copy()
    corrupted[mask_np == 255] = 0

    return Image.fromarray(corrupted)

def corrupt_single_texture(input_path, images_path, masks_path, num_shapes, thickness_min, thickness_max, min_shape_size, shape_max_ratio, line_max_ratio):
    texture = Image.open(input_path).convert("RGB")
    w, h = texture.size

    mask = generate_random_mask(w, h, num_shapes, thickness_min, thickness_max, min_shape_size, shape_max_ratio, line_max_ratio)
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

    extensions = [
        ext.lower().lstrip(".")
        for ext in cfg["extensions"]
    ]

    num_shapes = corrupt_cfg["num_shapes"]
    thickness_min = corrupt_cfg["line_thickness_min"]
    thickness_max = corrupt_cfg["line_thickness_max"]
    min_shape_size = corrupt_cfg["min_shape_size"]
    shape_max_ratio = corrupt_cfg["shape_max_ratio"]
    line_max_ratio = corrupt_cfg["line_max_ratio"]

    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f"*.{ext}"))
        files.extend(input_dir.glob(f"*.{ext.upper()}"))

    for img_path in sorted(files):
        name = img_path.stem

        images_path = images_dir / f"{name}_corrupted.png"
        masks_path = masks_dir / f"{name}_mask.png"

        try:
            corrupt_single_texture(img_path, images_path, masks_path, num_shapes, thickness_min, thickness_max, min_shape_size, shape_max_ratio, line_max_ratio)
        except Exception as e:
            print(f"Error corrupting {name}: {e}")
            continue


if __name__ == "__main__":
    corrupt_main()