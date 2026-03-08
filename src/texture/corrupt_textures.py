import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import yaml


def load_config(path="configs/texture_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def generate_random_mask(width, height, num_shapes, thickness_min=10, thickness_max=40):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for _ in range(num_shapes):
        shape_type = random.choice(["rectangle", "ellipse", "line"])

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

def corrupt_single_texture(input_path, corrupted_path, mask_path, num_shapes, thickness_min=10, thickness_max=40):
    texture = Image.open(input_path).convert("RGB")
    w, h = texture.size

    mask = generate_random_mask(w, h, num_shapes, thickness_min, thickness_max)
    corrupted = apply_corruption(texture, mask)

    corrupted.save(corrupted_path)
    mask.save(mask_path)

def corrupt_main():
    cfg = load_config()

    input_dir = Path(cfg["resized_dir"])
    output_dir = Path(cfg["corrupted_dir"])

    corrupted_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    corrupted_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    num_shapes = cfg["corruption"]["num_shapes"]
    thickness_min = cfg["corruption"].get("line_thickness_min", 10)
    thickness_max = cfg["corruption"].get("line_thickness_max", 40)

    extensions = [ext.lower() for ext in cfg["extensions"]]

    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f"*.{ext}"))
        files.extend(input_dir.glob(f"*.{ext.upper()}"))

    for img_path in files:
        name = img_path.stem

        corrupted_path = corrupted_dir / f"{name}_corrupted.png"
        mask_path = mask_dir / f"{name}_mask.png"

        corrupt_single_texture(img_path, corrupted_path, mask_path, num_shapes, thickness_min, thickness_max)


if __name__ == "__main__":
    corrupt_main()