import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def generate_random_mask(width, height, num_shapes=6):
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
            thickness = random.randint(10, 40)
            draw.line([x1, y1, x2, y2], fill=255, width=thickness)

    return mask

def apply_corruption(texture, mask):
    tex_np = np.array(texture)
    mask_np = np.array(mask)

    corrupted = tex_np.copy()
    corrupted[mask_np == 255] = 0

    return Image.fromarray(corrupted)

def corrupt_single_texture(input_path, corrupted_path, mask_path):
    texture = Image.open(input_path).convert("RGB")
    w, h = texture.size

    mask = generate_random_mask(w, h)
    corrupted = apply_corruption(texture, mask)

    corrupted.save(corrupted_path)
    mask.save(mask_path)

def corrupt_main():
    input_dir = Path("data/textures/resized")
    output_dir = Path("data/textures/corrupted")

    corrupted_dir = output_dir / "images"
    mask_dir = output_dir / "masks"

    corrupted_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    extensions = [".png", ".jpg", ".jpeg"]

    for img_path in input_dir.iterdir():

        if img_path.suffix.lower() not in extensions:
            continue

        name = img_path.stem

        corrupted_path = corrupted_dir / f"{name}_corrupted.png"
        mask_path = mask_dir / f"{name}_mask.png"

        corrupt_single_texture(img_path, corrupted_path, mask_path)


if __name__ == "__main__":
    corrupt_main()