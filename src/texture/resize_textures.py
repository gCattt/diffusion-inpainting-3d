from pathlib import Path
from PIL import Image

from src.utils.config_utils import load_yaml_config


def resize_texture(src_path: Path, dst_path: Path, size: int, quality=95):
    with Image.open(src_path) as img:
        img = img.convert("RGB")

        if img.width != size or img.height != size:
            img = img.resize((size, size), Image.LANCZOS)

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        img.save(dst_path, quality=quality)

def resize_main():
    cfg = load_yaml_config("configs/texture_config.yaml")
    resize_cfg = cfg["resize"]

    input_dir = Path(cfg["textures_original_dir"])
    resized_dir = Path(cfg["textures_resized_dir"])

    size = resize_cfg["size"]
    quality = resize_cfg.get("quality", 95)

    extensions = [
        ext.lower().lstrip(".")
        for ext in cfg.get("extensions", [])
    ]

    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f"*.{ext}"))
        files.extend(input_dir.glob(f"*.{ext.upper()}"))

    for f in sorted(files):
        dst_file = resized_dir / f"{f.stem}.png"
        resize_texture(f, dst_file, size, quality)


if __name__ == "__main__":
    resize_main()