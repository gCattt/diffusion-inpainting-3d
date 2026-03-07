from pathlib import Path
from PIL import Image
import yaml


def load_config(path="configs/texture_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def resize_texture(src_path: Path, dst_path: Path, size: int, quality=95):
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        if img.width != size or img.height != size:
            img = img.resize((size, size), Image.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, quality=quality)

def resize_main():
    cfg = load_config()
    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    size = cfg["size"]
    exts = cfg["extensions"]
    quality = cfg.get("quality", 95)

    files = []
    for ext in exts:
        files.extend(input_dir.glob(f"*.{ext}"))

    for f in files:
        dst_file = output_dir / f.name
        resize_texture(f, dst_file, size, quality)


if __name__ == "__main__":
    resize_main()