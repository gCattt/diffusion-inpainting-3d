from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from PIL import Image
try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None


def load_image(path) -> np.ndarray:
    """
    Load an RGB image as float32 array in [0, 1], shape [H, W, 3].
    """
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0

def load_mask(path) -> np.ndarray:
    """
    Load a grayscale mask as float32 array in [0, 1], shape [H, W].
    """
    mask = Image.open(path).convert("L")
    return np.asarray(mask, dtype=np.float32) / 255.0

def mse(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Mean squared error over RGB channels.
    """
    if img_a.shape != img_b.shape:
        raise ValueError(f"Shape mismatch: {img_a.shape} vs {img_b.shape}")
    return float(np.mean((img_a - img_b) ** 2))

def psnr(img_a: np.ndarray, img_b: np.ndarray, data_range: float = 1.0) -> float:
    """
    Peak signal-to-noise ratio in dB.
    """
    m = mse(img_a, img_b)
    if m == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(m))

def ssim(img_a: np.ndarray, img_b: np.ndarray) -> Optional[float]:
    """
    Structural similarity index.
    Requires scikit-image. Returns None if unavailable.
    """
    if skimage_ssim is None:
        return None

    if img_a.shape != img_b.shape:
        raise ValueError(f"Shape mismatch: {img_a.shape} vs {img_b.shape}")

    value = skimage_ssim(
        img_a,
        img_b,
        channel_axis=-1,
        data_range=1.0,
    )
    return float(value)

def masked_mse(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """
    MSE computed only on pixels where mask > 0.5.
    mask shape: [H, W]
    images shape: [H, W, 3]
    """
    if ref.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {pred.shape}")
    if mask.shape != ref.shape[:2]:
        raise ValueError(f"Mask shape mismatch: {mask.shape} vs {ref.shape[:2]}")

    m = (mask > 0.5).astype(np.float32)
    denom = float(m.sum() * ref.shape[-1])
    if denom == 0:
        return 0.0

    diff2 = (ref - pred) ** 2
    return float((diff2 * m[..., None]).sum() / denom)

def masked_psnr(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> float:
    """
    PSNR computed only on masked pixels.
    """
    m = masked_mse(ref, pred, mask)
    if m == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(m))

def masked_ssim(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """
    Optional masked SSIM.
    Simple implementation: zeroes out unmasked pixels before computing SSIM.
    This is weaker than a true masked-SSIM, but keeps the evaluation self-contained.
    Returns None if scikit-image is unavailable.
    """
    if skimage_ssim is None:
        return None

    if ref.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {pred.shape}")
    if mask.shape != ref.shape[:2]:
        raise ValueError(f"Mask shape mismatch: {mask.shape} vs {ref.shape[:2]}")

    m = (mask > 0.5).astype(np.float32)[..., None]
    ref_m = ref * m
    pred_m = pred * m

    value = skimage_ssim(
        ref_m,
        pred_m,
        channel_axis=-1,
        data_range=1.0,
    )
    return float(value)

def compute_metrics(reference: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Global image metrics.
    """
    out = {
        "mse": mse(reference, predicted),
        "psnr": psnr(reference, predicted),
    }

    ssim_value = ssim(reference, predicted)
    if ssim_value is not None:
        out["ssim"] = ssim_value

    return out

def compute_masked_metrics(reference: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Metrics restricted to the masked region.
    """
    out = {
        "masked_mse": masked_mse(reference, predicted, mask),
        "masked_psnr": masked_psnr(reference, predicted, mask),
    }

    ssim_value = masked_ssim(reference, predicted, mask)
    if ssim_value is not None:
        out["masked_ssim"] = ssim_value

    return out