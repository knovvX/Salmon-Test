"""
Image preprocessing functions for fish scale images
"""

import numpy as np
import cv2
from src.config import (
    ROBUST_NORM_PERCENTILES,
    DESPECKLE_MEDIAN_KERNEL,
    DESPECKLE_MORPH_RADIUS,
    FFT_RLO_FRAC,
    FFT_RHI_FRAC,
    FFT_SOFTNESS,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    LAPLACIAN_WEIGHT
)


def to_float01(a: np.ndarray) -> np.ndarray:
    """
    Normalize any bit-depth image to float32 [0,1]

    Parameters
    ----------
    a : np.ndarray
        Input image, supports uint8 (0-255), uint16 (0-65535) or any float format

    Returns
    -------
    np.ndarray
        Normalized float32 image with values in [0,1]
    """
    a = np.asarray(a)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    a = a.astype(np.float32)
    m = a.max()
    return a / m if m > 0 else a


def normalize_robust(x: np.ndarray, p1=1, p2=99) -> np.ndarray:
    """
    Percentile-based stretching to [0,1], robust against extreme values

    Parameters
    ----------
    x : np.ndarray
        Input single-channel image (float/uint)
    p1 : int or float, optional
        Lower percentile (e.g., 1 = 1st percentile). Default 1
    p2 : int or float, optional
        Upper percentile (e.g., 99 = 99th percentile). Default 99

    Returns
    -------
    np.ndarray
        Linearly stretched and clipped float32 image in [0,1]
    """
    lo, hi = np.percentile(x, [p1, p2])
    if hi <= lo:
        return np.clip(x, 0, 1)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1).astype(np.float32)


def despeckle(x: np.ndarray, med_k=5, open_r=2) -> np.ndarray:
    """
    Remove speckles: median filter + morphological opening
    Input/output are [0,1] float

    Parameters
    ----------
    x : np.ndarray
        Single-channel image, values in [0,1]
    med_k : int or None, optional
        Median filter kernel size (must be odd and >=3)
        If None or <3, skip median filtering. Default 5
    open_r : int or None, optional
        Morphological opening radius (pixels)
        If None or 0, skip opening. Default 2

    Returns
    -------
    np.ndarray
        Despeckled single-channel float32 image in [0,1]
    """
    x8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    if med_k and med_k >= 3:
        x8 = cv2.medianBlur(x8, med_k)
    if open_r and open_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1, 2 * open_r + 1))
        x8 = cv2.morphologyEx(x8, cv2.MORPH_OPEN, k)
    return (x8.astype(np.float32) / 255.0)


def fft_bandpass(a: np.ndarray, rlo_frac=0.015, rhi_frac=0.22, softness=6.0):
    """
    Frequency domain ring bandpass filter (smooth transition)
    Returns filtered image and mask for debugging/visualization

    Parameters
    ----------
    a : np.ndarray
        Single-channel image (float), shape HxW
    rlo_frac : float, optional
        Inner radius (normalized to half of shortest edge)
        Frequencies below this are suppressed. Default 0.015
    rhi_frac : float, optional
        Outer radius (normalized to half of shortest edge)
        Frequencies above this are suppressed. Default 0.22
    softness : float, optional
        Controls smoothness of rlo/rhi boundaries
        Larger = steeper transition. Default 6.0

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        band : Bandpass filtered and linearly stretched image [0,1] (float32)
        mask : Frequency domain mask (float32) for debugging
    """
    a = a.astype(np.float32)
    h, w = a.shape[:2]

    F = np.fft.fftshift(np.fft.fft2(a))
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (min(h, w) / 2.0)

    def smooth_step(x, edge, k):
        # Logistic transition; larger k = harder edge
        return 1.0 / (1.0 + np.exp(k * (edge - x)))

    lo = smooth_step(r, rlo_frac, softness)
    hi = 1.0 - smooth_step(r, rhi_frac, softness)
    mask = lo * hi

    Ff = F * mask
    out = np.fft.ifft2(np.fft.ifftshift(Ff)).real
    # Linear stretch to [0,1]
    mn, mx = out.min(), out.max()
    band = (out - mn) / (mx - mn + 1e-8)
    return band.astype(np.float32), mask.astype(np.float32)


def preprocess_img_pipeline(img: np.ndarray) -> np.ndarray:
    """
    Complete preprocessing pipeline: converts any input image to HxWx3 float32 [0,1]

    Pipeline:
      1) to_float01 -> ensure float [0,1]
      2) Convert to grayscale if color
      3) normalize_robust(p1, p2) -> percentile stretching
      4) despeckle(med_k, open_r) -> remove noise
      5) fft_bandpass(rlo_frac, rhi_frac, softness) -> preserve mid-frequency texture
      6) CLAHE(clipLimit, tileGridSize) -> local contrast enhancement
      7) Laplacian sharpening (enh = clahe - weight * lap)
      8) Duplicate to 3 channels for torchvision transforms

    Parameters
    ----------
    img : np.ndarray
        Input image, can be grayscale or color, int or float, any bit-depth
        Supports HxW or HxWxC

    Returns
    -------
    np.ndarray
        Processed HxWx3 float32 image in [0,1], ready for PIL RGB conversion
    """
    # 1) Load & convert to grayscale
    x = to_float01(img)
    if x.ndim == 3:
        # Convert to uint8 first for stable color space conversion
        x8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        # Assume RGB input (most S3 images are RGB)
        x = cv2.cvtColor(x8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # 2) Percentile stretching
    p1, p2 = ROBUST_NORM_PERCENTILES
    x = normalize_robust(x, p1, p2)

    # 3) Despeckle
    x = despeckle(x, med_k=DESPECKLE_MEDIAN_KERNEL, open_r=DESPECKLE_MORPH_RADIUS)

    # 4) Bandpass (preserve mid-frequency texture)
    band, _ = fft_bandpass(x, rlo_frac=FFT_RLO_FRAC, rhi_frac=FFT_RHI_FRAC, softness=FFT_SOFTNESS)

    # 5) CLAHE on bandpassed image
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    band_clahe = clahe.apply((band * 255).astype(np.uint8)).astype(np.float32) / 255.0

    # 6) Laplacian sharpening
    lap = cv2.Laplacian(band_clahe, cv2.CV_32F, ksize=3)
    enh = np.clip(band_clahe - LAPLACIAN_WEIGHT * lap, 0, 1)

    # 7) Return as 3-channel for torchvision transforms
    return np.stack([enh] * 3, axis=-1)  # H W 3
