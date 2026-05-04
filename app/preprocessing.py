"""
Mammography Preprocessing Functions
Modular implementation - each step is a separate function for step-by-step visualization
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List

# ═══════════════════════════════════════════════════════════════════════════
# COMMON UTILITY FUNCTIONS (used by multiple methods)
# ═══════════════════════════════════════════════════════════════════════════

def auto_crop_to_content(img: np.ndarray, margin: int = 10) -> np.ndarray:
    """
    Crops image to breast bounding box, removing black borders.
    Used by: ALL methods
    """
    _, binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(img.shape[1], x + w + margin)
    y_end = min(img.shape[0], y + h + margin)
    
    return img[y_start:y_end, x_start:x_end]


def resize_with_padding(img: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Letterbox resize - maintains aspect ratio, adds black padding.
    Used by: ALL methods
    """
    if len(img.shape) == 2:  # Grayscale
        h, w = img.shape
        channels = 1
    else:  # RGB
        h, w, channels = img.shape
    
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    if channels == 1:
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    else:
        canvas = np.zeros((target_size, target_size, channels), dtype=np.uint8)
    
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    
    return canvas


# ═══════════════════════════════════════════════════════════════════════════
# INDIVIDUAL PREPROCESSING STEPS
# ═══════════════════════════════════════════════════════════════════════════

def apply_gaussian_blur(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian filtering for noise reduction"""
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def apply_nlm_denoising(img: np.ndarray, strength: int = 10) -> np.ndarray:
    """Non-Local Means denoising"""
    return cv2.fastNlMeansDenoising(img, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def extract_breast_mask_otsu(img: np.ndarray) -> np.ndarray:
    """
    Otsu thresholding + connected components for breast segmentation.
    Used by: otsu_clahe_nlm method
    """
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return img
    
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest_idx).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return cv2.bitwise_and(img, img, mask=mask)


def remove_artefacts(img: np.ndarray) -> np.ndarray:
    """
    Removes small bright artifacts while preserving breast tissue.
    Used by: test method
    """
    _, binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return img
    
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest_idx).astype(np.uint8) * 255
    
    k = max(15, int(min(img.shape) * 0.01))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return cv2.bitwise_and(img, img, mask=mask)


def remove_scanlines(img: np.ndarray) -> np.ndarray:
    """
    Removes white scanning lines at top/bottom edges.
    Used by: test2 method
    """
    result = img.copy()
    H = img.shape[0]
    edge_zone = int(H * 0.06)
    
    # Top edge
    for r in range(edge_zone):
        if (img[r] > 40).mean() > 0.25:
            result[r, :] = 0
        else:
            break
    
    # Bottom edge
    for r in range(H - 1, H - 1 - edge_zone, -1):
        if (img[r] > 40).mean() > 0.25:
            result[r, :] = 0
        else:
            break
    
    return result


def apply_otsu_mask_fixed(img: np.ndarray) -> np.ndarray:
    """
    Fixed Otsu masking - computes threshold only on breast pixels.
    Used by: test2 method
    """
    breast_pixels = img[img > 5].reshape(-1, 1).astype(np.uint8)
    if breast_pixels.size == 0:
        return img
    
    t_raw, _ = cv2.threshold(breast_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_final = min(int(t_raw), 50)
    
    _, binary = cv2.threshold(img, t_final, 255, cv2.THRESH_BINARY)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return img
    
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest_idx).astype(np.uint8) * 255
    
    k = max(15, int(min(img.shape) * 0.01))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return cv2.bitwise_and(img, img, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════
# COMPLETE PREPROCESSING PIPELINES (with step tracking)
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_gaussian_clahe(img: np.ndarray, return_steps: bool = False) -> Dict:
    """
    Method 1: Gaussian + CLAHE
    Pipeline: Gaussian Blur → CLAHE → Crop → Resize
    """
    steps = {'original': img.copy()}
    
    # Step 1: Gaussian filtering
    img = apply_gaussian_blur(img, sigma=1.0)
    steps['gaussian'] = img.copy()
    
    # Step 2: CLAHE
    img = apply_clahe(img, clip_limit=2.0, tile_grid=(8, 8))
    steps['clahe'] = img.copy()
    
    # Step 3: Crop
    img = auto_crop_to_content(img, margin=10)
    steps['cropped'] = img.copy()
    
    # Step 4: Resize
    img = resize_with_padding(img, target_size=640)
    steps['resized'] = img.copy()
    
    # Convert to RGB
    final = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    steps['final'] = final
    
    if return_steps:
        return {'final': final, 'steps': steps}
    return {'final': final}


def preprocess_otsu_clahe_nlm(img: np.ndarray, return_steps: bool = False) -> Dict:
    """
    Method 2: Otsu + CLAHE + NLM
    Pipeline: NLM → Otsu Mask → Crop → CLAHE → Resize
    """
    steps = {'original': img.copy()}
    
    # Step 1: NLM denoising
    img = apply_nlm_denoising(img, strength=10)
    steps['nlm'] = img.copy()
    
    # Step 2: Otsu masking
    img = extract_breast_mask_otsu(img)
    steps['otsu_mask'] = img.copy()
    
    # Step 3: Crop
    img = auto_crop_to_content(img, margin=10)
    steps['cropped'] = img.copy()
    
    # Step 4: CLAHE
    img = apply_clahe(img, clip_limit=2.0, tile_grid=(8, 8))
    steps['clahe'] = img.copy()
    
    # Step 5: Resize
    img = resize_with_padding(img, target_size=640)
    steps['resized'] = img.copy()
    
    # Convert to RGB
    final = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    steps['final'] = final
    
    if return_steps:
        return {'final': final, 'steps': steps}
    return {'final': final}


def preprocess_test(img: np.ndarray, return_steps: bool = False) -> Dict:
    """
    Method 3: NLM + CLAHE (Optimized)
    Pipeline: Remove Artifacts → Crop → NLM → CLAHE → Resize
    """
    steps = {'original': img.copy()}
    
    # Step 1: Remove artifacts
    img = remove_artefacts(img)
    steps['artifacts_removed'] = img.copy()
    
    # Step 2: Crop
    img = auto_crop_to_content(img, margin=10)
    steps['cropped'] = img.copy()
    
    # Step 3: NLM (strength=5)
    img = apply_nlm_denoising(img, strength=5)
    steps['nlm'] = img.copy()
    
    # Step 4: CLAHE
    img = apply_clahe(img, clip_limit=2.0, tile_grid=(8, 8))
    steps['clahe'] = img.copy()
    
    # Step 5: Resize
    img = resize_with_padding(img, target_size=640)
    steps['resized'] = img.copy()
    
    # Convert to RGB
    final = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    steps['final'] = final
    
    if return_steps:
        return {'final': final, 'steps': steps}
    return {'final': final}


def preprocess_test2(img: np.ndarray, return_steps: bool = False) -> Dict:
    """
    Method 4: Full Pipeline (Best for CC+MLO)
    Pipeline: Crop → Remove Scanlines → Otsu Mask → CLAHE → NLM → Resize
    """
    steps = {'original': img.copy()}
    
    # Step 1: Crop
    img = auto_crop_to_content(img, margin=10)
    steps['cropped'] = img.copy()
    
    # Step 2: Remove scanlines
    img = remove_scanlines(img)
    steps['scanlines_removed'] = img.copy()
    
    # Step 3: Fixed Otsu mask
    img = apply_otsu_mask_fixed(img)
    steps['otsu_mask'] = img.copy()
    
    # Step 4: CLAHE
    img = apply_clahe(img, clip_limit=2.0, tile_grid=(8, 8))
    steps['clahe'] = img.copy()
    
    # Step 5: NLM (strength=5)
    img = apply_nlm_denoising(img, strength=5)
    steps['nlm'] = img.copy()
    
    # Step 6: Resize
    img = resize_with_padding(img, target_size=640)
    steps['resized'] = img.copy()
    
    # Convert to RGB
    final = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    steps['final'] = final
    
    if return_steps:
        return {'final': final, 'steps': steps}
    return {'final': final}


def preprocess_unprocessed(img: np.ndarray, return_steps: bool = False) -> Dict:
    """
    Method 5: Unprocessed (Baseline)
    Pipeline: Just Resize (with padding)
    """
    steps = {'original': img.copy()}
    
    # Only resize with padding
    img = resize_with_padding(img, target_size=640)
    steps['resized'] = img.copy()
    
    # Convert to RGB
    final = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    steps['final'] = final
    
    if return_steps:
        return {'final': final, 'steps': steps}
    return {'final': final}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PREPROCESSING DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_image(img: np.ndarray, method: str, return_steps: bool = False) -> Dict:
    """
    Main preprocessing function - dispatches to appropriate method.
    
    Args:
        img: Input grayscale image
        method: One of ['gaussian_clahe', 'otsu_clahe_nlm', 'test', 'test2', 'unprocessed']
        return_steps: If True, returns all intermediate steps for visualization
    
    Returns:
        Dictionary with 'final' image and optionally 'steps' dict
    """
    METHODS = {
        'gaussian_clahe': preprocess_gaussian_clahe,
        'otsu_clahe_nlm': preprocess_otsu_clahe_nlm,
        'test': preprocess_test,
        'test2': preprocess_test2,
        'unprocessed': preprocess_unprocessed
    }
    
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHODS.keys())}")
    
    return METHODS[method](img, return_steps=return_steps)