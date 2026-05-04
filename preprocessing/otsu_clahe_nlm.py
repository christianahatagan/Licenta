"""
Mammography Preprocessing Pipeline 
Pipeline: NLM Denoising -> Otsu Masking -> CLAHE -> Auto-Crop
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# 0. CONFIGURATION 
# ─────────────────────────────────────────────

CONFIG = {
    "project_root"    : r"C:\Users\Christiana\Desktop\LICENTA\project",
    "mass_train_csv"  : r"CBIS-DDSM\csv\mass_case_description_train_set.csv",
    "mass_test_csv"   : r"CBIS-DDSM\csv\mass_case_description_test_set.csv",
    "dicom_info_csv"  : r"CBIS-DDSM\csv\dicom_info.csv",

    "output_dir"      : r"project/src/data/otsu_clahe_nlm",

    "img_size"        : 640, 
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid" : (8, 8),
    "nlm_strength"    : 10,  
    "crop_margin"     : 10
}

# ─────────────────────────────────────────────
# 1. PREPROCESSING ALGORITHMS
# ─────────────────────────────────────────────

def apply_nlm(img: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Applies Non-Local Means (NLM) Denoising to reduce image noise while preserving edges and fine anatomical details.
    """
    return cv2.fastNlMeansDenoising(img, None, h=strength, templateWindowSize=7, searchWindowSize=21)

def extract_breast_mask(img: np.ndarray) -> np.ndarray:
    """
    Segments the breast tissue from the background using Otsu's thresholding and connected components analysis to remove artifacts/noise.
    """
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1: return img
    
    # Identify the largest connected component (assumed to be the breast)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    
    # Morphological closing to fill small holes in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return cv2.bitwise_and(img, img, mask=mask)

def auto_crop_to_content(img: np.ndarray, margin: int = 10) -> np.ndarray:
    """
    Automatically crops the image to the breast's bounding box to eliminate unnecessary black borders and maximize effective resolution.
    """
    _, binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    x_start, y_start = max(0, x - margin), max(0, y - margin)
    x_end, y_end = min(img.shape[1], x + w + margin), min(img.shape[0], y + h + margin)
    
    return img[y_start:y_end, x_start:x_end]

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance local contrast and highlight dense features in the mammogram.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)

def resize_with_padding(img: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Resizes the image to the target size while maintaining aspect ratio, adding letterbox padding to ensure the final output is square.
    """
    h, w = img.shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off, x_off = (target_size - new_h) // 2, (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

def preprocess_image(image_path: str, config: dict):
    """
    Full preprocessing pipeline: Load -> NLM -> Mask -> Crop -> CLAHE -> Resize.
    Returns the processed image in RGB format for YOLO compatibility.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    img = apply_nlm(img, strength=config["nlm_strength"]) 
    img = extract_breast_mask(img)                        
    img = auto_crop_to_content(img, margin=config["crop_margin"]) 
    img = apply_clahe(img, clip_limit=config["clahe_clip_limit"]) 
    img = resize_with_padding(img, target_size=config["img_size"])
    
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# ─────────────────────────────────────────────
# 2. FILE HELPERS
# ─────────────────────────────────────────────

def build_series_lookup(project_root: str):
    """SeriesInstanceUID to absolute JPEG path for full mammogram images."""
    df = pd.read_csv(Path(project_root) / r"CBIS-DDSM\csv\dicom_info.csv")
    full_mammo = df[df["SeriesDescription"] == "full mammogram images"].copy()
    return {str(row["SeriesInstanceUID"]).strip(): str(Path(project_root) / str(row["image_path"]).strip()) 
            for _, row in full_mammo.iterrows()}

def resolve_image_path(csv_image_path: str, series_lookup: dict):
    """ SeriesUID from the CSV path and returns the absolute system path"""
    parts = Path(str(csv_image_path).strip()).parts
    return series_lookup.get(parts[2]) if len(parts) >= 3 else None

# ─────────────────────────────────────────────
# 3. DATA SPLIT & BATCH PROCESSING
# ─────────────────────────────────────────────

def split_and_save_manifest(records, output_path, folder_name):
    """
    Splits records into Train (80%), Val (10%), and Test (10%) sets and saves a manifest CSV.
    """
    if not records: return
    df = pd.DataFrame(records)
    
    # Initial split: Train (80%) vs Temp (20%)
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    # Secondary split: Val (10%) and Test (10%)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    final_df = pd.concat([train_df, val_df, test_df])
    final_df.to_csv(output_path / f"manifest_{folder_name}.csv", index=False)
    print(f"[INFO] Saved manifest for {folder_name}: {len(final_df)} images.")

def process_dataset(config: dict):
    root = Path(config["output_dir"])
    for f in ['cc', 'mlo', 'cc_mlo']: (root / f).mkdir(parents=True, exist_ok=True)
    
    series_lookup = build_series_lookup(config["project_root"])
    data_store = {'cc': [], 'mlo': [], 'cc_mlo': []}

    for split_csv in ["mass_train_csv", "mass_test_csv"]:
        csv_path = Path(config["project_root"]) / config[split_csv]
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_csv}"):
            img_path = resolve_image_path(row.get("image_file_path", ""), series_lookup)
            if not img_path or not Path(img_path).exists(): continue

            view = str(row.get('image_view', 'unk')).lower().strip()
            pathology = 1 if "MALIGNANT" in str(row.get("pathology")).upper() else 0
            
            processed_img = preprocess_image(img_path, config)
            if processed_img is None: continue

            # Create descriptive filename
            filename = f"{row.get('patient_id')}_{view}_{idx}.png"
            record = {"image_path": filename, "label": pathology, "view": view}

            # Categorization logic
            if 'cc' in view:
                save_dir = root / 'cc'
                data_store['cc'].append(record)
            elif 'mlo' in view:
                save_dir = root / 'mlo'
                data_store['mlo'].append(record)
            else: continue 

            # Save processed image to specific folder
            Image.fromarray(processed_img).save(save_dir / filename)
            
            # Save to mixed folder (independent copies for easy access)
            Image.fromarray(processed_img).save(root / 'cc_mlo' / filename)
            data_store['cc_mlo'].append(record)

    # Generate split manifests (80/10/10)
    for category in ['cc', 'mlo', 'cc_mlo']:
        split_and_save_manifest(data_store[category], root, category)

if __name__ == "__main__":
    process_dataset(CONFIG)