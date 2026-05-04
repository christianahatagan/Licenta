"""
Mammography Preprocessing Pipeline 
Methodology Source: SOCO 2026 Paper 
Algorithms: Gaussian Filter + CLAHE (Sequential)
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

    "output_dir"      : r"project/src/data/gaussian_clahe",
    "img_size"        : 640, 
    
    "gaussian_sigma"  : 1.0, 
    "clahe_clip_limit": 2.0, #
    "clahe_tile_grid" : (8, 8), 
    
    "crop_margin"     : 10
}

# ─────────────────────────────────────────────
# 1. PREPROCESSING FUNCTIONS
# ─────────────────────────────────────────────

def apply_combined_preprocessing(img: np.ndarray, config: dict) -> np.ndarray:
    """
    Applies the sequential combination of Gaussian filtering followed by CLAHE as described in the paper.
    """
    # Step 1: Gaussian Filtering (Noise Reduction)
    gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=config["gaussian_sigma"], sigmaY=config["gaussian_sigma"])
    
    # Step 2: CLAHE (Contrast Enhancement) 
    clahe_tool = cv2.createCLAHE(clipLimit=config["clahe_clip_limit"], tileGridSize=config["clahe_tile_grid"])
    combined = clahe_tool.apply(gaussian)
    
    return combined

def auto_crop_to_content(img: np.ndarray, margin: int = 10) -> np.ndarray:
    """
    Crops the image to the breasts bounding box to remove non-informative black borders.
    """
    _, binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    x_start, y_start = max(0, x - margin), max(0, y - margin)
    x_end, y_end = min(img.shape[1], x + w + margin), min(img.shape[0], y + h + margin)
    
    return img[y_start:y_end, x_start:x_end]

def resize_with_padding(img: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Resizes the image to 640x640 using letterbox padding to avoid distortion.
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
    Execution pipeline: Load -> Combined (Gauss+CLAHE) -> Crop -> Resize.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    img = apply_combined_preprocessing(img, config)
    
    img = auto_crop_to_content(img, margin=config["crop_margin"]) 
    img = resize_with_padding(img, target_size=config["img_size"])
    
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# ─────────────────────────────────────────────
# 2. DATA MANAGEMENT
# ─────────────────────────────────────────────

def build_series_lookup(project_root: str):
    """ SeriesInstanceUID to real system paths using DDSM metadata[cite: 62]. """
    df = pd.read_csv(Path(project_root) / r"CBIS-DDSM\csv\dicom_info.csv")
    full_mammo = df[df["SeriesDescription"] == "full mammogram images"].copy()
    return {str(row["SeriesInstanceUID"]).strip(): str(Path(project_root) / str(row["image_path"]).strip()) 
            for _, row in full_mammo.iterrows()}

def resolve_image_path(csv_image_path: str, series_lookup: dict):
    parts = Path(str(csv_image_path).strip()).parts
    return series_lookup.get(parts[2]) if len(parts) >= 3 else None

def split_and_save_manifest(records, output_path, folder_name):
  
    if not records: return
    df = pd.DataFrame(records)
    
    # 80/20 split
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    # Split the remaining 20% into two 10% halves
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    train_df['split'], val_df['split'], test_df['split'] = 'train', 'val', 'test'
    
    final_df = pd.concat([train_df, val_df, test_df])
    final_df.to_csv(output_path / f"manifest_{folder_name}.csv", index=False)
    print(f"[INFO] {folder_name} split complete: {len(final_df)} images.")

def run_pipeline(config: dict):
    root = Path(config["output_dir"])
    for f in ['cc', 'mlo', 'cc_mlo']: (root / f).mkdir(parents=True, exist_ok=True)
    
    series_lookup = build_series_lookup(config["project_root"])
    data_store = {'cc': [], 'mlo': [], 'cc_mlo': []}

    for csv_key in ["mass_train_csv", "mass_test_csv"]:
        df = pd.read_csv(Path(config["project_root"]) / config[csv_key])
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_key}"):
            img_path = resolve_image_path(row.get("image_file_path", ""), series_lookup)
            if not img_path or not Path(img_path).exists(): continue

            view = str(row.get('image_view', 'unk')).lower().strip()
            pathology = 1 if "MALIGNANT" in str(row.get("pathology")).upper() else 0
            
            processed_img = preprocess_image(img_path, config)
            if processed_img is None: continue

            filename = f"{row.get('patient_id')}_{view}_{idx}.png"
            record = {"image_path": filename, "label": pathology, "view": view}

            if 'cc' in view:
                save_dir = root / 'cc'
                data_store['cc'].append(record)
            elif 'mlo' in view:
                save_dir = root / 'mlo'
                data_store['mlo'].append(record)
            else: continue 

            # Save to specific folder and mixed folder
            Image.fromarray(processed_img).save(save_dir / filename)
            Image.fromarray(processed_img).save(root / 'cc_mlo' / filename)
            data_store['cc_mlo'].append(record)

    # Generate 80/10/10 split manifests 
    for category in ['cc', 'mlo', 'cc_mlo']:
        split_and_save_manifest(data_store[category], root, category)

if __name__ == "__main__":
    run_pipeline(CONFIG)