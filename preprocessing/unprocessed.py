"""
Mammography Extraction Pipeline — Raw Version
Dataset: CBIS-DDSM
Operations: Simple Resize (640x640), Folder Categorization, Data Splitting
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
    "output_dir"      : r"project/src/data/unprocessed",
    "img_size"        : 640 
}

# ─────────────────────────────────────────────
# 1. IMAGE HANDLING FUNCTIONS
# ─────────────────────────────────────────────

def simple_resize_with_padding(img: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Resizes the image to the target size maintaining aspect ratio. 
    Adds black padding to fill the square (Letterboxing).
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create black canvas
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # Center the image on the canvas
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    
    return canvas

# ─────────────────────────────────────────────
# 2. PATH RESOLUTION HELPERS
# ─────────────────────────────────────────────

def build_series_lookup(project_root: str):
    # Scans the dicom_info.csv to map SeriesInstanceUIDs to the actual JPEG/PNG file paths
    dicom_csv = Path(project_root) / r"CBIS-DDSM\csv\dicom_info.csv"
    df = pd.read_csv(dicom_csv)
    full_mammo = df[df["SeriesDescription"] == "full mammogram images"].copy()
    
    return {
        str(row["SeriesInstanceUID"]).strip(): str(Path(project_root) / str(row["image_path"]).strip()) 
        for _, row in full_mammo.iterrows()
    }

def resolve_image_path(csv_image_path: str, series_lookup: dict):
    # Parses the fake path from the case description CSV to extract the UID and find the real file.
    parts = Path(str(csv_image_path).strip()).parts
    if len(parts) >= 3:
        series_uid = parts[2]
        return series_lookup.get(series_uid)
    return None

# ─────────────────────────────────────────────
# 3. SPLITTING AND BATCH PROCESSING
# ─────────────────────────────────────────────

def split_and_save_manifest(records, output_path, folder_name):
    # Performs an 80/10/10 split on the records and saves a manifest CSV for the specific category.
    if not records:
        print(f"[WARN] No records found for category: {folder_name}")
        return
        
    df = pd.DataFrame(records)
    
    # Split 80% Train, 20% Temporary
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    # Split the remaining 20% into half Validation (10%) and half Testing (10%)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    final_df = pd.concat([train_df, val_df, test_df])
    
    final_df.to_csv(output_path / f"manifest_{folder_name}.csv", index=False)
    print(f"[INFO] Manifest saved for {folder_name}: {len(final_df)} total images.")

def run_extraction(config: dict):
    # Iterates through dataset CSVs, resizes images, and saves them into CC, MLO, and CC_MLO folders.
    root_out = Path(config["output_dir"])
    for folder in ['cc', 'mlo', 'cc_mlo']:
        (root_out / folder).mkdir(parents=True, exist_ok=True)
    
    series_lookup = build_series_lookup(config["project_root"])
    
    data_store = {'cc': [], 'mlo': [], 'cc_mlo': []}

    for csv_key in ["mass_train_csv", "mass_test_csv"]:
        csv_path = Path(config["project_root"]) / config[csv_key]
        if not csv_path.exists():
            print(f"[ERROR] CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {csv_key}"):
            img_path = resolve_image_path(row.get("image_file_path", ""), series_lookup)
            
            if not img_path or not Path(img_path).exists():
                continue

            view = str(row.get('image_view', 'unk')).lower().strip()
            # 1 for Malignant, 0 for Benign
            pathology = 1 if "MALIGNANT" in str(row.get("pathology")).upper() else 0
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            # Basic resize for YOLO
            resized_img = simple_resize_with_padding(img, target_size=config["img_size"])
            # Convert to RGB for standard YOLO inputs
            final_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)

            # File naming convention
            filename = f"{row.get('patient_id')}_{view}_{idx}.png"
            record = {"image_path": filename, "label": pathology, "view": view}

            # Filter logic for folder organization
            if 'cc' in view:
                Image.fromarray(final_img).save(root_out / 'cc' / filename)
                data_store['cc'].append(record)
            elif 'mlo' in view:
                Image.fromarray(final_img).save(root_out / 'mlo' / filename)
                data_store['mlo'].append(record)
            else:
                continue

            # Every valid CC/MLO image also goes into the cc_mlo folder
            Image.fromarray(final_img).save(root_out / 'cc_mlo' / filename)
            data_store['cc_mlo'].append(record)

    # Generate 80/10/10 split manifests
    print("\n--- Generating Split Manifests ---")
    for category in ['cc', 'mlo', 'cc_mlo']:
        split_and_save_manifest(data_store[category], root_out, category)

if __name__ == "__main__":
    run_extraction(CONFIG)