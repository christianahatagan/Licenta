"""
Mammography Preprocessing Pipeline — NLM + CLAHE 
Pipeline: Load -> Artefact removal -> Crop -> NLM -> CLAHE -> Letterbox resize -> RGB
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "project_root"    : r"C:\Users\Christiana\Desktop\LICENTA\project",
    "mass_train_csv"  : r"CBIS-DDSM\csv\mass_case_description_train_set.csv",
    "mass_test_csv"   : r"CBIS-DDSM\csv\mass_case_description_test_set.csv",
    "dicom_info_csv"  : r"CBIS-DDSM\csv\dicom_info.csv",

    "output_dir"      : r"C:\Users\Christiana\Desktop\LICENTA\project\src\data\test",

    "img_size"        : 640,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid" : (8, 8),
    "nlm_strength"    : 5,
    "crop_margin"     : 10,
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PREPROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def remove_artefacts(img: np.ndarray) -> np.ndarray:
    """
    Removes small bright artefacts (e.g. radiological markers) while keeping all breast tissue
    FIX vs. original extract_breast_mask():
    - Threshold > 10 instead of Otsu. Otsu on a dark mammogram auto-picks ~70, destroying ~80% of pixels including soft-tissue edges around masses.
    - Morphological close kernel is scaled to 1% of image size (min 15 px) so it actually fills gaps at any resolution, not just at 640×640.
    """
    _, binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return img

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest_idx).astype(np.uint8) * 255

    # Kernel scaled to image size — effective at any resolution
    k = max(15, int(min(img.shape) * 0.01))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return cv2.bitwise_and(img, img, mask=mask)


def auto_crop_to_content(img: np.ndarray, margin: int = 10) -> np.ndarray:
    """
    Crops to the breast bounding box, removing black borders.
    Identical logic to the Gaussian+CLAHE pipeline
    """
    _, binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return img

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end   = min(img.shape[1], x + w + margin)
    y_end   = min(img.shape[0], y + h + margin)

    return img[y_start:y_end, x_start:x_end]


def apply_nlm(img: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Non-Local Means denoising.
    FIX vs. original: called AFTER crop, so it runs only on the breastegion (fewer pixels) instead of the full 3000-6000 px scan.
    """
    return cv2.fastNlMeansDenoising( img, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def resize_with_padding(img: np.ndarray, target_size: int = 640) -> np.ndarray:
    """Letterbox resize — keeps aspect ratio, pads remainder with zeros."""
    h, w   = img.shape
    scale  = target_size / max(h, w)
    new_h  = int(h * scale)
    new_w  = int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off  = (target_size - new_h) // 2
    x_off  = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def preprocess_image(image_path: str, config: dict):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = remove_artefacts(img)
    img = auto_crop_to_content(img, margin=config["crop_margin"])
    img = apply_nlm(img, strength=config["nlm_strength"])
    img = apply_clahe(img,
                      clip_limit=config["clahe_clip_limit"],
                      tile_grid=config["clahe_tile_grid"])
    img = resize_with_padding(img, target_size=config["img_size"])

    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FILE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_series_lookup(project_root: str) -> dict:
    """SeriesInstanceUID to absolute JPEG path for full mammogram images."""
    df = pd.read_csv(Path(project_root) / r"CBIS-DDSM\csv\dicom_info.csv")
    full_mammo = df[df["SeriesDescription"] == "full mammogram images"].copy()
    return {
        str(row["SeriesInstanceUID"]).strip():
        str(Path(project_root) / str(row["image_path"]).strip())
        for _, row in full_mammo.iterrows()
    }


def resolve_image_path(csv_image_path: str, series_lookup: dict):
    """SeriesUID from the CSV path and returns the absolute file path"""
    parts = Path(str(csv_image_path).strip()).parts
    return series_lookup.get(parts[2]) if len(parts) >= 3 else None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MANIFEST / SPLIT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def split_and_save_manifest(records: list, output_path: Path, folder_name: str):
    """Splits records 80 / 10 / 10 and saves a manifest CSV."""
    if not records:
        return
    df = pd.DataFrame(records)

    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    val_df,   test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    train_df["split"] = "train"
    val_df["split"]   = "val"
    test_df["split"]  = "test"

    final_df = pd.concat([train_df, val_df, test_df])
    final_df.to_csv(output_path / f"manifest_{folder_name}.csv", index=False)
    print(f"[INFO] Manifest saved — {folder_name}: {len(final_df)} images "
          f"(train={len(train_df)} / val={len(val_df)} / test={len(test_df)})")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BATCH PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(config: dict):
    """
    Iterates over both mass CSVs, preprocesses every image with the fixed pipeline, and saves the results under:
        src/data/test/
            cc/         
            mlo/         
            cc_mlo/      
            manifest_cc.csv
            manifest_mlo.csv
            manifest_cc_mlo.csv
    """
    root = Path(config["output_dir"])
    for folder in ("cc", "mlo", "cc_mlo"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    series_lookup = build_series_lookup(config["project_root"])
    data_store    = {"cc": [], "mlo": [], "cc_mlo": []}
    skipped       = 0

    for csv_key in ("mass_train_csv", "mass_test_csv"):
        csv_path = Path(config["project_root"]) / config[csv_key]
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        for idx, row in tqdm(df.iterrows(), total=len(df),
                             desc=f"Processing {Path(csv_path).name}"):

            img_path = resolve_image_path(
                row.get("image_file_path", ""), series_lookup
            )
            if not img_path or not Path(img_path).exists():
                skipped += 1
                continue

            view = str(row.get("image_view", "unk")).lower().strip()
            pathology = 1 if "MALIGNANT" in str(row.get("pathology", "")).upper() else 0

            processed_img = preprocess_image(img_path, config)
            if processed_img is None:
                skipped += 1
                continue

            filename = f"{row.get('patient_id')}_{view}_{idx}.png"
            record   = {"image_path": filename, "label": pathology, "view": view}

            if "cc" in view:
                Image.fromarray(processed_img).save(root / "cc" / filename)
                data_store["cc"].append(record)
            elif "mlo" in view:
                Image.fromarray(processed_img).save(root / "mlo" / filename)
                data_store["mlo"].append(record)
            else:
                skipped += 1
                continue

            # Always save to the combined folder as well
            Image.fromarray(processed_img).save(root / "cc_mlo" / filename)
            data_store["cc_mlo"].append(record)

    for category in ("cc", "mlo", "cc_mlo"):
        split_and_save_manifest(data_store[category], root, category)

    print(f"\n[INFO] Skipped (missing / unreadable): {skipped}")
    print(f"[DONE] All images saved to: {root}")

if __name__ == "__main__":
    process_dataset(CONFIG)