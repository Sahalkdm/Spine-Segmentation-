import os
import glob
import random
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json

# --- CONFIGURATION ---
RAW_IMAGES_DIR = Path("../data/images")
RAW_MASKS_DIR = Path("../data/masks")
OUTPUT_NNUNET_DIR = Path("../nnUNet_raw/Dataset103_SpiderOddEven")

TRAIN_RATIO = 0.85
RANDOM_SEED = 42

# --- LABEL DEFINITIONS (Spider) ---
INPUT_DISC_MIN = 200
INPUT_DISC_MAX = 299
INPUT_CANAL_ID = 100

# =========================================================
# ------------------ Utility Functions --------------------
# =========================================================

def enforce_orientation(img, orientation="RAS"):
    """Force image to a standard orientation (RAS)."""
    return sitk.DICOMOrient(img, orientation)

def safe_resample_mask(mask, reference_img):
    """
    Explicitly resample mask to reference image grid
    using Nearest Neighbor interpolation.
    """
    return sitk.Resample(
        mask,
        reference_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask.GetPixelID()
    )

def sort_vertebrae_spatially(arr, vertebra_ids):
    """
    Explicit Top → Bottom sorting (Superior → Inferior)
    after RAS orientation.
    """
    centroids = []
    for vid in vertebra_ids:
        coords = np.column_stack(np.where(arr == vid))
        z_mean = coords[:, 0].mean()  # Z-axis in RAS = Superior-Inferior
        centroids.append((vid, z_mean))

    # Sort from Superior (head, +Z) → Inferior (feet, -Z)
    centroids.sort(key=lambda x: x[1], reverse=True)
    return [vid for vid, _ in centroids]

def get_hybrid_vertebra_order(arr, vertebra_ids):
    """
    Hybrid strategy:
    - Prefer anatomical ID order
    - Validate with spatial monotonicity
    - Fallback to spatial order if needed
    """
    if len(vertebra_ids) <= 1:
        return vertebra_ids

    id_order = sorted(vertebra_ids)
    spatial_order = sort_vertebrae_spatially(arr, vertebra_ids)

    spatial_pos = {
        vid: np.mean(np.where(arr == vid)[0]) for vid in vertebra_ids
    }

    monotonic = all(
        spatial_pos[id_order[i]] > spatial_pos[id_order[i + 1]]
        for i in range(len(id_order) - 1)
    )

    return id_order if monotonic else spatial_order

# =========================================================
# ---------------- Label Remapping ------------------------
# =========================================================

def remap_to_5class_oddeven_hybrid(nifti_label):
    """
    Convert labels to:
    0: Background
    1: Vertebra (Odd)
    2: Vertebra (Even)
    3: Disc
    4: Spinal Canal
    """
    arr = sitk.GetArrayFromImage(nifti_label)
    new_arr = np.zeros_like(arr, dtype=np.uint8)

    unique_ids = np.unique(arr)
    unique_ids = unique_ids[unique_ids > 0]

    vertebra_ids = []
    for uid in unique_ids:
        if INPUT_DISC_MIN <= uid <= INPUT_DISC_MAX:
            continue
        if uid == INPUT_CANAL_ID:
            continue
        vertebra_ids.append(uid)

    vertebra_ids = get_hybrid_vertebra_order(arr, vertebra_ids)

    for idx, vid in enumerate(vertebra_ids):
        new_arr[arr == vid] = 1 if (idx % 2 == 0) else 2

    new_arr[(arr >= INPUT_DISC_MIN) & (arr <= INPUT_DISC_MAX)] = 3
    new_arr[arr == INPUT_CANAL_ID] = 4

    new_label = sitk.GetImageFromArray(new_arr)
    new_label.CopyInformation(nifti_label)
    return new_label

# =========================================================
# ---------------- Dataset JSON ---------------------------
# =========================================================

def generate_dataset_json(output_folder, num_training):
    json_dict = {
        "channel_names": {
            "0": "MRI"
        },
        "labels": {
            "background": 0,
            "vertebra_odd": 1,
            "vertebra_even": 2,
            "disc": 3,
            "spinal_canal": 4
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }

    with open(output_folder / "dataset.json", "w") as f:
        json.dump(json_dict, f, indent=4)

# =========================================================
# ---------------- Main Pipeline --------------------------
# =========================================================

def process_dataset():
    imagesTr = OUTPUT_NNUNET_DIR / "imagesTr"
    labelsTr = OUTPUT_NNUNET_DIR / "labelsTr"
    imagesTs = OUTPUT_NNUNET_DIR / "imagesTs"
    labelsTs = OUTPUT_NNUNET_DIR / "labelsTs"
    labelsTr_instGT = OUTPUT_NNUNET_DIR / "labelsTr_instanceGT"
    labelsTs_instGT = OUTPUT_NNUNET_DIR / "labelsTs_instanceGT"

    for d in [imagesTr, labelsTr, imagesTs, labelsTs, labelsTr_instGT, labelsTs_instGT]:
        d.mkdir(parents=True, exist_ok=True)

    # Base patient IDs off the T2 files
    t2_files = list(RAW_IMAGES_DIR.glob("*_t2.mha"))
    subject_ids = sorted({f.name.split("_t2")[0] for f in t2_files})

    random.seed(RANDOM_SEED)
    random.shuffle(subject_ids)

    split_idx = int(len(subject_ids) * TRAIN_RATIO)
    train_ids = set(subject_ids[:split_idx])

    written_train = 0

    for i, subject_id in enumerate(subject_ids):
        print(f"Processing Subject: {subject_id}")
        is_train = subject_id in train_ids

        # Process both T2 and T1 independently
        for modality in ["t2", "t1"]:
            
            # Locate image
            img_path = RAW_IMAGES_DIR / f"{subject_id}_{modality}.mha"
            if not img_path.exists() and modality == "t1":
                img_path = RAW_IMAGES_DIR / f"{subject_id}_t1_.mha" # Fallback check

            # Locate mask
            mask_path = RAW_MASKS_DIR / f"{subject_id}_{modality}.mha"
            if not mask_path.exists() and modality == "t1":
                mask_path = RAW_MASKS_DIR / f"{subject_id}_t1_.mha" # Fallback check

            if not (img_path.exists() and mask_path.exists()):
                print(f"  -> Missing {modality.upper()} image or mask. Skipping this modality.")
                continue

            # Unique nnUNet ID (e.g., Spider_001_T2)
            nnunet_id = f"Spider_{i+1:03d}_{modality.upper()}"

            dest_img = imagesTr if is_train else imagesTs
            dest_lbl = labelsTr if is_train else labelsTs
            dest_inst = labelsTr_instGT if is_train else labelsTs_instGT

            # Load
            img = sitk.ReadImage(str(img_path))
            lbl = sitk.ReadImage(str(mask_path))

            # Orientation normalization
            img = enforce_orientation(img)
            lbl = enforce_orientation(lbl)

            # Safely force mask onto its OWN image grid
            # This corrects minor sub-millimeter shifts if the mask header is slightly off
            lbl = safe_resample_mask(lbl, img)

            # Preserve original instance GT
            lbl_instance_gt = sitk.Image(lbl)

            # Hybrid odd-even remapping
            lbl_final = remap_to_5class_oddeven_hybrid(lbl)

            # Save
            sitk.WriteImage(img, str(dest_img / f"{nnunet_id}_0000.nii.gz"))
            sitk.WriteImage(lbl_final, str(dest_lbl / f"{nnunet_id}.nii.gz"))
            sitk.WriteImage(lbl_instance_gt, str(dest_inst / f"{nnunet_id}.nii.gz"))

            if is_train:
                written_train += 1
        if i >= 15:
            break

    generate_dataset_json(OUTPUT_NNUNET_DIR, written_train)
    print(f"Dataset preparation completed successfully. {written_train} training samples generated.")

if __name__ == "__main__":
    process_dataset()