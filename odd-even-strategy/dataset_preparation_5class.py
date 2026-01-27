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
OUTPUT_NNUNET_DIR = Path("../nnUNet_raw/Dataset102_SpiderOddEven")

TRAIN_RATIO = 0.8
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
    FIX #1:
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
    FIX #2:
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
# ---------------- Registration ---------------------------
# =========================================================

def register_t1_to_t2(t1_img, t2_img):
    t1_img = sitk.Cast(t1_img, sitk.sitkFloat32)
    t2_img = sitk.Cast(t2_img, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        t2_img,
        t1_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(50)
    registration.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=100
    )
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(t2_img, t1_img)

    t1_registered = sitk.Resample(
        t1_img,
        t2_img,
        final_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32
    )

    return t1_registered

def resample_to_reference(img, reference, interp=sitk.sitkLinear, default_value=0.0):
    """
    ALIGNMENT FIX:
    Instead of 'registering' (calculating new rotation), we simply
    'resample' the image onto the reference grid.
    
    This relies on the physical coordinates (Origin/Spacing) in the file header,
    which is usually correct for T1/T2 acquired in the same session.
    """
    return sitk.Resample(
        img,
        reference,
        sitk.Transform(), # Identity transform (no rotation/shift calculated)
        interp,
        default_value,
        img.GetPixelID()
    )

# =========================================================
# ---------------- Dataset JSON ---------------------------
# =========================================================

def generate_dataset_json(output_folder, num_training):
    json_dict = {
        "channel_names": {
            "0": "T2",
            "1": "T1"
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

    t2_files = list(RAW_IMAGES_DIR.glob("*_t2.mha"))
    subject_ids = sorted({f.name.split("_t2")[0] for f in t2_files})

    random.seed(RANDOM_SEED)
    random.shuffle(subject_ids)

    split_idx = int(len(subject_ids) * TRAIN_RATIO)
    train_ids = set(subject_ids[:split_idx])

    written_train = 0

    for i, subject_id in enumerate(subject_ids):

        print("processing: ", subject_id)

        is_train = subject_id in train_ids
        dest_img = imagesTr if is_train else imagesTs
        dest_lbl = labelsTr if is_train else labelsTs
        dest_inst = labelsTr_instGT if is_train else labelsTs_instGT

        t2_path = RAW_IMAGES_DIR / f"{subject_id}_t2.mha"
        t1_path = RAW_IMAGES_DIR / f"{subject_id}_t1.mha"
        mask_path = RAW_MASKS_DIR / f"{subject_id}_t2.mha"

        if not (t2_path.exists() and mask_path.exists()):
            continue

        if not t1_path.exists():
            t1_path = RAW_IMAGES_DIR / f"{subject_id}_t1_.mha"
            if not t1_path.exists():
                continue

        nnunet_id = f"Spider_{i+1:03d}"

        # Load
        t2 = sitk.ReadImage(str(t2_path))
        t1 = sitk.ReadImage(str(t1_path))
        lbl = sitk.ReadImage(str(mask_path))

        # Orientation normalization
        t2 = enforce_orientation(t2)
        t1 = enforce_orientation(t1)
        lbl = enforce_orientation(lbl)

        # FIX #1: Force mask onto T2 grid
        lbl = safe_resample_mask(lbl, t2)

        # Preserve original instance GT
        lbl_instance_gt = sitk.Image(lbl)

        # Resample T1 to T2 Geometry (NO REGISTRATION CALCULATION)
        # This trusts the physical coordinates in the header are correct.
        t1_reg = resample_to_reference(t1, t2, interp=sitk.sitkLinear)

        # Hybrid odd-even remapping
        lbl_final = remap_to_5class_oddeven_hybrid(lbl)

        # Save
        sitk.WriteImage(t2, str(dest_img / f"{nnunet_id}_0000.nii.gz"))
        sitk.WriteImage(t1_reg, str(dest_img / f"{nnunet_id}_0001.nii.gz"))
        sitk.WriteImage(lbl_final, str(dest_lbl / f"{nnunet_id}.nii.gz"))
        sitk.WriteImage(lbl_instance_gt, str(dest_inst / f"{nnunet_id}.nii.gz"))

        if is_train:
            written_train += 1

    generate_dataset_json(OUTPUT_NNUNET_DIR, written_train)
    print("Dataset preparation completed successfully.")

if __name__ == "__main__":
    process_dataset()

# import os
# import glob
# import random
# import numpy as np
# import SimpleITK as sitk
# from pathlib import Path
# import json

# # --- CONFIGURATION ---
# RAW_IMAGES_DIR = Path("../data/images")   
# RAW_MASKS_DIR = Path("../data/masks")     
# OUTPUT_NNUNET_DIR = Path("../nnUNet_raw/Dataset501_SpiderOddEven")

# # Dataset Split
# TRAIN_RATIO = 0.8
# RANDOM_SEED = 42

# # --- ROBUST LABEL DEFINITIONS ---
# # Instead of hard ranges, we define "Exclusion Zones"
# # We assume:
# #   - Background is 0
# #   - Canal is a specific high number (e.g. 100)
# #   - Discs are in a specific high range (e.g. 200-299)
# #   - EVERYTHING ELSE > 0 is treated as a Vertebra
# INPUT_DISC_MIN = 200
# INPUT_DISC_MAX = 299
# INPUT_CANAL_ID = 100  

# # =========================================================

# def enforce_orientation(img, orientation="RAS"):
#     """
#     Forces the image into a standard orientation (e.g., RAS)
#     to ensure 'Up' is always 'Up' across all patients.
#     """
#     return sitk.DICOMOrient(img, orientation)

# def remap_to_5class_oddeven_robust(nifti_label):
#     """
#     Robustly converts labels to 5 Classes using DYNAMIC sorting.
#     """
#     # 1. Get raw data
#     arr = sitk.GetArrayFromImage(nifti_label)
#     new_arr = np.zeros_like(arr, dtype=np.uint8)
    
#     # 2. Find all unique labels in this specific file
#     unique_ids = np.unique(arr)
#     unique_ids = unique_ids[unique_ids > 0] # Ignore background
    
#     # 3. Identify Vertebrae Dynamically
#     # Logic: It's a vertebra if it's NOT a disc and NOT the canal
#     vertebra_ids = []
#     for uid in unique_ids:
#         if (uid >= INPUT_DISC_MIN and uid <= INPUT_DISC_MAX):
#             continue # It's a disc
#         if uid == INPUT_CANAL_ID:
#             continue # It's the canal
#         vertebra_ids.append(uid)
    
#     # 4. Sort them (Spatial order usually correlates with ID magnitude in Spider)
#     # If IDs are random, we might need spatial sorting, but usually ID 1 < ID 2.
#     vertebra_ids = sorted(vertebra_ids)
    
#     # 5. Assign Odd/Even based on SORTED ORDER (Relative Index)
#     # 1st bone found -> Class 1
#     # 2nd bone found -> Class 2
#     # 3rd bone found -> Class 1 ...
#     for idx, v_id in enumerate(vertebra_ids):
#         if (idx + 1) % 2 != 0: # 1st, 3rd, 5th...
#             new_arr[arr == v_id] = 1 # ODD Class
#         else:
#             new_arr[arr == v_id] = 2 # EVEN Class

#     # 6. Map Discs (Class 3)
#     mask_discs = (arr >= INPUT_DISC_MIN) & (arr <= INPUT_DISC_MAX)
#     new_arr[mask_discs] = 3

#     # 7. Map Canal (Class 4)
#     new_arr[arr == INPUT_CANAL_ID] = 4

#     # 8. Reconstruct Image
#     new_label = sitk.GetImageFromArray(new_arr)
#     new_label.CopyInformation(nifti_label) 
#     return new_label

# def register_t1_to_t2(t1_img, t2_img):
#     t1_img = sitk.Cast(t1_img, sitk.sitkFloat32)
#     t2_img = sitk.Cast(t2_img, sitk.sitkFloat32)
    
#     # Initialize
#     initial_transform = sitk.CenteredTransformInitializer(
#         t2_img, t1_img,
#         sitk.Euler3DTransform(),
#         sitk.CenteredTransformInitializerFilter.GEOMETRY
#     )
    
#     # Register
#     registration = sitk.ImageRegistrationMethod()
#     registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
#     registration.SetOptimizerAsGradientDescentLineSearch(
#         learningRate=1.0, numberOfIterations=100
#     )
#     registration.SetInterpolator(sitk.sitkLinear)
#     registration.SetInitialTransform(initial_transform, inPlace=False)
#     final_transform = registration.Execute(t2_img, t1_img)
    
#     # Resample
#     t1_registered = sitk.Resample(
#         t1_img, t2_img, final_transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32
#     )
#     return t1_registered

# def generate_dataset_json(output_folder, num_training):
#     json_dict = {
#         "channel_names": { "0": "T2", "1": "T1" },
#         "labels": {
#             "background": 0,
#             "vertebra_odd": 1,
#             "vertebra_even": 2,
#             "disc": 3,
#             "spinal_canal": 4
#         },
#         "numTraining": num_training,
#         "file_ending": ".nii.gz"
#     }
#     with open(output_folder / "dataset.json", 'w') as f:
#         json.dump(json_dict, f, indent=4)

# def process_dataset():
#     # Setup
#     imagesTr = OUTPUT_NNUNET_DIR / "imagesTr"
#     labelsTr = OUTPUT_NNUNET_DIR / "labelsTr"
#     imagesTs = OUTPUT_NNUNET_DIR / "imagesTs"
#     labelsTs = OUTPUT_NNUNET_DIR / "labelsTs"
#     labelsTr_instGT = OUTPUT_NNUNET_DIR / "labelsTr_instanceGT"
#     labelsTs_instGT = OUTPUT_NNUNET_DIR / "labelsTs_instanceGT"
    
#     for d in [imagesTr, labelsTr, imagesTs, labelsTs, labelsTr_instGT, labelsTs_instGT]:
#         d.mkdir(parents=True, exist_ok=True)

#     # Scan Data
#     print(f"Scanning {RAW_IMAGES_DIR}...")
#     t2_files = list(RAW_IMAGES_DIR.glob("*_t2.mha"))
#     subject_ids = sorted(list(set([f.name.split('_t2')[0] for f in t2_files])))
#     print(f"Found {len(subject_ids)} subjects.")

#     # Split
#     random.seed(RANDOM_SEED)
#     random.shuffle(subject_ids)
#     split_idx = int(len(subject_ids) * TRAIN_RATIO)
#     train_ids = set(subject_ids[:split_idx])

#     print("Starting Robust Processing (RAS + Dynamic IDs)...")

#     for i, subject_id in enumerate(subject_ids):
#         is_train = subject_id in train_ids
#         dest_img = imagesTr if is_train else imagesTs
#         dest_lbl = labelsTr if is_train else labelsTs
        
#         # Paths
#         t2_path = RAW_IMAGES_DIR / f"{subject_id}_t2.mha"
#         mask_path = RAW_MASKS_DIR / f"{subject_id}_t2.mha"
        
#         # T1 Check
#         t1_path = RAW_IMAGES_DIR / f"{subject_id}_t1.mha"
#         if not t1_path.exists(): 
#             t1_path = RAW_IMAGES_DIR / f"{subject_id}_t1_.mha"

#         if not t1_path.exists() or not mask_path.exists():
#             print(f"Skipping {subject_id}: Missing files.")
#             continue

#         nnunet_id = f"Spider_{i+1:03d}"
#         print(f"Processing {subject_id} -> {nnunet_id}...")

#         # 1. LOAD IMAGES
#         t1 = sitk.ReadImage(str(t1_path))
#         t2 = sitk.ReadImage(str(t2_path))
#         lbl = sitk.ReadImage(str(mask_path))

#         # 2. ENFORCE ORIENTATION (RAS) - Critical Fix!
#         t1 = enforce_orientation(t1, "RAS")
#         t2 = enforce_orientation(t2, "RAS")
#         lbl = enforce_orientation(lbl, "RAS")

#         # ORIGINAL instance GT (DO NOT MODIFY)
#         lbl_instance_gt = lbl

#         # 3. REGISTER T1 -> T2
#         # (Must be done AFTER orientation fix to ensure they match)
#         t1_reg = register_t1_to_t2(t1, t2)

#         # 4. REMAP LABELS (Robust Method)
#         lbl_final = remap_to_5class_oddeven_robust(lbl)

#         # 5. SAVE
#         sitk.WriteImage(t2, str(dest_img / f"{nnunet_id}_0000.nii.gz"))
#         sitk.WriteImage(t1_reg, str(dest_img / f"{nnunet_id}_0001.nii.gz"))
#         sitk.WriteImage(lbl_final, str(dest_lbl / f"{nnunet_id}.nii.gz"))
#         # Save ORIGINAL GT (ignored by nnU-Net)
#         sitk.WriteImage(lbl_instance_gt, str((labelsTr_instGT if is_train else labelsTs_instGT) / f"{nnunet_id}.nii.gz"))

#     generate_dataset_json(OUTPUT_NNUNET_DIR, len(train_ids))
#     print("\nProcessing Complete.")

# if __name__ == "__main__":
#     process_dataset()