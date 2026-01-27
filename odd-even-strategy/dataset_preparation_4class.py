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
OUTPUT_NNUNET_DIR = Path("../nnUNet_raw/Dataset101_Spider4Class")

# Dataset Split
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# --- LABEL DEFINITIONS ---
# Background = 0
# Vertebrae = everything >0 except disc + canal
INPUT_DISC_MIN = 200
INPUT_DISC_MAX = 299
INPUT_CANAL_ID = 100

# =========================================================

def enforce_orientation(img, orientation="RAS"):
    return sitk.DICOMOrient(img, orientation)

def resample_to_reference(img, reference, interp):
    return sitk.Resample(
        img,
        reference,
        sitk.Transform(),
        interp,
        0,
        img.GetPixelID()
    )

def remap_to_4class(nifti_label):
    """
    Maps labels to:
    0 = background
    1 = vertebra
    2 = disc
    3 = spinal canal
    """
    arr = sitk.GetArrayFromImage(nifti_label)
    new_arr = np.zeros_like(arr, dtype=np.uint8)

    # Disc
    disc_mask = (arr >= INPUT_DISC_MIN) & (arr <= INPUT_DISC_MAX)
    new_arr[disc_mask] = 2

    # Canal
    new_arr[arr == INPUT_CANAL_ID] = 3

    # Vertebra = everything else > 0
    vertebra_mask = (arr > 0) & (~disc_mask) & (arr != INPUT_CANAL_ID)
    new_arr[vertebra_mask] = 1

    new_label = sitk.GetImageFromArray(new_arr)
    new_label.CopyInformation(nifti_label)
    return new_label

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
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)
    registration.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        windowSize=10
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    try:
        final_transform = registration.Execute(t2_img, t1_img)
    except Exception as e:
        print(f"Registration failed, using initial transform: {e}")
        final_transform = initial_transform

    t1_registered = sitk.Resample(
        t1_img,
        t2_img,
        final_transform,
        sitk.sitkLinear,
        0.0,
        t1_img.GetPixelID()
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

def generate_dataset_json(output_folder, num_training):
    json_dict = {
        "channel_names": {
            "0": "T2",
            "1": "T1"
        },
        "labels": {
            "background": 0,
            "vertebra": 1,
            "disc": 2,
            "spinal_canal": 3
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }
    with open(output_folder / "dataset.json", "w") as f:
        json.dump(json_dict, f, indent=4)

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

    for i, subject_id in enumerate(subject_ids):
        is_train = subject_id in train_ids
        dest_img = imagesTr if is_train else imagesTs
        dest_lbl = labelsTr if is_train else labelsTs

        t2_path = RAW_IMAGES_DIR / f"{subject_id}_t2.mha"
        t1_path = RAW_IMAGES_DIR / f"{subject_id}_t1.mha"
        if not t1_path.exists():
            t1_path = RAW_IMAGES_DIR / f"{subject_id}_t1_.mha"
        mask_path = RAW_MASKS_DIR / f"{subject_id}_t2.mha"

        if not t1_path.exists() or not mask_path.exists():
            print(f"Skipping {subject_id}: missing files")
            continue

        nnunet_id = f"Spider_{i+1:03d}"
        print(f"Processing {subject_id} → {nnunet_id}")

        t2 = enforce_orientation(sitk.ReadImage(str(t2_path)))
        t1 = enforce_orientation(sitk.ReadImage(str(t1_path)))
        lbl = enforce_orientation(sitk.ReadImage(str(mask_path)))

        # Register T1 → T2
        # t1_reg = register_t1_to_t2(t1, t2)
        
        # Resample T1 to T2 Geometry (NO REGISTRATION CALCULATION)
        # This trusts the physical coordinates in the header are correct.
        t1_reg = resample_to_reference(t1, t2, interp=sitk.sitkLinear)

        # Ensure label is on T2 grid
        lbl = resample_to_reference(lbl, t2, sitk.sitkNearestNeighbor)

        # Remap labels
        lbl_final = remap_to_4class(lbl)

        # Save original instance GT (optional)
        lbl_instance_gt = lbl

        sitk.WriteImage(t2, str(dest_img / f"{nnunet_id}_0000.nii.gz"))
        sitk.WriteImage(t1_reg, str(dest_img / f"{nnunet_id}_0001.nii.gz"))
        sitk.WriteImage(lbl_final, str(dest_lbl / f"{nnunet_id}.nii.gz"))
        sitk.WriteImage(
            lbl_instance_gt,
            str((labelsTr_instGT if is_train else labelsTs_instGT) / f"{nnunet_id}.nii.gz")
        )

    generate_dataset_json(OUTPUT_NNUNET_DIR, len(train_ids))
    print("Processing complete.")

if __name__ == "__main__":
    process_dataset()
