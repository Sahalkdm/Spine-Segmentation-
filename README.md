# Spine-Segmentation-
DDP Project - Spine Segmentation 

Data Set Link
https://doi.org/10.5281/zenodo.10159290

Doc Link
https://docs.google.com/document/d/1N2Ne-edD7aRawdxYOvBqriVZDZ7mYYiBtoV1Pa0BQtg/edit?tab=t.0

# nnU-Net v2 – Spine Vertebrae Segmentation (Spider Dataset)

This repository contains the complete pipeline for preparing spine MRI data and training nnU-Net v2 models for vertebral segmentation using **T1 + T2 MRI**.  
It supports both **Odd–Even vertebra labeling** and **standard multi-class vertebra labeling**.

The pipeline is designed to be:
- Robust to orientation issues
- Safe from silent mask misalignment
- Laptop-friendly (2D first, optional 3D full resolution)
- Easy to reproduce

---

## 1. Initial Raw Data Format

Raw data **must** be organized in the following structure:

```text
data/
├── images/
│   ├── <subject_id>_t1.mha
│   ├── <subject_id>_t2.mha
│   └── ...
└── masks/
    ├── <subject_id>_t1.mha
    ├── <subject_id>_t2.mha
    └── ...
```

### Important Notes
- `<subject_id>` must match across **T1**, **T2**, and **mask**
- File format must be `.mha`
- Orientation, spacing, and grid consistency are handled by the preprocessing code

---

## 2. Generated nnU-Net Dataset Structure

```text
nnUNet_raw/
└── Dataset102_SpiderOddEven/
    ├── imagesTr/
    │   ├── Spider_001_0000.nii.gz
    │   ├── Spider_001_0001.nii.gz
    ├── labelsTr/
    ├── imagesTs/
    ├── labelsTs/
    ├── labelsTr_instanceGT/
    └── dataset.json
```

---

## 3. Running the Pipeline

### Dataset Preparation
```bash
python dataset_preparation_5class.py
```

### Preprocessing
```bash
nnUNetv2_plan_and_preprocess -d 102 --verify_dataset_integrity
```

### Training (2D)
```bash
nnUNetv2_train 102 2d 0
```

### Training (3D Full Resolution)
```bash
nnUNetv2_train 102 3d_fullres 0
```

### Prediction
```bash
nnUNetv2_predict -i nnUNet_raw/Dataset102_SpiderOddEven/imagesTs \
                 -o nnUNet_results/Dataset102_SpiderOddEven/predictions_2d \
                 -d 102 -c 2d -f 0 --save_probabilities
```

---
