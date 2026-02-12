# scripts/convert_brats2020_to_nnunet.py

from pathlib import Path
import shutil
import json
import re
import numpy as np
import nibabel as nib
from tqdm import tqdm


def convert_brats2020(
    raw_external_dir: Path,
    nnunet_raw_dir: Path,
    dataset_id: int = 101,
):
    """
    Convert BraTS 2020 to nnU-Net v2 raw format.
    """
    train_src = (
    raw_external_dir
    / "BraTS2020_TrainingData"
    / "MICCAI_BraTS2020_TrainingData"
    )

    assert train_src.exists(), f"Missing {train_src}"

    dataset_name = "BraTS2020"
    dataset_dir = nnunet_raw_dir / f"Dataset{dataset_id:03d}_{dataset_name}"
    imagesTr = dataset_dir / "imagesTr"
    labelsTr = dataset_dir / "labelsTr"

    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([p for p in train_src.iterdir() if p.is_dir()])
    print(f"Found {len(patient_dirs)} training cases")

    patient_count = 0
    id_pattern = re.compile(r"BraTS20_Training_(\d+)")

    for pdir in tqdm(patient_dirs):
        m = id_pattern.match(pdir.name)
        if not m:
            continue

        pid = m.group(1)
        out_prefix = f"BRATS_{pid}"

        # source files
        src = {
            "flair": pdir / f"{pdir.name}_flair.nii",
            "t1":    pdir / f"{pdir.name}_t1.nii",
            "t1ce":  pdir / f"{pdir.name}_t1ce.nii",
            "t2":    pdir / f"{pdir.name}_t2.nii",
            "seg":   pdir / f"{pdir.name}_seg.nii",
        }

        if not all(f.exists() for f in src.values()):
            print(f"Skip {pdir.name} (missing files)")
            continue

        # copy images
        shutil.copy(src["flair"], imagesTr / f"{out_prefix}_0000.nii.gz")
        shutil.copy(src["t1"],    imagesTr / f"{out_prefix}_0001.nii.gz")
        shutil.copy(src["t1ce"],  imagesTr / f"{out_prefix}_0002.nii.gz")
        shutil.copy(src["t2"],    imagesTr / f"{out_prefix}_0003.nii.gz")

        # remap label 4 -> 3
        seg_img = nib.load(src["seg"])
        seg = seg_img.get_fdata()
        seg[seg == 4] = 3
        seg = seg.astype(np.uint8)

        new_seg = nib.Nifti1Image(seg, seg_img.affine, seg_img.header)
        new_seg.set_data_dtype(np.uint8)
        nib.save(new_seg, labelsTr / f"{out_prefix}.nii.gz")

        patient_count += 1

    # dataset.json
    dataset_json = {
        "dataset_name": "BraTS2020",
        "description": "Brain Tumor Segmentation 2020",
        "channel_names": {
            "0": "flair",
            "1": "t1",
            "2": "t1ce",
            "3": "t2",
        },
        "labels": {
            "background": 0,
            "NCR/NET": 1,
            "ED": 2,
            "ET": 3,
        },
        "numTraining": patient_count,
        "file_ending": ".nii.gz",
    }

    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Done. Converted {patient_count} cases.")
