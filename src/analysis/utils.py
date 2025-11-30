# src/analysis/utils.py
import os
import nibabel as nib
import numpy as np

def load_nifti_safe(path):
    """Load nifti an toàn, trả về mảng phẳng (flattened) hoặc None nếu lỗi."""
    try:
        if not os.path.exists(path):
            return None
        data = nib.load(path).get_fdata()
        return data.flatten()
    except Exception:
        return None

def get_binary_mask(data, target_class):
    """Chuyển mask đa lớp sang nhị phân theo vùng BraTS."""
    # WT: Whole Tumor (Tất cả > 0)
    if target_class == 'WT': 
        return (data > 0)
    # TC: Tumor Core (1: Necrotic + 3: Enhancing)
    elif target_class == 'TC': 
        return np.isin(data, [1, 3])
    # ET: Enhancing Tumor (3)
    elif target_class == 'ET': 
        return (data == 3)
    # Mặc định WT
    return (data > 0)