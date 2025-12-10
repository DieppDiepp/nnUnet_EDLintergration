# src/experiment_ood/utils.py
import numpy as np
import nibabel as nib

def get_roi_center(data):
    """Tìm tâm trọng lượng của vùng ROI (Non-zero)."""
    coords = np.argwhere(data > 0)
    if coords.size == 0:
        return np.array(data.shape) // 2
    
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    return (min_coords + max_coords) // 2

def add_artifact(data, gt_data=None, type='Box_White'):
    """
    Vẽ dị vật đè lên vị trí quan trọng nhất (Tumor nếu có, hoặc Brain Center).
    """
    modified_data = data.copy()
    shape = data.shape
    
    # 1. Xác định vị trí đặt (Targeting Strategy)
    if gt_data is not None and np.sum(gt_data) > 0:
        # ƯU TIÊN TUYỆT ĐỐI: Đặt ngay giữa khối u
        # (Để đảm bảo Visualizer chụp dính nó)
        center = get_roi_center(gt_data)
        # print(f"   🎯 Targeted Tumor Center: {center}")
    else:
        # Fallback: Đặt giữa não
        center = get_roi_center(data)
        # print(f"   📍 Targeted Brain Center: {center}")

    cx, cy = center[0], center[1]
    
    # 2. Xác định giá trị (Contrast)
    max_val = np.max(data)
    if max_val == 0: max_val = 1.0
    
    # Tăng độ sáng lên nữa để chống Clipping
    val = max_val * 4.0 if "White" in type else max_val * 0.5
    
    # 3. Kích thước (To lên chút nữa)
    xy_size = 20     # To hơn (40px)
    border_size = 4  # Viền dày hơn
    
    # Full Z-Axis (Giữ nguyên chiến thuật Beam xuyên thấu)
    z_start, z_end = 0, shape[2]
    
    # 4. Vẽ
    if "Box" in type:
        # A. Hộp đen (Viền)
        bx_start, bx_end = max(0, cx - xy_size - border_size), min(shape[0], cx + xy_size + border_size)
        by_start, by_end = max(0, cy - xy_size - border_size), min(shape[1], cy + xy_size + border_size)
        modified_data[bx_start:bx_end, by_start:by_end, z_start:z_end] = 0.0 # BLACK HOLE
        
        # B. Hộp trắng (Lõi)
        x_start, x_end = max(0, cx - xy_size), min(shape[0], cx + xy_size)
        y_start, y_end = max(0, cy - xy_size), min(shape[1], cy + xy_size)
        modified_data[x_start:x_end, y_start:y_end, z_start:z_end] = val

    elif "Sphere" in type:
        x_grid, y_grid = np.ogrid[:shape[0], :shape[1]]
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        
        mask_border = dist_sq <= (xy_size + border_size)**2
        mask_core = dist_sq <= xy_size**2
        
        # Broadcasting cho nhanh
        for z in range(z_start, z_end):
             modified_data[mask_border, z] = 0.0
             modified_data[mask_core, z] = val
        
    return modified_data

# ... (Giữ nguyên các hàm khác)
def apply_structural_mutation(data, type='Flip_Horizontal'):
    if type == 'Flip_Horizontal': return np.flip(data, axis=0) 
    elif type == 'Flip_Vertical': return np.flip(data, axis=1)
    return data

def apply_intensity_shift(data, factor):
    return data * float(factor)

def save_temp_nifti(data, affine, path):
    try:
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        nib.save(img, path)
    except: pass