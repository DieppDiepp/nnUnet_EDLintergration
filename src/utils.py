"""
🛠️ UTILITIES MODULE
Chứa các hàm phụ trợ xử lý file và tính toán chỉ số.
"""
import os
import json
import numpy as np
try:
    from medpy.metric.binary import hd95
except ImportError:
    hd95 = None

def get_case_list(folder):
    """Lấy danh sách tất cả Case ID trong folder"""
    if not os.path.exists(folder):
        raise FileNotFoundError(f"❌ Folder {folder} không tồn tại!")
    files = sorted([f for f in os.listdir(folder) if f.endswith("_0000.nii") or f.endswith("_0000.nii.gz")])
    case_ids = []
    for f in files:
        if f.endswith(".nii"): cid = f.replace("_0000.nii", "")
        else: cid = f.replace("_0000.nii.gz", "")
        case_ids.append(cid)
    return case_ids

def get_validation_cases(split_file, fold=0):
    """Lấy danh sách validation từ file split json"""
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"❌ Không tìm thấy file split tại: {split_file}. Hãy backup lại từ preprocessed!")
    
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    if fold >= len(splits):
        raise ValueError(f"❌ Fold {fold} không tồn tại!")
        
    val_keys = splits[fold]['val']
    print(f"📂 Đã load danh sách Validation Fold {fold}: {len(val_keys)} ca.")
    return val_keys

def calculate_dice(pred_slice, gt_slice):
    """Tính Dice Score 2D (Dùng cho visualizer)"""
    p = (pred_slice > 0).astype(np.float32)
    g = (gt_slice > 0).astype(np.float32)
    intersection = np.sum(p * g)
    sum_areas = np.sum(p) + np.sum(g)
    if sum_areas == 0: return 1.0
    return (2.0 * intersection) / sum_areas

def calculate_metric_per_class(pred, gt, spacing=None):
    """
    Tính Dice & HD95 cho từng lớp trong BraTS.
    Giả sử nhãn BraTS: 0 (Bg), 1 (Necrotic), 2 (Edema), 3 (Enhancing).
    """
    results = {}
    # Tìm các class có trong GT (trừ nền 0)
    classes = np.unique(gt)
    classes = [c for c in classes if c != 0]
    
    # Nếu muốn cố định 3 lớp chuẩn BraTS:
    target_classes = [1, 2, 3] 
    
    for c in target_classes:
        # Tạo mask nhị phân cho class c
        p_c = (pred == c)
        g_c = (gt == c)
        
        # 1. Dice
        intersection = np.logical_and(p_c, g_c).sum()
        sum_areas = p_c.sum() + g_c.sum()
        if sum_areas == 0:
            dice = 1.0 if g_c.sum() == 0 else 0.0
        else:
            dice = (2.0 * intersection) / sum_areas
            
        # 2. HD95
        if hd95 is None:
            hd_val = np.nan
        elif p_c.sum() == 0 or g_c.sum() == 0:
            hd_val = np.nan # Không tính được nếu 1 bên rỗng
        else:
            try:
                hd_val = hd95(p_c, g_c, voxelspacing=spacing)
            except:
                hd_val = np.nan
                
        results[f"Class_{c}_Dice"] = dice
        results[f"Class_{c}_HD95"] = hd_val
        
    # Tính Mean (cho các class có mặt)
    dices = [v for k, v in results.items() if "Dice" in k]
    if dices:
        results["Mean_Dice"] = np.mean(dices)
        
    return results


# def compute_metrics_3d(pred, gt, spacing=None):
#     """
#     Tính Dice và HD95 cho khối 3D.
#     - pred, gt: Numpy array (H, W, D)
#     - spacing: Tuple (z, y, x) resolution để tính HD95 ra mm
#     """
#     # Chuyển về Boolean
#     p = (pred > 0)
#     g = (gt > 0)
    
#     # 1. Dice Score
#     intersection = np.logical_and(p, g).sum()
#     sum_areas = p.sum() + g.sum()
    
#     if sum_areas == 0:
#         dice = 1.0 # Cả 2 đều trống
#     else:
#         dice = (2.0 * intersection) / sum_areas
        
#     # 2. HD95
#     # Xử lý trường hợp ngoại lệ cho HD95
#     if p.sum() == 0 and g.sum() == 0:
#         hd95_val = 0.0
#     elif p.sum() == 0 or g.sum() == 0:
#         hd95_val = np.nan # Không tính được nếu 1 trong 2 rỗng (lệch vô cùng)
#     else:
#         try:
#             # spacing trong properties thường là [z, y, x] hoặc [x, y, z] tùy format
#             # medpy cần spacing để ra mm. Nếu None thì tính theo voxels.
#             hd95_val = hd95(p, g, voxelspacing=spacing)
#         except Exception as e:
#             print(f"    ⚠️ HD95 Error: {e}")
#             hd95_val = np.nan
            
#     return dice, hd95_val