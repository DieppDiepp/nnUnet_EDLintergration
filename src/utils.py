"""
🛠️ UTILITIES MODULE (UPDATED V5 - BRATS STANDARD)
Chứa các hàm phụ trợ xử lý file và tính toán chỉ số theo chuẩn BraTS:
1. WT (Whole Tumor): Cả 3 lớp (1 U 2 U 3)
2. TC (Tumor Core):  Lớp 1 U 3 (Hoại tử + Lõi thuốc)
3. ET (Enhancing):   Lớp 3 (Lõi thuốc)
"""
import os
import json
import numpy as np

# Import HD95 an toàn (tránh lỗi nếu chưa cài thư viện)
try:
    from medpy.metric.binary import hd95
except ImportError:
    hd95 = None

def get_case_list(folder):
    """
    Lấy danh sách tất cả Case ID trong folder (Robust check).
    Hỗ trợ cả file .nii và .nii.gz
    """
    if not os.path.exists(folder):
        raise FileNotFoundError(f"❌ Folder {folder} không tồn tại!")
        
    files = sorted([f for f in os.listdir(folder) if f.endswith("_0000.nii") or f.endswith("_0000.nii.gz")])
    
    case_ids = []
    for f in files:
        # Xử lý string thông minh để lấy ID sạch
        if f.endswith(".nii"): cid = f.replace("_0000.nii", "")
        else: cid = f.replace("_0000.nii.gz", "")
        case_ids.append(cid)
        
    return case_ids

def get_validation_cases(split_file, fold=0):
    """
    Lấy danh sách validation từ file split json.
    Kiểm tra kỹ sự tồn tại của file để tránh crash.
    """
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"❌ Không tìm thấy file split tại: {split_file}. Hãy backup lại từ preprocessed!")
    
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    if fold >= len(splits):
        raise ValueError(f"❌ Fold {fold} không tồn tại trong file split!")
        
    val_keys = splits[fold]['val']
    print(f"📂 Đã load danh sách Validation Fold {fold}: {len(val_keys)} ca.")
    return val_keys

def calculate_dice_2d(pred_slice, gt_slice):
    """
    Tính Dice Score 2D nhanh (Dùng cho visualizer).
    """
    p = (pred_slice > 0).astype(np.float32)
    g = (gt_slice > 0).astype(np.float32)
    intersection = np.sum(p * g)
    sum_areas = np.sum(p) + np.sum(g)
    if sum_areas == 0: return 1.0
    return (2.0 * intersection) / sum_areas

def calculate_metric_binary(pred_mask, gt_mask, spacing):
    """
    Hàm phụ trợ cốt lõi: Tính Dice & HD95 cho 1 cặp mask nhị phân.
    Xử lý triệt để các trường hợp ngoại lệ (mask rỗng, thiếu thư viện).
    """
    # 1. Dice Score
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    sum_areas = pred_mask.sum() + gt_mask.sum()
    
    if sum_areas == 0:
        dice = 1.0 # Cả 2 đều trống -> Đúng tuyệt đối
    else:
        dice = (2.0 * intersection) / sum_areas
        
    # 2. HD95 (Hausdorff Distance 95%)
    if hd95 is None:
        hd_val = np.nan # Chưa cài medpy
    elif pred_mask.sum() == 0 or gt_mask.sum() == 0:
        # Nếu 1 trong 2 mask rỗng thì khoảng cách là vô tận (hoặc không xác định)
        # Để an toàn cho tính toán trung bình, ta gán NaN hoặc 1 giá trị phạt lớn (tùy chọn)
        hd_val = np.nan 
    else:
        try:
            # voxelspacing=None nghĩa là tính theo pixel, có spacing thì ra mm
            hd_val = hd95(pred_mask, gt_mask, voxelspacing=spacing)
        except Exception:
            hd_val = np.nan # Lỗi tính toán hình học
            
    return dice, hd_val

def calculate_metric_per_class(pred, gt, spacing=None):
    """
    Tính Metrics theo 3 vùng chuẩn BraTS: WT, TC, ET.
    Đây là chuẩn quốc tế để so sánh hiệu quả mô hình.
    
    Mapping (Giả định theo nnU-Net BraTS):
    - Label 1: Necrotic (Hoại tử)
    - Label 2: Edema (Phù nề)
    - Label 3: Enhancing (Lõi thuốc)
    """
    results = {}
    
    # --- 1. WT (Whole Tumor): Tất cả các lớp cộng lại (Label > 0) ---
    mask_pred_WT = (pred > 0)
    mask_gt_WT   = (gt > 0)
    d_wt, h_wt = calculate_metric_binary(mask_pred_WT, mask_gt_WT, spacing)
    results["Dice_WT"] = d_wt
    results["HD95_WT"] = h_wt
    
    # --- 2. TC (Tumor Core): Lớp 1 (NCR) + Lớp 3 (ET) ---
    # Lưu ý: Class 2 là Edema (Phù nề) nằm ngoài Core
    mask_pred_TC = np.logical_or(pred == 1, pred == 3)
    mask_gt_TC   = np.logical_or(gt == 1, gt == 3)
    d_tc, h_tc = calculate_metric_binary(mask_pred_TC, mask_gt_TC, spacing)
    results["Dice_TC"] = d_tc
    results["HD95_TC"] = h_tc
    
    # --- 3. ET (Enhancing Tumor): Chỉ Lớp 3 ---
    mask_pred_ET = (pred == 3)
    mask_gt_ET   = (gt == 3)
    d_et, h_et = calculate_metric_binary(mask_pred_ET, mask_gt_ET, spacing)
    results["Dice_ET"] = d_et
    results["HD95_ET"] = h_et
    
    # --- Tính Mean Dice (Trung bình cộng 3 chỉ số quan trọng này) ---
    # Đây là con số tổng hợp hay dùng để so sánh nhanh
    results["Mean_Dice"] = (d_wt + d_tc + d_et) / 3.0
        
    return results