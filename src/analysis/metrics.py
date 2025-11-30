# src/analysis/metrics.py
import numpy as np
from sklearn.metrics import auc
from .utils import get_binary_mask 

def compute_step_metrics(pred_sub, gt_sub, tp_total, tn_total):
    """Tính Dice, FTP, FTN tại một bước cắt cụ thể."""
    # 1. Dice Score
    inter = np.logical_and(pred_sub, gt_sub).sum()
    union = pred_sub.sum() + gt_sub.sum()
    dice = (2.0 * inter) / union if union > 0 else 1.0
    
    # 2. FTP Ratio
    tp_curr = inter
    ftp = (tp_total - tp_curr) / tp_total if tp_total > 0 else 0.0
    
    # 3. FTN Ratio
    tn_curr = np.logical_and(pred_sub == 0, gt_sub == 0).sum()
    ftn = (tn_total - tn_curr) / tn_total if tn_total > 0 else 0.0
    
    return dice, ftp, ftn

# --- [NEW] Hàm tính theo Ngưỡng Tuyệt đối (QU-BraTS Standard) ---
def compute_metrics_by_thresholds(pred_flat, gt_flat, unc_flat, target_class, thresholds):
    """
    Tính toán metrics dựa trên ngưỡng tuyệt đối (0-100).
    Input unc_flat: Đã được chuẩn hóa về 0-100.
    Thresholds: List các ngưỡng (ví dụ [100, 75, 50, 25, 0]).
    """
    try:
        pred_bin = get_binary_mask(pred_flat, target_class)
        gt_bin = get_binary_mask(gt_flat, target_class)
        n_pixels = len(pred_flat)
        
        # Tổng Global (Làm mốc so sánh)
        tp_total = np.logical_and(pred_bin, gt_bin).sum()
        union_all = np.logical_or(pred_bin, gt_bin).sum()
        tn_total = n_pixels - union_all

        results = {
            "thresholds": [],
            "dice": [],
            "ftp": [],
            "ftn": []
        }

        for tau in thresholds:
            # QU-BraTS Logic:
            # Giữ lại các pixel có Uncertainty < Tau
            # (Tức là: Bỏ đi các pixel có Uncertainty >= Tau)
            mask_keep = unc_flat < tau
            
            if mask_keep.sum() == 0:
                # Nếu lọc hết sạch thì Dice không xác định (hoặc = 1 tuỳ định nghĩa), FTP=1, FTN=1
                d, ftp, ftn = 1.0, 1.0, 1.0
            else:
                sub_pred = pred_bin[mask_keep]
                sub_gt = gt_bin[mask_keep]
                d, ftp, ftn = compute_step_metrics(sub_pred, sub_gt, tp_total, tn_total)

            results["thresholds"].append(tau)
            results["dice"].append(d)
            results["ftp"].append(ftp)
            results["ftn"].append(ftn)

        return results

    except Exception as e:
        print(f"Error in compute_metrics_by_thresholds: {e}")
        return None
    
# --- [UPDATED] Thêm tham số retention_range ---
def compute_curves_exact(pred_flat, gt_flat, unc_flat, target_class, retention_range=[1.0, 0.05], steps=20):
    """
    Tính toán đường cong AUSE với khoảng Retention tùy chỉnh.
    """
    try:
        # 1. Chuẩn bị Mask
        pred_bin = get_binary_mask(pred_flat, target_class)
        gt_bin = get_binary_mask(gt_flat, target_class)
        n_pixels = len(pred_flat)
        
        # 2. Global Totals
        tp_total = np.logical_and(pred_bin, gt_bin).sum()
        union_all = np.logical_or(pred_bin, gt_bin).sum()
        tn_total = n_pixels - union_all
        
        # 3. Sort Uncertainty
        sorted_indices = np.argsort(unc_flat)
        pred_sorted = pred_bin[sorted_indices]
        gt_sorted = gt_bin[sorted_indices]
        
        # Oracle Sort
        error_flat = (pred_bin != gt_bin).astype(int)
        sorted_indices_opt = np.argsort(error_flat)
        pred_opt = pred_bin[sorted_indices_opt]
        gt_opt = gt_bin[sorted_indices_opt]

        # 4. Loop theo Range Config
        start, end = retention_range # Lấy từ tham số truyền vào
        fractions = np.linspace(start, end, steps)
        
        dice_list, ftp_list, ftn_list, opt_dices = [], [], [], []
        
        for frac in fractions:
            n_keep = int(n_pixels * frac)
            if n_keep < 1: n_keep = 1
            
            # --- Model ---
            sub_pred = pred_sorted[:n_keep]
            sub_gt = gt_sorted[:n_keep]
            d, ftp, ftn = compute_step_metrics(sub_pred, sub_gt, tp_total, tn_total)
            
            dice_list.append(d)
            ftp_list.append(ftp)
            ftn_list.append(ftn)
            
            # --- Optimal ---
            sub_pred_opt = pred_opt[:n_keep]
            sub_gt_opt = gt_opt[:n_keep]
            inter_opt = np.logical_and(sub_pred_opt, sub_gt_opt).sum()
            union_opt = sub_pred_opt.sum() + sub_gt_opt.sum()
            d_opt = (2.0 * inter_opt) / union_opt if union_opt > 0 else 1.0
            opt_dices.append(d_opt)

        return fractions, np.array(dice_list), np.array(ftp_list), np.array(ftn_list), np.array(opt_dices)

    except Exception:
        return None, None, None, None, None

def calculate_auc_score(x_axis, y_axis):
    """Wrapper tính AUC an toàn"""
    if x_axis[0] > x_axis[-1]:
        return auc(x_axis[::-1], y_axis[::-1])
    return auc(x_axis, y_axis)