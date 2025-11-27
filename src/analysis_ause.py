"""
📊 AUSE ANALYSIS MODULE (FINAL MULTI-CLASS V5)
Tính toán Risk-Coverage Curve và AUSE cho từng vùng BraTS (WT, TC, ET).
Đảm bảo tính đúng đắn về logic BraTS và an toàn về code.
"""
import sys
import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# --- 1. ROBUST PATH SETUP ---
# Tự động thêm thư mục gốc project vào sys.path để import config không lỗi
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
except NameError:
    pass # Fallback nếu chạy interactive

# --- 2. CONFIG IMPORT ---
try:
    from src.config import BASE_CONFIG, MODEL_CONFIGS
except ImportError:
    print("❌ Critical Error: Không thể import 'src.config'. Check lại đường dẫn!")
    sys.exit(1)

# ==============================================================================
# 3. CORE FUNCTIONS
# ==============================================================================

def load_nifti_safe(path):
    """Load nifti an toàn, trả về None nếu lỗi"""
    try:
        if not os.path.exists(path): return None
        data = nib.load(path).get_fdata()
        return data.flatten()
    except Exception:
        return None

def get_binary_mask(data, target_class):
    """
    Chuyển đổi Mask đa lớp sang nhị phân theo chuẩn BraTS Regions.
    Input data: Mảng 1D chứa các label (0, 1, 2, 3)
    """
    if target_class == 'WT':   # Whole Tumor: Tất cả (1, 2, 3)
        return (data > 0)
    elif target_class == 'TC': # Tumor Core: Hoại tử (1) + Lõi thuốc (3)
        return np.isin(data, [1, 3])
    elif target_class == 'ET': # Enhancing Tumor: Lớp 3
        return (data == 3)
    else:
        return (data > 0) # Mặc định WT

def compute_dice_score_binary(pred_bin, gt_bin):
    """Tính Dice 1D nhanh"""
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    sum_areas = pred_bin.sum() + gt_bin.sum()
    if sum_areas == 0: return 1.0
    return (2.0 * intersection) / sum_areas

def compute_risk_coverage_curve(pred_flat, gt_flat, unc_flat, target_class, steps=20):
    """Tính đường cong Risk-Coverage cho Class cụ thể"""
    try:
        # 1. Tạo mask nhị phân theo Class
        pred_bin = get_binary_mask(pred_flat, target_class)
        gt_bin = get_binary_mask(gt_flat, target_class)
        
        # 2. Sắp xếp theo Uncertainty tăng dần (Confident -> Uncertain)
        sorted_indices = np.argsort(unc_flat)
        
        n_pixels = len(pred_flat)
        fractions = np.linspace(1.0, 0.05, steps)
        
        dice_list = []
        retention_list = []
        
        # Pre-sort mảng để loop nhanh hơn
        pred_sorted = pred_bin[sorted_indices]
        gt_sorted = gt_bin[sorted_indices]
        
        for frac in fractions:
            n_keep = int(n_pixels * frac)
            if n_keep < 1: n_keep = 1
            
            # Giữ lại n_keep pixel đầu tiên
            sub_pred = pred_sorted[:n_keep]
            sub_gt = gt_sorted[:n_keep]
            
            d = compute_dice_score_binary(sub_pred, sub_gt)
            dice_list.append(d)
            retention_list.append(frac)
            
        return np.array(retention_list), np.array(dice_list)
    except Exception:
        return None, None

def compute_optimal_curve(pred_flat, gt_flat, target_class, steps=20):
    """Tính đường Optimal (Oracle)"""
    try:
        pred_bin = get_binary_mask(pred_flat, target_class)
        gt_bin = get_binary_mask(gt_flat, target_class)
        
        # Error map: 0 (Đúng), 1 (Sai). Sort tăng dần để giữ lại số 0.
        error_flat = (pred_bin != gt_bin).astype(int)
        sorted_indices = np.argsort(error_flat) 
        
        pred_sorted = pred_bin[sorted_indices]
        gt_sorted = gt_bin[sorted_indices]
        
        n_pixels = len(pred_flat)
        fractions = np.linspace(1.0, 0.05, steps)
        dice_opt_list = []
        
        for frac in fractions:
            n_keep = int(n_pixels * frac)
            if n_keep < 1: n_keep = 1
            
            sub_pred = pred_sorted[:n_keep]
            sub_gt = gt_sorted[:n_keep]
            dice_opt_list.append(compute_dice_score_binary(sub_pred, sub_gt))
            
        return np.array(dice_opt_list)
    except Exception:
        return None

# ==============================================================================
# 4. MAIN PIPELINE
# ==============================================================================

def run_analysis_pipeline(mode='edl', n_cases=None):
    print(f"🚀 BẮT ĐẦU PHÂN TÍCH AUSE ĐA LỚP (WT, TC, ET) | MODE: {mode.upper()}")
    
    # --- A. CONFIG PATHS ---
    try:
        if mode not in MODEL_CONFIGS: raise ValueError(f"Mode '{mode}' không hợp lệ.")
        model_cfg = MODEL_CONFIGS[mode]
        base_folder = model_cfg["output_folder"]
        
        nifti_dir_name = BASE_CONFIG.get("dir_nifti", "3d_nifti")
        input_folder = os.path.join(base_folder, nifti_dir_name)
        output_folder = os.path.join(base_folder, "analysis_ause")
        os.makedirs(output_folder, exist_ok=True)
        
        if not os.path.exists(input_folder): 
            raise FileNotFoundError(f"Input not found: {input_folder}")
            
    except Exception as e:
        print(f"❌ Lỗi cấu hình: {e}")
        return

    # --- B. SCAN CASES ---
    all_cases = sorted([d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))])
    if n_cases and n_cases > 0: 
        all_cases = all_cases[:n_cases]
    
    if not all_cases:
        print("❌ Không tìm thấy ca bệnh nào.")
        return
    print(f"🔍 Tìm thấy {len(all_cases)} ca.")

    summary_ause = []
    TARGET_CLASSES = ["WT", "TC", "ET"]
    
    # --- C. LOOP QUA TỪNG LỚP (WT -> TC -> ET) ---
    for target in TARGET_CLASSES:
        print(f"\n⚡ Đang phân tích lớp: {target}...")
        
        curves_data = {"total": [], "aleatoric": [], "epistemic": [], "optimal": []}
        valid_retention = None
        
        # Loop qua từng ca
        for case_id in tqdm(all_cases, desc=f"Processing {target}"):
            try:
                case_path = os.path.join(input_folder, case_id)
                pred = load_nifti_safe(os.path.join(case_path, "prediction.nii.gz"))
                gt = load_nifti_safe(os.path.join(case_path, "ground_truth.nii.gz"))
                
                if pred is None or gt is None: continue

                # ROI Mask: Chỉ xét vùng có u (Pred hoặc GT) của Class đang xét
                # Để tránh tính toán hàng triệu pixel nền đen
                bin_pred = get_binary_mask(pred, target)
                bin_gt = get_binary_mask(gt, target)
                roi_mask = np.logical_or(bin_pred, bin_gt)
                
                if roi_mask.sum() == 0: continue # Ca này sạch, không có u loại này

                pred_roi = pred[roi_mask]
                gt_roi = gt[roi_mask]
                
                # 1. Tính Optimal
                d_opt = compute_optimal_curve(pred_roi, gt_roi, target)
                if d_opt is not None: 
                    curves_data["optimal"].append(d_opt)
                
                # 2. Tính Uncertainty Curves
                for u_type in ["total", "aleatoric", "epistemic"]:
                    u_path = os.path.join(case_path, f"unc_{u_type}.nii.gz")
                    u_data = load_nifti_safe(u_path)
                    
                    if u_data is not None:
                        u_roi = u_data[roi_mask]
                        ret, d_curve = compute_risk_coverage_curve(pred_roi, gt_roi, u_roi, target)
                        
                        if d_curve is not None:
                            curves_data[u_type].append(d_curve)
                            if valid_retention is None: valid_retention = ret
                            
            except Exception:
                continue # Bỏ qua ca lỗi để chạy ca tiếp theo

        # --- D. VẼ BIỂU ĐỒ & TÍNH AUSE ---
        if not curves_data["optimal"]:
            print(f"⚠️ Không có dữ liệu hợp lệ cho lớp {target} (Có thể do mask rỗng).")
            continue

        plt.figure(figsize=(10, 6))
        mean_opt = np.mean(curves_data["optimal"], axis=0)
        plt.plot(valid_retention, mean_opt, 'k--', label='Optimal (Ideal)', linewidth=2, alpha=0.8)

        colors = {'total': 'r', 'aleatoric': 'g', 'epistemic': 'b'}
        current_ause_scores = {"Class": target}

        for u_type in ["total", "aleatoric", "epistemic"]:
            if curves_data[u_type]:
                mean_curve = np.mean(curves_data[u_type], axis=0)
                
                # Tính AUSE (Diện tích sai số)
                ause = np.trapz(mean_opt - mean_curve, dx=1.0/len(valid_retention))
                current_ause_scores[f"AUSE_{u_type.capitalize()}"] = ause
                
                plt.plot(valid_retention, mean_curve, color=colors[u_type], 
                         label=f'{u_type.capitalize()} (AUSE={ause:.4f})')

        # Trang trí biểu đồ
        plt.gca().invert_xaxis() # 1.0 -> 0.0
        plt.xlabel("Retention Fraction (Tỷ lệ pixel giữ lại)")
        plt.ylabel(f"Dice Score ({target})")
        plt.title(f"Risk-Coverage Curve: {target} Region")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Lưu ảnh
        save_name = f"risk_coverage_{target}.png"
        plt.savefig(os.path.join(output_folder, save_name), dpi=300)
        plt.close()
        
        summary_ause.append(current_ause_scores)
        print(f"✅ Đã lưu biểu đồ: {save_name}")

    # --- E. LƯU CSV TỔNG HỢP ---
    if summary_ause:
        df = pd.DataFrame(summary_ause)
        # Sắp xếp cột
        cols = ["Class", "AUSE_Total", "AUSE_Aleatoric", "AUSE_Epistemic"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        csv_path = os.path.join(output_folder, "ause_scores_summary.csv")
        df.to_csv(csv_path, index=False)
        
        print("\n" + "="*60)
        print(f"{'📊 TỔNG HỢP AUSE (Càng thấp càng tốt)':^60}")
        print("-" * 60)
        print(df.to_string(index=False))
        print("-" * 60)
        print(f"✅ File CSV tổng hợp: {csv_path}")

# ==============================================================================
# 5. EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='edl', choices=['edl', 'baseline'])
    parser.add_argument('--limit', type=int, default=0)
    
    # Xử lý an toàn cho Jupyter/Colab
    if 'ipykernel' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    run_analysis_pipeline(mode=args.mode, n_cases=args.limit)