"""
MAIN ANALYSIS SCRIPT (REFACTORED V13 - FINAL QU-BRATS STANDARD)
Chạy phân tích QU-BraTS Score và AUSE với chuẩn hóa 0-100.
Tích hợp logic tính toán chính xác và vẽ biểu đồ.
"""
import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings

# --- SETUP PATH (Lùi 2 bước) ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(src_dir)
    if project_root not in sys.path: sys.path.insert(0, project_root)
except NameError: pass

# --- IMPORTS ---
try:
    from src.config import BASE_CONFIG, MODEL_CONFIGS
    from src.analysis.utils import load_nifti_safe, get_binary_mask
    from src.analysis.metrics import compute_metrics_by_thresholds, calculate_auc_score
    # Lưu ý: Cần cập nhật plotting.py để hỗ trợ vẽ theo threshold nếu muốn, 
    # nhưng ở đây ta sẽ dùng matplotlib trực tiếp cho đơn giản hoặc cập nhật sau.
    # Tạm thời ta sẽ vẽ trực tiếp trong file này để kiểm soát tốt hơn.
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Gợi ý: Đảm bảo bạn đã tạo đúng cấu trúc folder src/analysis/...")
    sys.exit(1)

import matplotlib.pyplot as plt # Import lại để vẽ

warnings.filterwarnings("ignore")

def normalize_uncertainty(unc_map):
    """Chuẩn hóa Uncertainty về khoảng [0, 100]"""
    u_min = unc_map.min()
    u_max = unc_map.max()
    if u_max == u_min:
        return np.zeros_like(unc_map)
    # Công thức Min-Max Scaling * 100
    return ((unc_map - u_min) / (u_max - u_min)) * 100.0

def run_analysis_pipeline(mode='edl', n_cases=None, eval_set='test'):
    print(f"🚀 STARTING ANALYSIS (QU-BRATS) | MODE: {mode.upper()} | SET: {eval_set.upper()}")
    
    # 1. Config & Paths
    try:
        model_cfg = MODEL_CONFIGS[mode]
        
        # --- 2. SỬA ĐƯỜNG DẪN Ở ĐÂY: Nối thêm eval_set vào sau output_folder ---
        base_folder = os.path.join(model_cfg["output_folder"], eval_set) 
        
        nifti_dir = os.path.join(base_folder, BASE_CONFIG.get("dir_nifti", "3d_nifti"))
        output_dir = os.path.join(base_folder, "analysis_qu_brats_v13") 
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(nifti_dir): raise FileNotFoundError(f"Input missing: {nifti_dir}")
    except Exception as e:
        print(f"❌ Config Error: {e}"); return

    all_cases = sorted([d for d in os.listdir(nifti_dir) if os.path.isdir(os.path.join(nifti_dir, d))])
    if n_cases: all_cases = all_cases[:n_cases]
    print(f"🔍 Found {len(all_cases)} cases.")

    final_report = []
    TARGET_CLASSES = ["WT", "TC", "ET"]
    
    # Định nghĩa các ngưỡng Threshold (0-100)
    THRESHOLDS = np.arange(100, -1, -1) # 100, 95, ..., 0

    # 2. Main Loop
    for target in TARGET_CLASSES:
        print(f"\n📊 Class: {target}")
        
        agg_data = {
            "total_dice": [], "total_ftp": [], "total_ftn": [],
            "aleatoric_dice": [], "epistemic_dice": []
        }

        for case_id in tqdm(all_cases, desc=f"Processing"):
            try:
                case_path = os.path.join(nifti_dir, case_id)
                pred = load_nifti_safe(os.path.join(case_path, "prediction.nii.gz"))
                gt = load_nifti_safe(os.path.join(case_path, "ground_truth.nii.gz"))
                mri = load_nifti_safe(os.path.join(case_path, "mri_crop.nii.gz"))

                if pred is None or gt is None: continue

                # Brain Masking
                brain_mask = (mri > 0) if mri is not None else np.logical_or(pred > 0, gt > 0)
                if brain_mask.sum() == 0: continue

                pred_roi = pred[brain_mask]
                gt_roi = gt[brain_mask]

                # --- Total Uncertainty ---
                u_total = load_nifti_safe(os.path.join(case_path, "unc_total.nii.gz"))
                if u_total is not None:
                    # 1. Chuẩn hóa
                    u_norm = normalize_uncertainty(u_total[brain_mask])
                    
                    # 2. Tính theo Thresholds (Gọi hàm từ metrics.py)
                    res = compute_metrics_by_thresholds(
                        pred_roi, gt_roi, u_norm, target, THRESHOLDS
                    )
                    
                    if res is not None:
                        agg_data["total_dice"].append(res["dice"])
                        agg_data["total_ftp"].append(res["ftp"])
                        agg_data["total_ftn"].append(res["ftn"])

                # --- Components ---
                for u_name in ["aleatoric", "epistemic"]:
                    u_map = load_nifti_safe(os.path.join(case_path, f"unc_{u_name}.nii.gz"))
                    if u_map is not None:
                        u_norm = normalize_uncertainty(u_map[brain_mask])
                        res = compute_metrics_by_thresholds(
                            pred_roi, gt_roi, u_norm, target, THRESHOLDS
                        )
                        if res is not None: agg_data[f"{u_name}_dice"].append(res["dice"])

            except Exception: continue

        if not agg_data["total_dice"]: 
            print(f"⚠️ No data for {target}")
            continue

        # 3. Aggregation & Metrics
        # Tính trung bình theo cột (axis=0)
        mean_dice = np.mean(agg_data["total_dice"], axis=0)
        mean_ftp = np.mean(agg_data["total_ftp"], axis=0)
        mean_ftn = np.mean(agg_data["total_ftn"], axis=0)

        # QU-BraTS Score Calculation
        # Lưu ý: x_axis ở đây là THRESHOLDS / 100 để về range [0, 1] cho AUC
        # Nhưng THRESHOLDS đang giảm dần (100 -> 0), AUC cần x tăng dần -> Đảo chiều
        x_norm = THRESHOLDS[::-1] / 100.0 
        
        auc_dice = calculate_auc_score(x_norm, mean_dice[::-1])
        auc_ftp = calculate_auc_score(x_norm, mean_ftp[::-1])
        auc_ftn = calculate_auc_score(x_norm, mean_ftn[::-1])
        
        qu_score = (auc_dice + (1 - auc_ftp) + (1 - auc_ftn)) / 3.0
        
        # Component AUCs (Chỉ tính Dice AUC để so sánh)
        auc_dice_alea = 0
        if agg_data["aleatoric_dice"]:
            mean_alea = np.mean(agg_data["aleatoric_dice"], axis=0)
            auc_dice_alea = calculate_auc_score(x_norm, mean_alea[::-1])
            
        auc_dice_epis = 0
        if agg_data["epistemic_dice"]:
            mean_epis = np.mean(agg_data["epistemic_dice"], axis=0)
            auc_dice_epis = calculate_auc_score(x_norm, mean_epis[::-1])

        final_report.append({
            "Class": target, 
            "QU_Score": qu_score,
            "AUC_Dice_Total": auc_dice,
            "AUC_Dice_Aleatoric": auc_dice_alea,
            "AUC_Dice_Epistemic": auc_dice_epis,
            "AUC_FTP": auc_ftp,
            "AUC_FTN": auc_ftn
        })

        # 4. Plotting (Vẽ trực tiếp tại đây)
        plt.figure(figsize=(10, 6))
        plt.plot(THRESHOLDS, mean_dice, 'r-', label=f'Total (AUC={auc_dice:.3f})', linewidth=2)
        
        if agg_data["aleatoric_dice"]:
            mean_alea = np.mean(agg_data["aleatoric_dice"], axis=0)
            plt.plot(THRESHOLDS, mean_alea, 'g:', label=f'Aleatoric (AUC={auc_dice_alea:.3f})')
            
        if agg_data["epistemic_dice"]:
            mean_epis = np.mean(agg_data["epistemic_dice"], axis=0)
            plt.plot(THRESHOLDS, mean_epis, 'b-.', label=f'Epistemic (AUC={auc_dice_epis:.3f})')

        plt.xlabel("Uncertainty Threshold (τ)", fontsize=12)
        plt.ylabel(f"Dice Score ({target})", fontsize=12)
        plt.title(f"Performance vs. Uncertainty Threshold ({target})\nScore={qu_score:.3f}", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis() # 100 -> 0
        
        plt.savefig(os.path.join(output_dir, f"qu_brats_curve_{target}.png"), dpi=300)
        plt.close()

    # 5. Final Report
    print("\n" + "="*80)
    print(f"{'🏆 QU-BRATS METRICS REPORT (V13)':^80}")
    print("="*80)
    df = pd.DataFrame(final_report)
    if not df.empty:
        cols = ["Class", "QU_Score", "AUC_Dice_Total", "AUC_FTP", "AUC_FTN"]
        print(df[cols].to_string(index=False, float_format="%.4f"))
        df.to_csv(os.path.join(output_dir, "final_qu_metrics.csv"), index=False)
        print(f"\n✅ Results saved to: {output_dir}")
    else:
        print("❌ No results computed.")
    print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='edl', 
                        choices=['edl', 'edl_raw', 'edl_250', 'edl_250_fixed_split', 'edl_50_fixed_split'])
    
    parser.add_argument('--limit', type=int, default=0)
    
    # --- THÊM DÒNG NÀY: Khai báo thư mục cần phân tích ---
    parser.add_argument('--eval_set', type=str, default='test', choices=['val', 'test', 'other'],
                        help="Chọn thư mục chứa kết quả inference (val, test, hoặc other)")
    
    if 'ipykernel' in sys.modules: args = parser.parse_args([])
    else: args = parser.parse_args()
    
    # --- TRUYỀN THÊM BIẾN eval_set VÀO HÀM ---
    run_analysis_pipeline(mode=args.mode, n_cases=args.limit, eval_set=args.eval_set)