"""
🕵️‍♂️ SCAN STRATEGY EXPERIMENT (FIXED IMPORT)
Giả lập chiến thuật của Team SCAN: Ép độ bất định về 0 tại vùng dự đoán dương tính.
"""
import sys
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. QUAN TRỌNG: SETUP PATH ĐỂ PYTHON HIỂU 'src' ---
try:
    # Lấy đường dẫn file hiện tại (.../src/run_scan_experiment.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Lấy thư mục gốc dự án (.../XUM_project)
    project_root = os.path.dirname(current_dir)
    # Thêm vào sys.path
    if project_root not in sys.path: sys.path.append(project_root)
except NameError: pass
# -------------------------------------------------------

# Giờ mới import được các module trong src
from src.config import BASE_CONFIG, MODEL_CONFIGS
from src.analysis.utils import load_nifti_safe, get_binary_mask
from src.analysis.metrics import compute_metrics_by_thresholds, calculate_auc_score

def normalize_uncertainty(unc_map):
    """Chuẩn hóa 0-100"""
    u_min, u_max = unc_map.min(), unc_map.max()
    if u_max == u_min: return np.zeros_like(unc_map)
    return ((unc_map - u_min) / (u_max - u_min)) * 100.0

def run_scan_experiment(mode='edl', n_cases=None):
    print(f"🧪 BẮT ĐẦU THÍ NGHIỆM: TEAM SCAN STRATEGY | MODE: {mode.upper()}")
    
    # --- Setup ---
    try:
        model_cfg = MODEL_CONFIGS[mode]
        base_folder = model_cfg["output_folder"]
        nifti_dir = os.path.join(base_folder, BASE_CONFIG.get("dir_nifti", "3d_nifti"))
        # Lưu vào folder riêng
        output_dir = os.path.join(base_folder, "experiment_scan_strategy")
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(nifti_dir): raise FileNotFoundError("Input missing")
    except Exception as e: print(f"❌ Error: {e}"); return

    all_cases = sorted([d for d in os.listdir(nifti_dir) if os.path.isdir(os.path.join(nifti_dir, d))])
    if n_cases: all_cases = all_cases[:n_cases]
    
    final_report = []
    TARGET_CLASSES = ["WT", "TC", "ET"]
    THRESHOLDS = np.arange(100, -1, -5)

    for target in TARGET_CLASSES:
        print(f"\n📊 Simulating SCAN Strategy for: {target}")
        
        agg_data = {"dice": [], "ftp": [], "ftn": []}

        for case_id in tqdm(all_cases, desc="Processing"):
            try:
                case_path = os.path.join(nifti_dir, case_id)
                pred = load_nifti_safe(os.path.join(case_path, "prediction.nii.gz"))
                gt = load_nifti_safe(os.path.join(case_path, "ground_truth.nii.gz"))
                mri = load_nifti_safe(os.path.join(case_path, "mri_crop.nii.gz"))

                if pred is None or gt is None: continue
                
                brain_mask = (mri > 0) if mri is not None else np.logical_or(pred > 0, gt > 0)
                if brain_mask.sum() == 0: continue

                pred_roi = pred[brain_mask]
                gt_roi = gt[brain_mask]
                
                # Load Total Uncertainty gốc
                u_total = load_nifti_safe(os.path.join(case_path, "unc_total.nii.gz"))
                
                if u_total is not None:
                    u_roi = u_total[brain_mask]
                    u_norm = normalize_uncertainty(u_roi)
                    
                    # ======================================================
                    # ⚡ THE SCAN TRICK
                    # ======================================================
                    pred_binary = get_binary_mask(pred_roi, target)
                    u_scan = u_norm.copy()
                    
                    # Ép Uncertainty = 0 tại tất cả pixel dự đoán là U
                    u_scan[pred_binary] = 0.0 
                    # ======================================================

                    # Tính toán lại
                    res = compute_metrics_by_thresholds(
                        pred_roi, gt_roi, u_scan, target, THRESHOLDS
                    )
                    
                    if res is not None:
                        agg_data["dice"].append(res["dice"])
                        agg_data["ftp"].append(res["ftp"])
                        agg_data["ftn"].append(res["ftn"])

            except Exception: continue

        if not agg_data["dice"]: continue

        # Tính Score
        mean_dice = np.mean(agg_data["dice"], axis=0)
        mean_ftp = np.mean(agg_data["ftp"], axis=0)
        mean_ftn = np.mean(agg_data["ftn"], axis=0)

        x_norm = THRESHOLDS[::-1] / 100.0
        auc_dice = calculate_auc_score(x_norm, mean_dice[::-1])
        auc_ftp = calculate_auc_score(x_norm, mean_ftp[::-1])
        auc_ftn = calculate_auc_score(x_norm, mean_ftn[::-1])
        
        qu_score = (auc_dice + (1 - auc_ftp) + (1 - auc_ftn)) / 3.0

        final_report.append({
            "Class": target, 
            "SCAN_Score": qu_score,
            "AUC_FTP": auc_ftp,
            "AUC_FTN": auc_ftn,
            "AUC_Dice": auc_dice
        })
        
        # Vẽ biểu đồ
        plt.figure(figsize=(10, 6))
        plt.plot(THRESHOLDS, mean_dice, 'r-', label='SCAN Strategy (Hack)', linewidth=2)
        plt.plot(THRESHOLDS, mean_ftp, 'b--', label='Filtered TP (Should be ~0)')
        plt.title(f"SCAN Strategy Simulation: {target} (Score={qu_score:.3f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()
        plt.savefig(os.path.join(output_dir, f"scan_curve_{target}.png"))
        plt.close()

    print("\n" + "="*60)
    print(f"{'🧪 KẾT QUẢ THÍ NGHIỆM SCAN STRATEGY':^60}")
    print("="*60)
    df = pd.DataFrame(final_report)
    cols = ["Class", "SCAN_Score", "AUC_Dice", "AUC_FTP", "AUC_FTN"]
    print(df[cols].to_string(index=False, float_format="%.4f"))
    print("-" * 60)
    df.to_csv(os.path.join(output_dir, "scan_metrics.csv"), index=False)

if __name__ == "__main__":
    # Có thể dùng argparse nếu muốn, ở đây chạy thẳng cho nhanh
    run_scan_experiment(mode='edl')