"""
🚀 SUEO ANALYSIS SCRIPT (SPATIAL UNCERTAINTY ESTIMATION OVERLAP)
Đánh giá mức độ trùng khớp giữa vùng dự đoán sai và vùng bất định.
Càng cao càng tốt.
"""
import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# --- SETUP PATH ---
try:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Đang ở src/analysis
    src_dir = os.path.dirname(current_dir)                   # Lùi 1 bước ra src
    project_root = os.path.dirname(src_dir)                  # Lùi 2 bước ra nnUnet
    
    # Nhét thư mục gốc vào danh sách tìm kiếm của Python
    if project_root not in sys.path: 
        sys.path.insert(0, project_root)
except NameError: pass

try:
    from src.config import BASE_CONFIG, MODEL_CONFIGS
    from src.analysis.utils import load_nifti_safe
    from src.analysis.metrics import calculate_auc_score
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")

def normalize_uncertainty(unc_map):
    """Chuẩn hóa Uncertainty về khoảng [0, 100]"""
    u_min = unc_map.min()
    u_max = unc_map.max()
    if u_max == u_min:
        return np.zeros_like(unc_map)
    return ((unc_map - u_min) / (u_max - u_min)) * 100.0

def compute_sueo_curve(error_map, unc_norm, thresholds):
    """Tính sUEO cho một mảng các ngưỡng thresholds"""
    sueo_scores = []
    sum_e = error_map.sum()
    
    for t in thresholds:
        u_tau = (unc_norm >= t)
        sum_u = u_tau.sum()
        
        intersection = np.logical_and(u_tau, error_map).sum()
        denominator = sum_u + sum_e
        
        # Nếu không có sai sót và không có bất định -> Hoàn hảo
        if denominator == 0:
            sueo = 1.0 
        else:
            sueo = (2.0 * intersection) / denominator
            
        sueo_scores.append(sueo)
    return np.array(sueo_scores)

def run_sueo_pipeline(mode='edl', n_cases=None):
    print(f"🚀 STARTING sUEO ANALYSIS | MODE: {mode.upper()}")
    
    # 1. Config & Paths
    try:
        model_cfg = MODEL_CONFIGS[mode]
        base_folder = model_cfg["output_folder"]
        nifti_dir = os.path.join(base_folder, BASE_CONFIG.get("dir_nifti", "3d_nifti"))
        output_dir = os.path.join(base_folder, "analysis_sueo_results") 
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(nifti_dir): raise FileNotFoundError(f"Input missing: {nifti_dir}")
    except Exception as e:
        print(f"❌ Config Error: {e}"); return

    all_cases = sorted([d for d in os.listdir(nifti_dir) if os.path.isdir(os.path.join(nifti_dir, d))])
    if n_cases: all_cases = all_cases[:n_cases]
    print(f"🔍 Found {len(all_cases)} cases.")

    final_report = []
    TARGET_CLASSES = ["WT", "TC", "ET"]
    THRESHOLDS = np.arange(100, -1, -1) # 100 -> 0

    # 2. Main Loop
    for target in TARGET_CLASSES:
        print(f"\n📊 Class: {target}")
        
        agg_sueo = {"total": [], "aleatoric": [], "epistemic": []}

        for case_id in tqdm(all_cases, desc=f"Processing"):
            try:
                case_path = os.path.join(nifti_dir, case_id)
                pred = load_nifti_safe(os.path.join(case_path, "prediction.nii.gz"))
                gt = load_nifti_safe(os.path.join(case_path, "ground_truth.nii.gz"))
                mri = load_nifti_safe(os.path.join(case_path, "mri_crop.nii.gz"))

                if pred is None or gt is None: continue

                brain_mask = (mri > 0) if mri is not None else np.logical_or(pred > 0, gt > 0)
                if brain_mask.sum() == 0: continue

                # Xác định vùng (Masking theo class)
                if target == "WT":
                    p_mask = (pred > 0); g_mask = (gt > 0)
                elif target == "TC":
                    p_mask = np.logical_or(pred == 1, pred == 3)
                    g_mask = np.logical_or(gt == 1, gt == 3)
                else: # ET
                    p_mask = (pred == 3); g_mask = (gt == 3)

                # Error Map (Chỗ đoán sai nằm trong não)
                error_map = (p_mask != g_mask) & brain_mask
                
                # --- Tính sUEO cho các loại Uncertainty ---
                for u_type in ["total", "aleatoric", "epistemic"]:
                    u_map = load_nifti_safe(os.path.join(case_path, f"unc_{u_type}.nii.gz"))
                    if u_map is not None:
                        u_norm = normalize_uncertainty(u_map[brain_mask])
                        sueo_curve = compute_sueo_curve(error_map[brain_mask], u_norm, THRESHOLDS)
                        agg_sueo[u_type].append(sueo_curve)

            except Exception as e: 
                continue

        if not agg_sueo["total"]: 
            print(f"⚠️ No data for {target}"); continue

        # 3. Aggregation & Metrics
        x_norm = THRESHOLDS[::-1] / 100.0 
        report_row = {"Class": target}
        
        plt.figure(figsize=(10, 6))
        colors = {"total": "r-", "aleatoric": "g:", "epistemic": "b-."}
        
        for u_type in ["total", "aleatoric", "epistemic"]:
            if agg_sueo[u_type]:
                # Tính đường cong sUEO trung bình cho toàn bộ dataset
                mean_sueo = np.mean(agg_sueo[u_type], axis=0)
                
                # AUC của đường sUEO
                auc_sueo = calculate_auc_score(x_norm, mean_sueo[::-1])
                # Giá trị sUEO cao nhất đạt được trên đường cong
                max_sueo = np.max(mean_sueo)
                
                report_row[f"AUC_sUEO_{u_type.capitalize()}"] = auc_sueo
                report_row[f"Max_sUEO_{u_type.capitalize()}"] = max_sueo
                
                plt.plot(THRESHOLDS, mean_sueo, colors[u_type], 
                         label=f'{u_type.capitalize()} (AUC={auc_sueo:.3f}, Max={max_sueo:.3f})', linewidth=2)

        final_report.append(report_row)

        # 4. Plotting
        plt.xlabel("Uncertainty Threshold (τ)", fontsize=12)
        plt.ylabel(f"sUEO Score ({target})", fontsize=12)
        plt.title(f"Spatial Uncertainty Estimation Overlap ({target})\nHigher is better", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis() # Trục X từ 100 về 0
        
        plt.savefig(os.path.join(output_dir, f"sueo_curve_{target}.png"), dpi=300)
        plt.close()

    # 5. Final Report
    print("\n" + "="*80)
    print(f"{'🏆 sUEO METRICS REPORT':^80}")
    print("="*80)
    df = pd.DataFrame(final_report)
    if not df.empty:
        print(df.to_string(index=False, float_format="%.4f"))
        df.to_csv(os.path.join(output_dir, "final_sueo_metrics.csv"), index=False)
        print(f"\n✅ Results saved to: {output_dir}")
    else:
        print("❌ No results computed.")
    print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='edl', choices=['edl', 'edl_raw'])
    parser.add_argument('--limit', type=int, default=0)
    if 'ipykernel' in sys.modules: args = parser.parse_args([])
    else: args = parser.parse_args()
    
    run_sueo_pipeline(mode=args.mode, n_cases=args.limit)