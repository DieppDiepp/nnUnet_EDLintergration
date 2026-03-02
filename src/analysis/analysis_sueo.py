"""
🚀 SOFT UEO (sUEO) ANALYSIS SCRIPT
Đánh giá sUEO (soft Uncertainty-Error Overlap) trực tiếp bằng giá trị liên tục.
Không cần phải cắt ngưỡng (thresholding) gây tốn thời gian.
"""
import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings

# --- SETUP PATH (Lùi 2 bước để tìm đúng thư mục gốc) ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(src_dir)
    if project_root not in sys.path: sys.path.insert(0, project_root)
except NameError: pass

try:
    from src.config import BASE_CONFIG, MODEL_CONFIGS
    from src.analysis.utils import load_nifti_safe
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")

def compute_soft_ueo(error_map, unc_map):
    """Tính điểm soft UEO trực tiếp trên khoảng [0, 1]"""
    # 1. Kéo độ bất định về chuẩn 0-1
    u_min = unc_map.min()
    u_max = unc_map.max()
    if u_max == u_min:
        u_01 = np.zeros_like(unc_map)
    else:
        u_01 = (unc_map - u_min) / (u_max - u_min)

    # 2. Ráp công thức sUEO chuẩn của bài báo
    # error_map vốn là 0 và 1, nên bình phương lên vẫn là chính nó
    intersection = np.sum(error_map * u_01)
    denominator = np.sum(error_map) + np.sum(u_01**2)
    
    # Nếu máy vẽ đúng 100% và không có độ bất định -> Điểm tuyệt đối
    if denominator == 0:
        return 1.0 
        
    return (2.0 * intersection) / denominator

def run_soft_sueo_pipeline(mode='edl', n_cases=None):
    print(f"🚀 BẮT ĐẦU TÍNH SOFT UEO (sUEO) | CHẾ ĐỘ: {mode.upper()}")
    
    # 1. Đọc đường dẫn
    try:
        model_cfg = MODEL_CONFIGS[mode]
        base_folder = model_cfg["output_folder"]
        nifti_dir = os.path.join(base_folder, BASE_CONFIG.get("dir_nifti", "3d_nifti"))
        output_dir = os.path.join(base_folder, "analysis_soft_sueo_results") 
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(nifti_dir): raise FileNotFoundError(f"Không tìm thấy dữ liệu: {nifti_dir}")
    except Exception as e:
        print(f"❌ Lỗi cấu hình: {e}"); return

    all_cases = sorted([d for d in os.listdir(nifti_dir) if os.path.isdir(os.path.join(nifti_dir, d))])
    if n_cases: all_cases = all_cases[:n_cases]
    print(f"🔍 Tìm thấy {len(all_cases)} ca bệnh.")

    final_report = []
    detailed_results = []
    TARGET_CLASSES = ["WT", "TC", "ET"]

    # 2. Chạy tính toán
    for target in TARGET_CLASSES:
        print(f"\n📊 Lớp u: {target}")
        
        agg_sueo = {"total": [], "aleatoric": [], "epistemic": []}

        for case_id in tqdm(all_cases, desc=f"Đang xử lý {target}"):
            try:
                case_path = os.path.join(nifti_dir, case_id)
                pred = load_nifti_safe(os.path.join(case_path, "prediction.nii.gz"))
                gt = load_nifti_safe(os.path.join(case_path, "ground_truth.nii.gz"))
                mri = load_nifti_safe(os.path.join(case_path, "mri_crop.nii.gz"))

                if pred is None or gt is None: continue

                # Chỉ tính trong vùng não
                brain_mask = (mri > 0) if mri is not None else np.logical_or(pred > 0, gt > 0)
                if brain_mask.sum() == 0: continue

                # Tách riêng lớp u đang xét
                if target == "WT":
                    p_mask = (pred > 0); g_mask = (gt > 0)
                elif target == "TC":
                    p_mask = np.logical_or(pred == 1, pred == 3)
                    g_mask = np.logical_or(gt == 1, gt == 3)
                else: # ET
                    p_mask = (pred == 3); g_mask = (gt == 3)

                # Vùng máy đoán sai
                error_map = (p_mask != g_mask) & brain_mask
                
                case_res = {"Case_ID": case_id, "Class": target}
                
                # Tính cho cả 3 loại bản đồ
                for u_type in ["total", "aleatoric", "epistemic"]:
                    u_map = load_nifti_safe(os.path.join(case_path, f"unc_{u_type}.nii.gz"))
                    if u_map is not None:
                        sueo_val = compute_soft_ueo(error_map[brain_mask], u_map[brain_mask])
                        agg_sueo[u_type].append(sueo_val)
                        case_res[f"sUEO_{u_type.capitalize()}"] = sueo_val
                        
                detailed_results.append(case_res)

            except Exception: 
                continue

        if not agg_sueo["total"]: 
            print(f"⚠️ Bỏ qua {target} vì thiếu dữ liệu"); continue

        # 3. Tính trung bình và độ lệch chuẩn
        report_row = {"Class": target}
        for u_type in ["total", "aleatoric", "epistemic"]:
            if agg_sueo[u_type]:
                report_row[f"sUEO_{u_type.capitalize()}_Mean"] = np.mean(agg_sueo[u_type])
                report_row[f"sUEO_{u_type.capitalize()}_Std"] = np.std(agg_sueo[u_type])

        final_report.append(report_row)

    # 4. In và lưu bảng kết quả
    print("\n" + "="*80)
    print(f"{'🏆 BẢNG KẾT QUẢ SOFT UEO (sUEO)':^80}")
    print("="*80)
    df_summary = pd.DataFrame(final_report)
    
    if not df_summary.empty:
        print(df_summary.to_string(index=False, float_format="%.4f"))
        
        # Lưu bảng chung
        df_summary.to_csv(os.path.join(output_dir, "summary_soft_sueo.csv"), index=False)
        # Lưu bảng chi tiết từng người
        df_detail = pd.DataFrame(detailed_results)
        df_detail.to_csv(os.path.join(output_dir, "detailed_soft_sueo_per_case.csv"), index=False)
        
        print(f"\n✅ Đã lưu kết quả tại: {output_dir}")
    else:
        print("❌ Không có kết quả nào được tính.")
    print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='edl', choices=['edl', 'edl_raw'])
    parser.add_argument('--limit', type=int, default=0)
    if 'ipykernel' in sys.modules: args = parser.parse_args([])
    else: args = parser.parse_args()
    
    run_soft_sueo_pipeline(mode=args.mode, n_cases=args.limit)