"""
🚀 ECE (EXPECTED CALIBRATION ERROR) ANALYSIS SCRIPT
Đánh giá mức độ trung thực của mô hình: Độ tự tin (Confidence) có khớp với Độ chính xác (Accuracy) không?
Điểm ECE càng gần 0 càng tốt.
"""
import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# --- SETUP PATH (Lùi 2 bước) ---
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

def compute_ece(confidences, accuracies, num_bins=15):
    """Tính ECE và vẽ dữ liệu Reliability Diagram"""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Xác định các pixel rơi vào bin hiện tại
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean() # Tỷ lệ pixel trong bin này so với tổng
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Cộng dồn vào lỗi ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_data.append({
                'bin_center': (bin_lower + bin_upper) / 2,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'count': in_bin.sum()
            })
            
    return ece * 100.0, bin_data # Nhân 100 để báo cáo theo %

def run_ece_pipeline(mode='edl', n_cases=None, num_bins=10):
    print(f"🚀 BẮT ĐẦU TÍNH ECE | CHẾ ĐỘ: {mode.upper()} | BINS: {num_bins}")
    
    try:
        model_cfg = MODEL_CONFIGS[mode]
        base_folder = model_cfg["output_folder"]
        nifti_dir = os.path.join(base_folder, BASE_CONFIG.get("dir_nifti", "3d_nifti"))
        output_dir = os.path.join(base_folder, "analysis_ece_results") 
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(nifti_dir): raise FileNotFoundError(f"Không tìm thấy dữ liệu: {nifti_dir}")
    except Exception as e:
        print(f"❌ Lỗi cấu hình: {e}"); return

    all_cases = sorted([d for d in os.listdir(nifti_dir) if os.path.isdir(os.path.join(nifti_dir, d))])
    if n_cases: all_cases = all_cases[:n_cases]
    
    all_confidences = []
    all_accuracies = []

    for case_id in tqdm(all_cases, desc=f"Đang gom dữ liệu Confidence"):
        try:
            case_path = os.path.join(nifti_dir, case_id)
            pred = load_nifti_safe(os.path.join(case_path, "prediction.nii.gz"))
            gt = load_nifti_safe(os.path.join(case_path, "ground_truth.nii.gz"))
            conf = load_nifti_safe(os.path.join(case_path, "confidence.nii.gz"))
            mri = load_nifti_safe(os.path.join(case_path, "mri_crop.nii.gz"))

            if pred is None or gt is None or conf is None: continue

            # Tạo mặt nạ não (Brain Mask) để không tính vùng nền đen
            brain_mask = (mri > 0) if mri is not None else np.logical_or(pred > 0, gt > 0)
            if brain_mask.sum() == 0: continue

            # Lấy các pixel trong vùng não
            p_brain = pred[brain_mask]
            g_brain = gt[brain_mask]
            c_brain = conf[brain_mask]

            # Mảng boolean: Đoán đúng (1) hay sai (0)
            acc_mask = (p_brain == g_brain).astype(np.float32)

            all_confidences.append(c_brain)
            all_accuracies.append(acc_mask)

        except Exception: continue

    if not all_confidences:
        print("❌ Không có dữ liệu để tính ECE. Hãy chắc chắn file confidence.nii.gz đã được tạo."); return

    print(f"\n⏳ Đang tính toán ECE tổng thể (Gom toàn bộ pixel) với {num_bins} bins...")
    # Nối tất cả mảng 1D lại với nhau (có thể tốn RAM nếu dataset lớn, Colab dư sức xử lý)
    global_confidences = np.concatenate(all_confidences)
    global_accuracies = np.concatenate(all_accuracies)

    ece_score, bin_data = compute_ece(global_confidences, global_accuracies, num_bins=num_bins)

    # --- VẼ BIỂU ĐỒ ĐỘ TIN CẬY (RELIABILITY DIAGRAM) ---
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    if bin_data:
        xs = [b['bin_center'] for b in bin_data]
        ys = [b['accuracy'] for b in bin_data]
        
        # --- SỬA LẠI ĐỘ RỘNG CỘT (width = 1/num_bins) ---
        bar_width = 1.0 / num_bins
        plt.bar(xs, ys, width=bar_width, alpha=0.5, edgecolor='black', color='blue', label='Model Accuracy')
        plt.bar(xs, [b['confidence'] - b['accuracy'] for b in bin_data], 
                bottom=ys, width=bar_width, alpha=0.3, color='red', hatch='//', label='Calibration Gap')

    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Reliability Diagram ({num_bins} Bins)\nExpected Calibration Error (ECE) = {ece_score:.3f}%', fontsize=14)

    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, f"reliability_diagram_{num_bins}bins.png"), dpi=300)
    plt.close()

    # --- IN KẾT QUẢ ---
    print("\n" + "="*50)
    print(f"{'🏆 KẾT QUẢ EXPECTED CALIBRATION ERROR':^50}")
    print("="*50)
    print(f"Overall ECE Score: {ece_score:.4f} % (Càng nhỏ càng tốt)")
    print("-" * 50)
    
    # Lưu kết quả
    df = pd.DataFrame([{"Metric": "Global ECE (%)", "Value": ece_score}])
    df.to_csv(os.path.join(output_dir, f"final_ece_score_{num_bins}bins.csv"), index=False)
    print(f"✅ Đã lưu kết quả tại: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- MỞ KHÓA CHO TẤT CẢ CÁC MODE ---
    parser.add_argument('--mode', type=str, default='edl', 
                        choices=['edl', 'edl_raw', 'baseline', 'baseline_raw', 'edl_250'])
    
    parser.add_argument('--limit', type=int, default=0)

    parser.add_argument('--bins', type=int, default=10, help="Số lượng bins để chia ECE (Thường dùng 10 hoặc 15)")

    if 'ipykernel' in sys.modules: args = parser.parse_args([])
    else: args = parser.parse_args()

    run_ece_pipeline(mode=args.mode, n_cases=args.limit, num_bins=args.bins)