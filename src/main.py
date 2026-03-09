"""
🚀 MAIN SCRIPT (HYBRID V7 - ROBUST CLI)
Hỗ trợ chạy dòng lệnh, nhưng vẫn giữ log bảng đẹp và xử lý lỗi an toàn.
"""
import sys
import os
import random
import pandas as pd
import numpy as np
import argparse

sys.path.append("/content/drive/MyDrive/NCKH/nnUnet")

from src.config import BASE_CONFIG, MODEL_CONFIGS
from src.utils import get_case_list, get_validation_cases, calculate_metric_per_class
from src.edl_engine import EDLInferenceEngine
from src.visualizer import visualize_comparison 

def parse_args():
    parser = argparse.ArgumentParser(description="Chạy dự đoán cho BraTS EDL/Baseline")
    parser.add_argument('--mode', type=str, default='edl', 
                        choices=['edl', 'baseline', 'edl_raw', 'baseline_raw', 'edl_250'],
                        help="Chọn chế độ chạy")
    
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--run_mode', type=str, default='validation_split', 
                        choices=['validation_split', 'range', 'random'])
                        
    return parser.parse_args()

def main():
    # 1. Parse Arguments & Setup Config
    args = parse_args()
    print(f"🏁 --- STARTING PIPELINE | MODE: {args.mode.upper()} | FOLD: {args.fold} ---")
    
    CONFIG = BASE_CONFIG.copy()
    CONFIG.update(MODEL_CONFIGS[args.mode])
    CONFIG["fold"] = args.fold # Override fold
    CONFIG["run_mode"] = args.run_mode # Override run mode
    CONFIG["checkpoint_path"] = CONFIG["checkpoint_path"].format(fold=args.fold)

    # Tạo folder output trước
    os.makedirs(CONFIG["output_folder"], exist_ok=True)

    # 2. Init Engine
    engine = EDLInferenceEngine(CONFIG)
    
    # 3. List Cases (Logic lọc file an toàn)
    all_cases_on_disk = get_case_list(CONFIG["image_folder"])
    run_mode = CONFIG["run_mode"]
    
    if run_mode == "validation_split":
        cases = get_validation_cases(CONFIG["split_file"], fold=CONFIG["fold"])
        available_set = set(all_cases_on_disk)
        cases = [c for c in cases if c in available_set]
        print(f"⚙️ Mode: VALIDATION SPLIT -> Found {len(cases)} cases (Filtered).")
    elif run_mode == "range":
        start, end = CONFIG.get("test_range", [0, 5])
        cases = all_cases_on_disk[start:end]
        print(f"⚙️ Mode: RANGE [{start}:{end}] -> {len(cases)} cases.")
    else:
        num = CONFIG.get("num_random", 5)
        cases = random.sample(all_cases_on_disk, min(len(all_cases_on_disk), num))
        print(f"⚙️ Mode: RANDOM -> {len(cases)} cases.")

    # 4. Loop
    all_metrics = []
    
    # Header Bảng
    print("\n" + "="*85)
    print(f"{'Index':<8} | {'Case ID':<15} | {'Dice WT':<8} | {'Dice TC':<8} | {'Dice ET':<8} | {'Mean':<8}")
    print("-" * 85)

    for i, case_id in enumerate(cases):
        try:
            # Process
            mri, gt, pred, unc_dict, props = engine.process_case(case_id)
            
            # [QUAN TRỌNG] Khôi phục tinh hoa: Check lỗi trả về None
            if mri is None:
                print(f"{i+1:<8} | {case_id:<15} | {'SKIPPED (Error)':<40}")
                continue
            
            if CONFIG["calc_metrics"]:
                spacing = props.get('spacing', None)
                metrics = calculate_metric_per_class(pred, gt[0], spacing)
                metrics["Case_ID"] = case_id
                all_metrics.append(metrics)
                
                d_wt = metrics.get('Dice_WT', 0)
                d_tc = metrics.get('Dice_TC', 0)
                d_et = metrics.get('Dice_ET', 0)
                d_mean = metrics.get('Mean_Dice', 0)
                print(f"{i+1:<8} | {case_id:<15} | {d_wt:.4f}   | {d_tc:.4f}   | {d_et:.4f}   | {d_mean:.4f}")
            else:
                print(f"{i+1:<8} | {case_id:<15} | {'Done':<40}")

            if CONFIG["save_2d_snapshot"]:
                # --- [SỬA LẠI] CHUYỂN SANG TRỤC 2 (Mặt cắt ngang chuẩn BraTS) ---
                AXIAL_AXIS = 0  

                axes_to_sum = tuple([i for i in (0,1,2) if i != AXIAL_AXIS])
                
                # --- [SỬA Ở ĐÂY] Thêm "> 0" để chỉ đếm số lượng pixel u thực sự ---
                tumor_distribution = np.sum(gt[0] > 0, axis=axes_to_sum)
                
                tumor_slices = np.where(tumor_distribution > 0)[0]

                if len(tumor_slices) > 0:
                    # 1. Vẽ lát cắt ngang giữa khối u (50% - Thường là bự nhất và đẹp nhất)
                    idx_50 = tumor_slices[int(len(tumor_slices) * 0.50)]
                    visualize_comparison(case_id, mri, gt, pred, unc_dict, CONFIG, 
                                        slice_idx=idx_50, view_axis=AXIAL_AXIS, suffix_name="Axial_50pct_Center")
                                        
                    # 2. Vẽ lát cắt 25% (Đỉnh u)
                    idx_25 = tumor_slices[int(len(tumor_slices) * 0.25)]
                    visualize_comparison(case_id, mri, gt, pred, unc_dict, CONFIG, 
                                        slice_idx=idx_25, view_axis=AXIAL_AXIS, suffix_name="Axial_25pct_Top")
                                        
                    # 3. Vẽ lát cắt 75% (Đáy u)
                    idx_75 = tumor_slices[int(len(tumor_slices) * 0.75)]
                    visualize_comparison(case_id, mri, gt, pred, unc_dict, CONFIG, 
                                        slice_idx=idx_75, view_axis=AXIAL_AXIS, suffix_name="Axial_75pct_Bottom")
                else:
                    print(f"    ⚠️ Không tìm thấy khối u trong Ground Truth của {case_id}")

            
        except Exception as e:
            print(f"\n❌ Critical Error {case_id}: {e}")
            import traceback
            traceback.print_exc()

    # 5. Report
    if CONFIG["calc_metrics"] and all_metrics:
        df = pd.DataFrame(all_metrics)
        cols = ["Case_ID", "Dice_WT", "Dice_TC", "Dice_ET", "Mean_Dice", "HD95_WT", "HD95_TC", "HD95_ET"]
        cols += [c for c in df.columns if c not in cols]
        df = df[cols]
        
        # --- LẤY SỐ FOLD HIỆN TẠI ---
        fold_num = CONFIG.get("fold", 0)
        
        # --- 1. LƯU BẢNG CHI TIẾT (VÀO THƯ MỤC RIÊNG) ---
        detail_dir = os.path.join(CONFIG["output_folder"], "metrics_detailed")
        os.makedirs(detail_dir, exist_ok=True) # Tự tạo folder nếu chưa có
        
        detail_filename = f"metrics_detailed_fold{fold_num}.csv"
        detail_path = os.path.join(detail_dir, detail_filename)
        df.to_csv(detail_path, index=False)
        
        if CONFIG["metrics_average"]:
            # --- 2. LƯU BẢNG TÓM TẮT (VÀO THƯ MỤC RIÊNG) ---
            summary_dir = os.path.join(CONFIG["output_folder"], "metrics_summary")
            os.makedirs(summary_dir, exist_ok=True)
            
            summary_filename = f"metrics_summary_fold{fold_num}.csv"
            summary_path = os.path.join(summary_dir, summary_filename)
            
            mean_df = df.drop(columns=["Case_ID"]).mean()
            mean_df.to_csv(summary_path)
            
            print("\n" + "="*60)
            print(f"{'📊 FINAL SUMMARY (AVERAGE)':^60}")
            print("-" * 60)
            print(f"{'Metric':<15} | {'WT':<10} | {'TC':<10} | {'ET':<10}")
            print("-" * 60)
            
            m_d_wt = mean_df.get('Dice_WT', 0)
            m_d_tc = mean_df.get('Dice_TC', 0)
            m_d_et = mean_df.get('Dice_ET', 0)
            
            m_h_wt = mean_df.get('HD95_WT', 0)
            m_h_tc = mean_df.get('HD95_TC', 0)
            m_h_et = mean_df.get('HD95_ET', 0)

            print(f"{'Dice Score':<15} | {m_d_wt:.4f}     | {m_d_tc:.4f}     | {m_d_et:.4f}")
            print(f"{'HD95 (mm)':<15} | {m_h_wt:.4f}     | {m_h_tc:.4f}     | {m_h_et:.4f}")
            print("-" * 60)
            print(f"Overall Mean Dice: {mean_df.get('Mean_Dice', 0):.4f}")
            print(f"✅ Report saved to: {summary_path}") # In ra đường dẫn mới cho dễ check

    print("\n✅ --- PIPELINE COMPLETED ---")

if __name__ == "__main__":
    main()