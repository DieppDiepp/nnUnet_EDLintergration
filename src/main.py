"""
🚀 MAIN SCRIPT (UPDATED V6 - FINAL)
In ra log chi tiết 3 chỉ số BraTS (WT, TC, ET) với định dạng bảng đẹp mắt.
Đảm bảo an toàn tuyệt đối (Robust Error Handling).
"""
import sys
import os
import random
import pandas as pd
import numpy as np

sys.path.append("/content/drive/MyDrive/XUM_project")

from src.config import CONFIG
from src.utils import get_case_list, get_validation_cases, calculate_metric_per_class
from src.edl_engine import EDLInferenceEngine
from src.visualizer import visualize_comparison 

def main():
    print("🏁 --- STARTING EDL PIPELINE (BRATS REGIONS) ---")
    engine = EDLInferenceEngine(CONFIG)
    
    # --- 1. LẬP DANH SÁCH CASE ---
    run_mode = CONFIG["run_mode"]
    all_cases_on_disk = get_case_list(CONFIG["image_folder"])
    
    if run_mode == "validation_split":
        cases = get_validation_cases(CONFIG["split_file"], fold=CONFIG["fold"])
        available_set = set(all_cases_on_disk)
        cases = [c for c in cases if c in available_set]
        print(f"⚙️ Mode: VALIDATION SPLIT -> Found {len(cases)} cases.")
    elif run_mode == "range":
        start, end = CONFIG["test_range"]
        cases = all_cases_on_disk[start:end]
        print(f"⚙️ Mode: RANGE [{start}:{end}] -> {len(cases)} cases.")
    else:
        num_rnd = CONFIG.get("num_random", 5)
        cases = random.sample(all_cases_on_disk, min(len(all_cases_on_disk), num_rnd))
        print(f"⚙️ Mode: RANDOM -> {len(cases)} cases.")

    # --- 2. VÒNG LẶP XỬ LÝ ---
    all_metrics = []
    
    # In Header bảng
    print("\n" + "="*85)
    print(f"{'Index':<8} | {'Case ID':<15} | {'Dice WT':<8} | {'Dice TC':<8} | {'Dice ET':<8} | {'Mean':<8}")
    print("-" * 85)

    for i, case_id in enumerate(cases):
        try:
            # Process (Trả về unc_dict)
            mri, gt, pred, unc_dict, props = engine.process_case(case_id)
            
            # [QUAN TRỌNG] Kiểm tra an toàn: Nếu lỗi load file -> Bỏ qua
            if mri is None:
                print(f"{i+1:<8} | {case_id:<15} | {'SKIPPED (Error)':<40}")
                continue

            if CONFIG["calc_metrics"]:
                spacing = props.get('spacing', None)
                # gt[0] vì gt shape là (1, X, Y, Z)
                metrics = calculate_metric_per_class(pred, gt[0], spacing)
                metrics["Case_ID"] = case_id
                all_metrics.append(metrics)
                
                # Lấy giá trị để in
                d_wt = metrics.get('Dice_WT', 0)
                d_tc = metrics.get('Dice_TC', 0)
                d_et = metrics.get('Dice_ET', 0)
                d_mean = metrics.get('Mean_Dice', 0)
                
                # In dòng kết quả thẳng hàng
                print(f"{i+1:<8} | {case_id:<15} | {d_wt:.4f}   | {d_tc:.4f}   | {d_et:.4f}   | {d_mean:.4f}")
            else:
                print(f"{i+1:<8} | {case_id:<15} | {'Done (No Metrics)':<40}")

            if CONFIG["save_2d_snapshot"]:
                visualize_comparison(case_id, mri, gt, pred, unc_dict, CONFIG)
            
        except Exception as e:
            # In lỗi nhưng không làm vỡ layout bảng quá nhiều
            print(f"\n❌ Error {case_id}: {e}")
            import traceback
            traceback.print_exc()

    # --- 3. TỔNG HỢP BÁO CÁO ---
    if CONFIG["calc_metrics"] and all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Sắp xếp cột thông minh
        # Ưu tiên Case_ID, sau đó đến các chỉ số Dice, rồi HD95
        priority_cols = ["Case_ID", "Dice_WT", "Dice_TC", "Dice_ET", "Mean_Dice", 
                         "HD95_WT", "HD95_TC", "HD95_ET"]
        # Giữ lại các cột khác nếu có (ví dụ spacing...)
        final_cols = [c for c in priority_cols if c in df.columns] + [c for c in df.columns if c not in priority_cols]
        df = df[final_cols]
        
        # Lưu chi tiết
        csv_detail_name = CONFIG.get("file_csv_detail", "metrics_detailed.csv")
        detail_path = os.path.join(CONFIG["output_folder"], csv_detail_name)
        df.to_csv(detail_path, index=False)
        
        # Tính trung bình và In bảng tổng kết đẹp
        if CONFIG["metrics_average"]:
            csv_summary_name = CONFIG.get("file_csv_summary", "metrics_summary.csv")
            summary_path = os.path.join(CONFIG["output_folder"], csv_summary_name)
            
            mean_df = df.drop(columns=["Case_ID"]).mean()
            mean_df.to_csv(summary_path)
            
            print("\n" + "="*60)
            print(f"{'📊 FINAL SUMMARY (AVERAGE)':^60}")
            print("-" * 60)
            print(f"{'Metric':<15} | {'WT':<10} | {'TC':<10} | {'ET':<10}")
            print("-" * 60)
            
            # Lấy giá trị an toàn (tránh lỗi nếu key thiếu)
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
            print(f"✅ Report saved to: {CONFIG['output_folder']}")

    print("\n✅ --- PIPELINE COMPLETED ---")

if __name__ == "__main__":
    main()