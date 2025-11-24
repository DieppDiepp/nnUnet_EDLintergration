"""
🚀 MAIN SCRIPT (UPDATED V2)
Hỗ trợ báo cáo chi tiết và lưu file 3D.
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
    print("🏁 --- STARTING ADVANCED PIPELINE ---")
    engine = EDLInferenceEngine(CONFIG)
    
    # 1. Lập danh sách
    run_mode = CONFIG["run_mode"]
    if run_mode == "validation_split":
        cases = get_validation_cases(CONFIG["split_file"], fold=CONFIG["fold"])
        # Lọc case có thực
        available = set(get_case_list(CONFIG["image_folder"]))
        cases = [c for c in cases if c in available]
        print(f"⚙️ Mode: VALIDATION SPLIT -> Found {len(cases)} cases.")
    else:
        # Demo/Range logic...
        cases = get_case_list(CONFIG["image_folder"])[:5] 

    # 2. Chạy vòng lặp
    all_metrics = []
    
    for i, case_id in enumerate(cases):
        try:
            print(f"[{i+1}/{len(cases)}] Processing {case_id}...", end=" ")
            
            # Xử lý & Lưu 3D (nếu config bật)
            mri, gt, pred, unc, props = engine.process_case(case_id)
            
            # Tính Metrics (nếu config bật)
            metrics = {}
            if CONFIG["calc_metrics"]:
                spacing = props.get('spacing', None)
                # gt[0] vì gt là (1, X, Y, Z)
                metrics = calculate_metric_per_class(pred, gt[0], spacing)
                metrics["Case_ID"] = case_id
                all_metrics.append(metrics)
                
                # In nhanh kết quả
                print(f"-> Mean Dice: {metrics.get('Mean_Dice', 0):.4f}")
            else:
                print("-> Done.")

            # Lưu ảnh 2D (nếu config bật)
            if CONFIG["save_2d_snapshot"]:
                visualize_comparison(case_id, mri, gt, pred, unc, CONFIG)
            
        except Exception as e:
            print(f"\\n❌ Error {case_id}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Tổng hợp báo cáo
    if CONFIG["calc_metrics"] and all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Sắp xếp cột cho đẹp
        cols = ["Case_ID"] + [c for c in df.columns if c != "Case_ID"]
        df = df[cols]
        
        # Lưu chi tiết từng ca
        # detail_path = os.path.join(CONFIG["output_folder"], "metrics_detailed.csv")
        csv_detail_name = CONFIG.get("file_csv_detail", "metrics_detailed.csv")
        detail_path = os.path.join(CONFIG["output_folder"], csv_detail_name)

        df.to_csv(detail_path, index=False)
        
        # Tính trung bình toàn tập (Average)
        if CONFIG["metrics_average"]:
            mean_df = df.drop(columns=["Case_ID"]).mean()
            # summary_path = os.path.join(CONFIG["output_folder"], "metrics_summary.csv")
            csv_summary_name = CONFIG.get("file_csv_summary", "metrics_summary.csv")
            summary_path = os.path.join(CONFIG["output_folder"], csv_summary_name)
            
            mean_df.to_csv(summary_path)
            
            print("\n📊 --- FINAL REPORT ---")
            print(mean_df)
            print(f"✅ Report saved to: {CONFIG['output_folder']}")

    print("\n✅ --- PIPELINE COMPLETED ---")

if __name__ == "__main__":
    main()