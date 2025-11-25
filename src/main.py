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

sys.path.append("/content/drive/MyDrive/XUM_project")

from src.config import BASE_CONFIG, MODEL_CONFIGS
from src.utils import get_case_list, get_validation_cases, calculate_metric_per_class
from src.edl_engine import EDLInferenceEngine
from src.visualizer import visualize_comparison 

def parse_args():
    parser = argparse.ArgumentParser(description="Run Inference for BraTS EDL/Baseline")
    parser.add_argument('--mode', type=str, default='edl', choices=['edl', 'baseline'],
                        help="Chọn chế độ chạy: 'edl' hoặc 'baseline'")
    parser.add_argument('--fold', type=int, default=0, help="Fold cần validation (Default: 0)")
    return parser.parse_args()

def main():
    # 1. Parse Arguments & Setup Config
    args = parse_args()
    print(f"🏁 --- STARTING PIPELINE | MODE: {args.mode.upper()} | FOLD: {args.fold} ---")
    
    CONFIG = BASE_CONFIG.copy()
    CONFIG.update(MODEL_CONFIGS[args.mode])
    CONFIG["fold"] = args.fold # Override fold
    
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
                visualize_comparison(case_id, mri, gt, pred, unc_dict, CONFIG)
            
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
        
        csv_detail_name = CONFIG.get("file_csv_detail", "metrics_detailed.csv")
        detail_path = os.path.join(CONFIG["output_folder"], csv_detail_name)
        df.to_csv(detail_path, index=False)
        
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