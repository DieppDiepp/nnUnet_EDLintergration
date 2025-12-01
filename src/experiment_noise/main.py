"""
🚀 ENTRY POINT (BATCH SUPPORT)
Script chạy thí nghiệm nhiễu trên nhiều ca.
"""
import sys
import os
import argparse
from tqdm import tqdm

# Setup Path
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(src_dir)
    if src_dir not in sys.path: sys.path.append(src_dir)
    if project_root not in sys.path: sys.path.append(project_root)
except: pass

from src.config import BASE_CONFIG
# Import hàm lấy danh sách ca từ utils gốc
from src.utils import get_validation_cases, get_case_list
from src.experiment_noise.runner import run_experiment_logic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default=None, help="Chạy 1 ca cụ thể (VD: BRATS_001)")
    parser.add_argument('--mode', type=str, default='edl')
    parser.add_argument('--limit', type=int, default=0, help="Giới hạn số ca chạy (0 = chạy hết)")
    parser.add_argument('--val_only', action='store_true', help="Chạy trên tập Validation (Fold 0)")
    args = parser.parse_args()
    
    cases_to_run = []
    
    # 1. Xác định danh sách ca cần chạy
    if args.case:
        # Ưu tiên 1: Chạy 1 ca cụ thể
        cases_to_run = [args.case]
        print(f"🎯 Mode: Single Case ({args.case})")
        
    elif args.val_only:
        # Ưu tiên 2: Chạy tập Validation (Chuẩn nhất)
        print("📂 Loading Validation set from split file...")
        try:
            cases_to_run = get_validation_cases(BASE_CONFIG["split_file"], fold=BASE_CONFIG["fold"])
            
            # Lọc chỉ lấy những ca có file ảnh thực tế trên đĩa (tránh lỗi file missing)
            available_files = set(get_case_list(BASE_CONFIG["image_folder"]))
            cases_to_run = [c for c in cases_to_run if c in available_files]
            
        except Exception as e:
            print(f"❌ Error loading split file: {e}")
            sys.exit(1)
            
    else:
        # Ưu tiên 3: Quét tất cả file trong folder ảnh (Fallback)
        print("📂 Scanning image folder for all cases...")
        cases_to_run = get_case_list(BASE_CONFIG["image_folder"])
    
    # 2. Áp dụng Limit
    if args.limit > 0:
        cases_to_run = cases_to_run[:args.limit]
        
    print(f"🔍 Found {len(cases_to_run)} cases to process.")
    
    # 3. Vòng lặp chạy thí nghiệm
    for case_id in tqdm(cases_to_run, desc="Running Noise Experiment"):
        try:
            run_experiment_logic(case_id, mode=args.mode)
        except Exception as e:
            print(f"\n❌ Error processing {case_id}: {e}")