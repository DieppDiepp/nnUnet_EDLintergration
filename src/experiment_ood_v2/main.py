"""
🚀 ENTRY POINT FOR OOD EXPERIMENT (BATCH SUPPORT)
Script chạy thí nghiệm OOD trên nhiều ca (Hỗ trợ Validation Set).
"""
import sys
import os
import argparse
from tqdm import tqdm

# Setup Path để Python hiểu 'src'
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(src_dir)
    if src_dir not in sys.path: sys.path.append(src_dir)
    if project_root not in sys.path: sys.path.append(project_root)
except: pass

from src.config import BASE_CONFIG
# Import các hàm tiện ích chọn file từ src/utils.py
from src.utils import get_validation_cases, get_case_list
# Import hàm chạy logic OOD
from src.experiment_ood_v2.runner import run_ood_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OOD Experiment")
    parser.add_argument('--case', type=str, default=None, help="Chạy 1 ca cụ thể (VD: BRATS_001)")
    parser.add_argument('--mode', type=str, default='edl', choices=['edl', 'baseline'])
    parser.add_argument('--limit', type=int, default=0, help="Giới hạn số ca chạy (0 = chạy hết)")
    parser.add_argument('--val_only', action='store_true', help="Chạy trên tập Validation (Fold 0)")
    args = parser.parse_args()
    
    cases_to_run = []
    
    # --- 1. XÁC ĐỊNH DANH SÁCH CA CẦN CHẠY ---
    if args.case:
        # Ưu tiên 1: Chạy 1 ca cụ thể (Debug nhanh)
        cases_to_run = [args.case]
        print(f"🎯 Mode: Single Case ({args.case})")
        
    elif args.val_only:
        # Ưu tiên 2: Chạy tập Validation (Chuẩn thí nghiệm)
        print("📂 Loading Validation set from split file...")
        try:
            # Lấy danh sách validation từ file split.json
            cases_to_run = get_validation_cases(BASE_CONFIG["split_file"], fold=BASE_CONFIG["fold"])
            
            # Lọc lại để đảm bảo file ảnh thực sự tồn tại trên ổ cứng
            available_files = set(get_case_list(BASE_CONFIG["image_folder"]))
            cases_to_run = [c for c in cases_to_run if c in available_files]
            
        except Exception as e:
            print(f"❌ Error loading split file: {e}")
            sys.exit(1)
            
    else:
        # Ưu tiên 3: Quét tất cả file trong folder ảnh (Chạy đại trà)
        print("📂 Scanning image folder for all cases...")
        cases_to_run = get_case_list(BASE_CONFIG["image_folder"])
    
    # --- 2. ÁP DỤNG GIỚI HẠN (LIMIT) ---
    if args.limit > 0:
        cases_to_run = cases_to_run[:args.limit]
        
    print(f"🔍 Found {len(cases_to_run)} cases to process.")
    
    # --- 3. VÒNG LẶP CHẠY THÍ NGHIỆM ---
    # Dùng tqdm để hiện thanh loading bar cho chuyên nghiệp
    for case_id in tqdm(cases_to_run, desc="Running OOD Experiment"):
        try:
            run_ood_experiment(case_id, mode=args.mode)
        except Exception as e:
            print(f"\n❌ Error processing {case_id}: {e}")
            import traceback
            traceback.print_exc()