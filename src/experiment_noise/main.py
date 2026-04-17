"""
🚀 ENTRY POINT (BATCH SUPPORT - NOISE EXPERIMENT)
Hỗ trợ tách bạch Model Fold (trọng số) và Data Test Fold (tập dữ liệu test).
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
from src.utils import get_test_cases, get_case_list
from src.experiment_noise.runner import run_experiment_logic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default=None, help="Chạy 1 ca cụ thể (VD: BRATS_001)")
    
    parser.add_argument('--mode', type=str, default='edl_250_fixed_split', 
                        choices=[
                            'edl', 'baseline', 'edl_raw', 'baseline_raw', 
                            'edl_250', 'baseline_250', 
                            'edl_50_fixed_split', 'baseline_50_fixed_split',
                            'edl_250_fixed_split', 'baseline_250_fixed_split'
                        ],
                        help="Chọn model để chạy thử nghiệm nhiễu")
                        
    parser.add_argument('--limit', type=int, default=0, help="Giới hạn số ca chạy (0 = chạy hết)")
    parser.add_argument('--test_only', action='store_true', help="Chạy trên tập Test cố định")
    
    # --- [SỬA Ở ĐÂY] Khai báo 2 biến Fold độc lập ---
    parser.add_argument('--fold', type=int, default=0, help="Fold của trọng số mô hình cần load (Model Fold)")
    parser.add_argument('--test_fold', type=int, default=0, help="Key fold_X trong file fixed_test.json để lấy data (Data Fold)")
    
    args = parser.parse_args()
    
    cases_to_run = []
    
    # 1. Xác định danh sách ca cần chạy
    if args.case:
        cases_to_run = [args.case]
        print(f"🎯 Mode: Single Case ({args.case}) | Model: {args.mode.upper()} (Weights Fold {args.fold})")
        
    elif args.test_only:
        # --- [SỬA Ở ĐÂY] Dùng args.test_fold để lấy danh sách ---
        print(f"📂 Loading Fixed Test set (Data Fold {args.test_fold})... | Model: {args.mode.upper()} (Weights Fold {args.fold})")
        try:
            test_file_path = BASE_CONFIG.get("test_file", "/content/drive/MyDrive/NCKH/nnUnet/data/experiments/fixed_test.json")
            
            # Truyền args.test_fold vào hàm get_test_cases
            cases_to_run = get_test_cases(test_file_path, fold=args.test_fold)
            
            # Lọc an toàn
            available_files = set(get_case_list(BASE_CONFIG["image_folder"]))
            cases_to_run = [c for c in cases_to_run if c in available_files]
            
        except Exception as e:
            print(f"❌ Error loading test file: {e}")
            sys.exit(1)
            
    else:
        print(f"📂 Scanning image folder for all cases... | Model: {args.mode.upper()} (Weights Fold {args.fold})")
        cases_to_run = get_case_list(BASE_CONFIG["image_folder"])
    
    # 2. Áp dụng Limit
    if args.limit > 0:
        cases_to_run = cases_to_run[:args.limit]
        
    print(f"🔍 Found {len(cases_to_run)} cases to process.")
    
    # 3. Vòng lặp chạy thí nghiệm
    for case_id in tqdm(cases_to_run, desc="Running Noise Experiment"):
        try:
            # Truyền args.fold vào để file runner biết lấy cục weights nào
            run_experiment_logic(case_id, mode=args.mode, fold=args.fold)
        except Exception as e:
            print(f"\n❌ Error processing {case_id}: {e}")
            import traceback
            traceback.print_exc()