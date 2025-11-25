"""
⚙️ CONFIGURATION MODULE (UPDATED V3)
Hỗ trợ Multi-Model Configuration (EDL vs Baseline).
"""

# 1. CẤU HÌNH CHUNG (Dùng chung cho cả 2)
BASE_CONFIG = {
    # Input Data
    "image_folder":     "/content/nnUNet_raw/Dataset101_BraTS2020/imagesTr",
    "label_folder":     "/content/nnUNet_raw/Dataset101_BraTS2020/labelsTr",
    "split_file":       "/content/nnUNet_preprocessed/Dataset101_BraTS2020/splits_final.json",
    
    # Feature Toggles
    "save_2d_snapshot": True,
    "save_3d_nifti":    True, 
    "calc_metrics":     True,
    
    # Metrics Settings
    "metrics_per_class": True,
    "metrics_average":   True,
    
    # Run Settings
    "run_mode":         "validation_split", # "validation_split" | "range" | "random"
    "fold":             0,
    "test_range":       [0, 5],
    "num_random":       5,
    "show_on_screen":   False,
    "figsize":          (30, 6),
    
    # Output File Names
    "dir_nifti":        "3d_nifti",
    "file_csv_detail":  "metrics_detailed.csv",
    "file_csv_summary": "metrics_summary.csv",
}

# 2. CẤU HÌNH RIÊNG (SPECIFIC PATHS)
MODEL_CONFIGS = {
    # --- Cấu hình cho EDL Model ---
    "edl": {
        "model_mode": "edl", # Bật tính năng Uncertainty
        "checkpoint_path": "/content/drive/MyDrive/XUM_project/nnUNet_results/Dataset101_BraTS2020/EDLTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
        "output_folder":   "/content/drive/MyDrive/XUM_project/inference_results_edl",
    },
    
    # --- Cấu hình cho Baseline Model ---
    "baseline": {
        "model_mode": "baseline", # Tắt Uncertainty, chỉ chạy Seg
        "checkpoint_path": "/content/drive/MyDrive/XUM_project/nnUNet_results/Dataset101_BraTS2020/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
        "output_folder":   "/content/drive/MyDrive/XUM_project/inference_results_baseline",
    }
}