"""
⚙️ CONFIGURATION MODULE (UPDATED V2)
Đã thêm cấu hình tên file/folder đầu ra.
"""

CONFIG = {
    # --- PATHS (INPUT) ---
    "checkpoint_path": "/content/drive/MyDrive/XUM_project/nnUNet_results/Dataset101_BraTS2020/EDLTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
    "image_folder":     "/content/nnUNet_raw/Dataset101_BraTS2020/imagesTr",
    "label_folder":     "/content/nnUNet_raw/Dataset101_BraTS2020/labelsTr",
    "split_file":       "/content/nnUNet_preprocessed/Dataset101_BraTS2020/splits_final.json",
    
    # --- PATHS (OUTPUT) ---
    "output_folder":    "/content/drive/MyDrive/XUM_project/inference_results_batch",
    
    # [MỚI] Tên các file/folder con đầu ra (Sửa tại đây)
    "dir_nifti":        "3d_nifti",             # Tên folder chứa file .nii.gz
    "file_csv_detail":  "metrics_detailed.csv", # Tên file báo cáo chi tiết
    "file_csv_summary": "metrics_summary.csv",  # Tên file báo cáo tổng hợp
    
    # --- FEATURES TOGGLE ---
    "save_2d_snapshot": True,   
    "save_3d_nifti":    True,   
    "calc_metrics":     True,   
    
    # --- METRICS SETTINGS ---
    "metrics_per_class": True,
    "metrics_average":   True,
    
    # --- RUN SETTINGS ---
    "run_mode":         "random", # "validation_split" | "range" | "random"
    "fold":             0,
    "test_range":       [0, 5],
    "num_random":       5,
    
    # --- VISUALIZATION ---
    "show_on_screen":   False,
    "figsize":          (24, 6),
}