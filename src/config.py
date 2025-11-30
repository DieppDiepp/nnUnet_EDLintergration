"""
⚙️ CONFIGURATION MODULE (UPDATED V3)
Hỗ trợ Multi-Model Configuration (EDL vs Baseline).
"""

# 1. CẤU HÌNH CHUNG (Dùng chung cho cả 2)
BASE_CONFIG = {
    # Input Data
    "image_folder":     "/content/nnUNet_raw/Dataset101_BraTS2020/imagesTr",
    "label_folder":     "/content/nnUNet_raw/Dataset101_BraTS2020/labelsTr",
    "split_file":       "/content/nnUNet_preprocessed/Dataset101_BraTS2020/splits_final.json", # File JSON chứa danh sách train/val split theo fold.
    
    # Feature Toggles
    "save_2d_snapshot": True, # Có lưu ảnh chụp lát cắt 2D (file .png) để xem nhanh không.
    "save_3d_nifti":    True, # Có lưu file kết quả 3D (.nii.gz) (Tốn dung lượng nhưng cần để phân tích AUSE), gồm ground_truth, mri_crop, prediction, 3 uncertainty (total/aleatoric/epistemic) 
    "calc_metrics":     True, # Có tính toán các chỉ số Dice/HD95 không
    
    # Metrics Settings
    "metrics_per_class": True, # Tính điểm riêng cho từng vùng u. Với mỗi ca bệnh, code sẽ tính điểm Dice/HD95 cho từng lớp riêng biệt: WT (Whole Tumor), TC (Tumor Core), và ET (Enhancing Tumor)
    "metrics_average":   True,
    
    # Run Settings
    "run_mode":         "validation_split", # "validation_split" - chạy full tập test | "range" - lấy bao nhiêu ca ra chạy thử | "random" - lấy ngẫu nhiên bao nhiêu ca
    "fold":             0,
    "test_range":       [0, 5],
    "num_random":       5,
    "show_on_screen":   False,
    "figsize":          (30, 6),
    
    # Output File Names
    "dir_nifti":        "3d_nifti",
    "file_csv_detail":  "metrics_detailed.csv", # Các cột chi tiết như Dice_WT, Dice_TC, Dice_ET cho từng bệnh nhân.
    "file_csv_summary": "metrics_summary.csv", # Tóm tắt trung bình, độ lệch chuẩn cho các lớp (trung bình trên các samples)

    # --- CẤU HÌNH AUSE MỚI ---
    # Giới hạn trục X của biểu đồ Risk-Coverage
    # [Start, End]: 1.0 là giữ 100%, 0.05 là giữ 5%
    "ause_retention_range": [1.0, 0.5], 
    "ause_steps": 20, # Độ mịn của biểu đồ (tính toán tại 20 điểm mốc).
}

# 2. CẤU HÌNH RIÊNG (SPECIFIC PATHS)
MODEL_CONFIGS = {
    # --- Cấu hình cho EDL Model ---
    "edl": {
        "model_mode": "edl", # Bật tính năng Uncertainty
        "checkpoint_path": "/content/drive/MyDrive/XUM_project/nnUNet_results/Dataset101_BraTS2020/EDLTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth", # Trỏ đến file .pth đã train bằng EDLTrainer (file này chứa trọng số mạng đã học cách dự đoán tham số Dirichlet).
        "output_folder":   "/content/drive/MyDrive/XUM_project/inference_results_edl", # Kết quả sẽ lưu vào thư mục inference_results_edl
    },
    
    # --- Cấu hình cho Baseline Model ---
    "baseline": {
        "model_mode": "baseline", # Tắt Uncertainty, chỉ chạy Seg
        "checkpoint_path": "/content/drive/MyDrive/XUM_project/nnUNet_results/Dataset101_BraTS2020/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth", # Trỏ tới file .pth của mô hình thường (nnUNetTrainer).
        "output_folder":   "/content/drive/MyDrive/XUM_project/inference_results_baseline",
    }
}