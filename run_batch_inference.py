import os
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import json

# ==============================================================================
# ⚙️ CONFIGURATION (CẤU HÌNH TẠI ĐÂY)
# ==============================================================================
CONFIG = {
    # 1. Đường dẫn Model & Data
    "checkpoint_path": "/content/drive/MyDrive/XUM_project/nnUNet_results/Dataset101_BraTS2020/EDLTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
    "image_folder":     "/content/nnUNet_raw/Dataset101_BraTS2020/imagesTr",
    "label_folder":     "/content/nnUNet_raw/Dataset101_BraTS2020/labelsTr",
    
    # 2. Đường dẫn lưu kết quả
    "output_folder":    "/content/drive/MyDrive/XUM_project/inference_results_batch",
    
    # 3. Cấu hình chạy (Batch Settings)
    "run_mode":         "validation_split",  # "random" (chọn ngẫu nhiên) hoặc "range" (chạy theo danh sách)
    "test_range":       [0, 10],  # Chạy từ ảnh số 0 đến số 10 trong danh sách (nếu mode="range")
    "num_random":       5,        # Số lượng ảnh random (nếu mode="random")
    
    # 4. Tùy chọn hiển thị
    "save_images":      True,     # Có lưu ảnh png không?
    "show_on_screen":   True,     # Có hiện lên màn hình Colab không? (Nên tắt nếu chạy quá nhiều ảnh)
}
# ==============================================================================

def get_case_list(folder):
    """Lấy danh sách tất cả Case ID trong folder và sắp xếp"""
    if not os.path.exists(folder):
        raise FileNotFoundError(f"❌ Folder {folder} không tồn tại!")
        
    # Lọc file _0000.nii hoặc .nii.gz
    files = sorted([f for f in os.listdir(folder) if f.endswith("_0000.nii") or f.endswith("_0000.nii.gz")])
    
    case_ids = []
    for f in files:
        if f.endswith(".nii"): cid = f.replace("_0000.nii", "")
        else: cid = f.replace("_0000.nii.gz", "")
        case_ids.append(cid)
        
    return case_ids

def calculate_dice(pred_slice, gt_slice):
    """Tính Dice Score 2D"""
    p = (pred_slice > 0).astype(np.float32)
    g = (gt_slice > 0).astype(np.float32)
    intersection = np.sum(p * g)
    sum_areas = np.sum(p) + np.sum(g)
    if sum_areas == 0: return 1.0
    return (2.0 * intersection) / sum_areas

def visualize_comparison(case_id, mri_data, gt_data, pred_data, uncertainty, slice_idx=None):
    """Vẽ và lưu ảnh"""
    # Logic tìm slice có khối u lớn nhất
    if slice_idx is None:
        sums_gt = np.sum(gt_data, axis=(0, 1, 2))
        sums_pred = np.sum(pred_data, axis=(0, 1))
        
        if sums_gt.max() > 0: slice_idx = np.argmax(sums_gt)
        elif sums_pred.max() > 0: slice_idx = np.argmax(sums_pred)
        else: slice_idx = gt_data.shape[3] // 2

    print(f"    📸 Drawing Slice: {slice_idx}")

    # Lấy dữ liệu slice (Xoay .T)
    img_slice = mri_data[0, :, :, slice_idx].T
    gt_slice = gt_data[0, :, :, slice_idx].T
    pred_slice = pred_data[:, :, slice_idx].T
    unc_slice = uncertainty[:, :, slice_idx].T

    # Tính chỉ số
    dice = calculate_dice(pred_slice, gt_slice)
    ratio = (np.sum(pred_slice>0) / np.sum(gt_slice>0) * 100) if np.sum(gt_slice>0) > 0 else 0

    # Vẽ
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    plt.suptitle(f"Case: {case_id} | Slice: {slice_idx}", fontsize=16, y=0.98)

    # 1. MRI
    ax[0].imshow(img_slice, cmap='gray', origin='lower')
    ax[0].set_title("MRI Input", fontsize=12)
    ax[0].axis('off')

    # 2. GT
    ax[1].imshow(img_slice, cmap='gray', origin='lower', alpha=0.6)
    if np.any(gt_slice): ax[1].imshow(gt_slice, cmap='Greens', origin='lower', alpha=0.6, interpolation='nearest')
    ax[1].set_title("Ground Truth", fontsize=12, color='green')
    ax[1].axis('off')

    # 3. Pred
    ax[2].imshow(img_slice, cmap='gray', origin='lower', alpha=0.6)
    if np.any(pred_slice): ax[2].imshow(pred_slice, cmap='jet', origin='lower', alpha=0.5, interpolation='nearest')
    ax[2].set_title(f"AI Prediction\nDice: {dice:.1%} | Area: {ratio:.0f}%", fontsize=12, color='blue')
    ax[2].axis('off')

    # 4. Uncertainty
    im = ax[3].imshow(unc_slice, cmap='hot', origin='lower', vmin=0, vmax=1.0)
    ax[3].set_title("Uncertainty Map", fontsize=12, color='red')
    ax[3].axis('off')
    plt.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)

    # Lưu ảnh
    if CONFIG["save_images"]:
        os.makedirs(CONFIG["output_folder"], exist_ok=True)
        save_path = os.path.join(CONFIG["output_folder"], f"{case_id}_slice{slice_idx}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        print(f"    ✅ Saved: {save_path}")
    
    if CONFIG["show_on_screen"]:
        plt.show()
    plt.close() # Giải phóng bộ nhớ

def process_case(predictor, case_id):
    print(f"\n🔍 Processing: {case_id}...")
    
    # 1. Tìm file input
    base_file = os.path.join(CONFIG["image_folder"], f"{case_id}_0000.nii")
    ext = ".nii" if os.path.exists(base_file) else ".nii.gz"
    
    image_files = [os.path.join(CONFIG["image_folder"], f"{case_id}_{i:04d}{ext}") for i in range(4)]
    gt_file = os.path.join(CONFIG["label_folder"], f"{case_id}{ext}")
    
    if not os.path.exists(gt_file):
        print(f"    ⚠️ Skipping {case_id}: Label file missing.")
        return

    # 2. Preprocessing (Crop)
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    data, seg, _ = preprocessor.run_case(image_files, gt_file, predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json)
    
    # 3. Inference & EDL Calculation
    data_tensor = torch.from_numpy(data).to(predictor.device)
    pred_logits = predictor.predict_logits_from_preprocessed_data(data_tensor)
    
    evidence = F.softplus(pred_logits)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=0)
    K = alpha.shape[0]
    uncertainty = (K / S).cpu().numpy()
    segmentation = torch.argmax(pred_logits, dim=0).cpu().numpy()
    
    # 4. Visualize
    visualize_comparison(case_id, data, seg, segmentation, uncertainty)

def get_validation_cases(fold=0):
    """Lấy danh sách các ca bệnh thuộc tập Validation của Fold cụ thể"""
    # Đường dẫn đến file splits_final.json (Nằm trong folder preprocessed)
    split_file = "/content/nnUNet_preprocessed/Dataset101_BraTS2020/splits_final.json"
    
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"❌ Không tìm thấy file split tại: {split_file}")
        
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    # splits là list các dict, mỗi dict đại diện cho 1 fold
    if fold >= len(splits):
        raise ValueError(f"❌ Fold {fold} không tồn tại trong file split!")
        
    val_keys = splits[fold]['val'] # Lấy danh sách key của tập validation
    print(f"📂 Đã load danh sách Validation Fold {fold}: {len(val_keys)} ca.")
    return val_keys

def main():
    print("🚀 --- BATCH INFERENCE STARTED ---")
    
    # 1. Load Model
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        verbose=False
    )
    
    checkpoint_folder = os.path.dirname(os.path.dirname(CONFIG["checkpoint_path"]))
    predictor.initialize_from_trained_model_folder(checkpoint_folder, use_folds=(0,), checkpoint_name="checkpoint_best.pth")
    print(f"📂 Model Loaded from: {checkpoint_folder}")

    # 2. Lấy danh sách Case
    all_cases = get_case_list(CONFIG["image_folder"])
    print(f"📂 Found total {len(all_cases)} cases in raw folder.")
    
    # 3. Lọc danh sách chạy
    # --- LOGIC CHỌN CASE MỚI ---
    if CONFIG["run_mode"] == "validation_split":
        # Lấy đúng danh sách Validation của Fold 0
        cases_to_run = get_validation_cases(fold=0)
        # Nếu muốn test nhanh thì chỉ lấy 10 ca đầu trong list val này
        # cases_to_run = cases_to_run[:10] 
        print(f"⚙️ Mode: VALIDATION SPLIT (Fold 0) -> Running {len(cases_to_run)} cases.")
        
    elif CONFIG["run_mode"] == "range":
        all_cases = get_case_list(CONFIG["image_folder"])
        start, end = CONFIG["test_range"]
        cases_to_run = all_cases[start:end]
        print(f"⚙️ Mode: RANGE [{start}:{end}] -> Running {len(cases_to_run)} cases.")
        
    else: # Random
        all_cases = get_case_list(CONFIG["image_folder"])
        import random
        cases_to_run = random.sample(all_cases, min(len(all_cases), CONFIG["num_random"]))
        print(f"⚙️ Mode: RANDOM -> Running {len(cases_to_run)} random cases.")

    # 4. Chạy vòng lặp
    for case_id in cases_to_run:
        try:
            process_case(predictor, case_id)
        except Exception as e:
            print(f"    ❌ Error processing {case_id}: {e}")

    print("\n✅ --- BATCH INFERENCE FINISHED ---")



if __name__ == "__main__":
    main()