"""
🏃 RUNNER LOGIC (FIXED ALIGNMENT)
Sửa lỗi lệch hình: Sử dụng MRI đã Crop (từ engine) để khớp với GT/Pred.
"""
import os
import shutil
import nibabel as nib
import numpy as np
from src.config import BASE_CONFIG, MODEL_CONFIGS
from src.edl_engine import EDLInferenceEngine
from .utils import add_gaussian_noise, add_gaussian_blur, add_motion_ghosting, save_temp_nifti
from .plotting import plot_combined_noise_levels
from src.analysis.utils import get_binary_mask, compute_dice_score_binary

def calculate_dice_all_classes(pred, gt):
    scores = {}
    for target in ["WT", "TC", "ET"]:
        p_bin = get_binary_mask(pred, target)
        g_bin = get_binary_mask(gt, target)
        scores[target] = compute_dice_score_binary(p_bin, g_bin)
    return scores

def run_experiment_logic(case_id, mode='edl'):
    print(f"🧪 PROCESSING CASE: {case_id}")
    
    # 1. LOAD CONFIG & PATHS
    noise_cfg = BASE_CONFIG.get("noise_experiment_config", {})
    experiments_map = noise_cfg.get("settings", {})
    
    if not experiments_map:
        print("⚠️ Không tìm thấy cấu hình 'settings' trong noise_experiment_config")
        return

    temp_dir = BASE_CONFIG.get("noise_temp_dir", "/content/temp_noise")
    original_config = MODEL_CONFIGS[mode]
    base_img_dir = BASE_CONFIG["image_folder"]
    base_lbl_dir = BASE_CONFIG["label_folder"]
    
    case_plot_dir = os.path.join(original_config["output_folder"], "noise_experiments", case_id)
    os.makedirs(case_plot_dir, exist_ok=True)
    
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Detect ext
    ext = ".nii"
    if os.path.exists(os.path.join(base_img_dir, f"{case_id}_0000.nii.gz")):
        ext = ".nii.gz"

    slice_idx = None 

    # 2. MAIN LOOP
    for noise_type, levels in experiments_map.items():
        print(f"\n🔥 Running Experiment: {noise_type} | Levels: {levels}")
        
        all_results_for_plot = {}
        
        for level in levels:
            print(f"   ⚡ Simulating {noise_type} Level = {level}...")
            
            curr_input_dir = os.path.join(temp_dir, f"{noise_type.replace(' ', '_')}_level_{level}")
            os.makedirs(curr_input_dir, exist_ok=True)
            
            affine = None
            
            # A. Tạo dữ liệu biến dạng
            for i in range(4):
                fname = f"{case_id}_{i:04d}{ext}"
                src_path = os.path.join(base_img_dir, fname)
                dst_path = os.path.join(curr_input_dir, fname)
                
                img = nib.load(src_path)
                data = img.get_fdata()
                affine = img.affine
                
                if noise_type == "Gaussian Noise":
                    noisy_data = add_gaussian_noise(data, sigma=level)
                elif noise_type == "Gaussian Blur":
                    noisy_data = add_gaussian_blur(data, sigma=level)
                elif noise_type == "Motion Ghost":
                    noisy_data = add_motion_ghosting(data, num_ghosts=int(level), intensity=0.5)
                else:
                    noisy_data = data 
                
                save_temp_nifti(noisy_data, affine, dst_path)

            # B. Inference
            temp_config = original_config.copy()
            temp_config["image_folder"] = curr_input_dir
            temp_config["label_folder"] = base_lbl_dir
            temp_config["save_3d_nifti"] = False
            
            engine = EDLInferenceEngine(temp_config)
            
            # [FIX] Lấy biến `data_crop` trả về từ engine (Thay vì dùng `_`)
            # data_crop chính là ảnh MRI đã được nnU-Net crop và chuẩn hóa
            data_crop, seg_crop, pred_crop, unc_dict, _ = engine.process_case(case_id)
            
            if pred_crop is None: continue
            
            # C. Tính Dice
            dices = calculate_dice_all_classes(pred_crop, seg_crop[0])
            print(f"      📊 Dice: WT={dices['WT']:.2f} | TC={dices['TC']:.2f} | ET={dices['ET']:.2f}")

            # D. Chọn Slice
            if slice_idx is None:
                slice_idx = np.argmax(np.sum(seg_crop[0] > 0, axis=(0, 1)))
                print(f"      📸 Selected Slice: {slice_idx}")

            # E. Lưu kết quả (Dùng data_crop[0] - Kênh FLAIR đã crop)
            all_results_for_plot[level] = {
                "mri": data_crop[0], # <--- FIX: Dùng ảnh đã crop để khớp với GT
                "gt": seg_crop[0],
                "pred": pred_crop,
                "total": unc_dict["total"],
                "aleatoric": unc_dict["aleatoric"],
                "epistemic": unc_dict["epistemic"],
                "dices": dices
            }

        # F. Vẽ và Lưu
        if all_results_for_plot:
            safe_name = noise_type.replace(" ", "_")
            out_path = os.path.join(case_plot_dir, f"combined_{safe_name}.png")
            
            print(f"   🎨 Saving plot to: {out_path}")
            plot_combined_noise_levels(all_results_for_plot, case_id, slice_idx, out_path)
        else:
            print(f"❌ No results for {noise_type}")

    print(f"\n✅ All experiments completed for {case_id}!")