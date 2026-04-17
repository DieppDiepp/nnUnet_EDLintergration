"""
🏃 RUNNER LOGIC (UPDATED V2)
Sửa lỗi lệch hình, đồng bộ trục Axial=0, hỗ trợ cả Baseline (không phân rã bất định).
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

def run_experiment_logic(case_id, mode='edl', fold=0):
    print(f"🧪 PROCESSING CASE: {case_id}")
    
    noise_cfg = BASE_CONFIG.get("noise_experiment_config", {})
    experiments_map = noise_cfg.get("settings", {})
    
    if not experiments_map:
        print("⚠️ Không tìm thấy cấu hình 'settings' trong noise_experiment_config")
        return

    temp_dir = BASE_CONFIG.get("noise_temp_dir", "/content/temp_noise")
    
    original_config = MODEL_CONFIGS[mode].copy()
    original_config["fold"] = fold
    original_config["checkpoint_path"] = original_config["checkpoint_path"].format(fold=fold)

    base_img_dir = BASE_CONFIG["image_folder"]
    base_lbl_dir = BASE_CONFIG["label_folder"]
    
    case_plot_dir = os.path.join(original_config["output_folder"], "noise_experiments", case_id)
    os.makedirs(case_plot_dir, exist_ok=True)
    
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    ext = ".nii.gz" if os.path.exists(os.path.join(base_img_dir, f"{case_id}_0000.nii.gz")) else ".nii"

    slice_idx = None 
    AXIAL_AXIS = 0 # Đồng bộ chuẩn Axial với Inference

    for noise_type, levels in experiments_map.items():
        print(f"\n🔥 Running: {noise_type}")
        all_results_for_plot = {}
        
        for level in levels:
            print(f"   ⚡ Simulating {noise_type} Level = {level}...")
            
            curr_input_dir = os.path.join(temp_dir, f"{noise_type.replace(' ', '_')}_level_{level}")
            os.makedirs(curr_input_dir, exist_ok=True)
            
            affine = None
            for i in range(4):
                fname = f"{case_id}_{i:04d}{ext}"
                src_path = os.path.join(base_img_dir, fname)
                dst_path = os.path.join(curr_input_dir, fname)
                
                img = nib.load(src_path)
                data = img.get_fdata()
                affine = img.affine
                
                if noise_type == "Gaussian Noise": noisy_data = add_gaussian_noise(data, sigma=level)
                elif noise_type == "Gaussian Blur": noisy_data = add_gaussian_blur(data, sigma=level)
                elif noise_type == "Motion Ghost": noisy_data = add_motion_ghosting(data, num_ghosts=int(level), intensity=0.5)
                else: noisy_data = data 
                
                save_temp_nifti(noisy_data, affine, dst_path)

            # Inference
            temp_config = original_config.copy()
            temp_config["image_folder"] = curr_input_dir
            temp_config["label_folder"] = base_lbl_dir
            temp_config["save_3d_nifti"] = False
            
            engine = EDLInferenceEngine(temp_config)
            data_crop, seg_crop, pred_crop, unc_dict, _ = engine.process_case(case_id)
            
            if pred_crop is None: continue
            
            dices = calculate_dice_all_classes(pred_crop, seg_crop[0])
            print(f"      📊 Dice: WT={dices['WT']:.2f} | TC={dices['TC']:.2f} | ET={dices['ET']:.2f}")

            # Chọn Slice tự động (Tính trên trục Axial)
            if slice_idx is None:
                axes_to_sum = tuple([i for i in (0,1,2) if i != AXIAL_AXIS])
                tumor_dist = np.sum(seg_crop[0] > 0, axis=axes_to_sum)
                tumor_slices = np.where(tumor_dist > 0)[0]
                slice_idx = tumor_slices[int(len(tumor_slices) * 0.50)] if len(tumor_slices) > 0 else seg_crop[0].shape[AXIAL_AXIS]//2
                print(f"      📸 Selected Axial Slice: {slice_idx}")

            # Xử lý an toàn cho Baseline (unc_dict không phải dictionary)
            is_dict = isinstance(unc_dict, dict)
            total_unc = unc_dict["total"] if is_dict and "total" in unc_dict else (unc_dict if not is_dict else np.zeros_like(pred_crop))
            alea_unc = unc_dict.get("aleatoric") if is_dict else None
            epis_unc = unc_dict.get("epistemic") if is_dict else None

            all_results_for_plot[level] = {
                "mri": data_crop[0],
                "gt": seg_crop[0],
                "pred": pred_crop,
                "total": total_unc,
                "aleatoric": alea_unc,
                "epistemic": epis_unc,
                "dices": dices
            }

        # Vẽ hình
        if all_results_for_plot:
            safe_name = noise_type.replace(" ", "_")
            out_path = os.path.join(case_plot_dir, f"combined_{safe_name}.png")
            plot_combined_noise_levels(all_results_for_plot, case_id, slice_idx, out_path, view_axis=AXIAL_AXIS)
        else:
            print(f"❌ No results for {noise_type}")

    print(f"\n✅ All experiments completed for {case_id}!")