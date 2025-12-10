"""
🏃 OOD RUNNER (ALIGNED & ORGANIZED)
Chạy thí nghiệm OOD, lưu folder theo Case ID, dùng ảnh crop để vẽ.
"""
import os
import shutil
import nibabel as nib
import numpy as np
from src.config import BASE_CONFIG, MODEL_CONFIGS
from src.edl_engine import EDLInferenceEngine
# Import hàm tính Dice từ analysis
from src.analysis.utils import get_binary_mask, compute_dice_score_binary
from .utils import add_artifact, apply_structural_mutation, apply_intensity_shift, save_temp_nifti
from .plotting import plot_ood_results

def calculate_dice_all_classes(pred, gt):
    scores = {}
    for target in ["WT", "TC", "ET"]:
        p_bin = get_binary_mask(pred, target)
        g_bin = get_binary_mask(gt, target)
        scores[target] = compute_dice_score_binary(p_bin, g_bin)
    return scores

def run_ood_experiment(case_id, mode='edl'):
    print(f"🧪 OOD EXPERIMENT (TARGETED): {case_id}")
    
    # ... (Phần Config giữ nguyên) ...
    ood_cfg = BASE_CONFIG.get("ood_experiment_config", {})
    active_type = ood_cfg.get("active_type", "Artificial Artifact")
    settings = ood_cfg.get("settings", {})
    
    if active_type == "All": target_settings = settings
    elif active_type in settings: target_settings = {active_type: settings[active_type]}
    else: return

    # Setup paths
    temp_dir = BASE_CONFIG.get("ood_temp_dir", "/content/temp_ood")
    original_config = MODEL_CONFIGS[mode]
    base_img_dir = BASE_CONFIG["image_folder"]
    base_lbl_dir = BASE_CONFIG["label_folder"]
    
    case_plot_dir = os.path.join(original_config["output_folder"], "ood_experiments", case_id)
    os.makedirs(case_plot_dir, exist_ok=True)
    
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Detect ext
    ext = ".nii"
    if os.path.exists(os.path.join(base_img_dir, f"{case_id}_0000.nii.gz")): ext = ".nii.gz"
    
    slice_idx = None

    # [NEW] Load GT trước để xác định vị trí đặt dị vật
    gt_data_np = None
    try:
        gt_path = os.path.join(base_lbl_dir, f"{case_id}.nii.gz")
        if not os.path.exists(gt_path): gt_path = os.path.join(base_lbl_dir, f"{case_id}.nii")
        if os.path.exists(gt_path):
            gt_data_np = nib.load(gt_path).get_fdata()
            # print("   ✅ GT Loaded for targeting.")
    except Exception as e:
        print(f"   ⚠️ Could not load GT for targeting: {e}")

    # MAIN LOOP
    for group_name, variants in target_settings.items():
        print(f"\n🔥 Running OOD Group: {group_name}")
        all_results = {}
        
        for variant in variants:
            print(f"   ⚡ Simulating: {variant}...")
            curr_input_dir = os.path.join(temp_dir, f"{group_name}_{variant}")
            os.makedirs(curr_input_dir, exist_ok=True)
            
            # A. Tạo dữ liệu OOD
            for i in range(4):
                fname = f"{case_id}_{i:04d}{ext}"
                src_path = os.path.join(base_img_dir, fname)
                dst_path = os.path.join(curr_input_dir, fname)
                
                img = nib.load(src_path)
                data = img.get_fdata()
                affine = img.affine
                
                if group_name == "Artificial Artifact":
                    # [FIX] Truyền thêm gt_data_np vào đây
                    ood_data = add_artifact(data, gt_data=gt_data_np, type=variant)
                elif group_name == "Structural Mutation":
                    ood_data = apply_structural_mutation(data, type=variant)
                elif group_name == "Intensity Shift":
                    ood_data = apply_intensity_shift(data, factor=variant)
                else:
                    ood_data = data
                
                save_temp_nifti(ood_data, affine, dst_path)

            # B. Inference (Giữ nguyên)
            temp_config = original_config.copy()
            temp_config["image_folder"] = curr_input_dir
            temp_config["label_folder"] = base_lbl_dir
            temp_config["save_3d_nifti"] = False
            
            engine = EDLInferenceEngine(temp_config)
            data_crop, seg_crop, pred_crop, unc_dict, _ = engine.process_case(case_id)
            
            if pred_crop is None: continue
            
            dices = calculate_dice_all_classes(pred_crop, seg_crop[0])
            
            if slice_idx is None:
                slice_idx = np.argmax(np.sum(seg_crop[0] > 0, axis=(0, 1)))
            
            all_results[str(variant)] = {
                "mri": data_crop[0],
                "gt": seg_crop[0],
                "pred": pred_crop,
                "total": unc_dict["total"],
                "aleatoric": unc_dict["aleatoric"],
                "epistemic": unc_dict["epistemic"],
                "dices": dices
            }
            
        if all_results:
            out_name = f"combined_{group_name.replace(' ', '_')}.png"
            plot_ood_results(all_results, group_name, case_id, slice_idx, os.path.join(case_plot_dir, out_name))