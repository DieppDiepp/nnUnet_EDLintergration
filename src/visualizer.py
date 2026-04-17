"""
🎨 VISUALIZER MODULE (UPDATED V10 - MULTI-PLANES & SMART FOLDERS)
- Hỗ trợ cắt theo 3 trục không gian (Axial, Coronal, Sagittal).
- Tự động tạo thư mục riêng cho từng case_id.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def hex_to_rgb(hex_str: str):
    assert len(hex_str) == 6
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))

def generate_nnunet_overlay(input_image: np.ndarray, segmentation: np.ndarray, overlay_intensity: float = 0.6):
    image = np.copy(input_image)
    if image.ndim == 2:
        image = np.tile(image[:, :, None], (1, 1, 3))

    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min) * 255.0
    else:
        image = np.zeros_like(image)

    color_mapping = {1: "e6194B", 2: "3cb44b", 3: "ffe119"}
    color_cycle = ["4363d8", "f58231", "911eb4", "42d4f4", "f032e6"]

    uniques = np.sort(np.unique(segmentation))
    uniques = uniques[uniques > 0]

    for idx, l in enumerate(uniques):
        hex_color = color_mapping.get(l, color_cycle[idx % len(color_cycle)])
        rgb_color = np.array(hex_to_rgb(hex_color))
        image[segmentation == l] += overlay_intensity * rgb_color

    img_max = image.max()
    if img_max > 0:
        image = image / img_max * 255.0
        
    return image.astype(np.uint8)

def calculate_dice_2d(pred_slice, gt_slice):
    p = (pred_slice > 0).astype(np.float32)
    g = (gt_slice > 0).astype(np.float32)
    intersection = np.sum(p * g)
    sum_areas = np.sum(p) + np.sum(g)
    if sum_areas == 0: return 1.0
    return (2.0 * intersection) / sum_areas

def extract_2d_slice(data, axis, idx):
    """Trích xuất mặt phẳng 2D và xoay ảnh cho thuận mắt"""
    if data is None: return None # <--- Thêm dòng bảo vệ này
    
    # Nếu data có kênh (C, X, Y, Z), bỏ kênh đi lấy lõi 3D
    if data.ndim == 4: data = data[0] 
    
    if axis == 0:
        slice_2d = data[idx, :, :]
    elif axis == 1:
        slice_2d = data[:, idx, :]
    else: # axis == 2
        slice_2d = data[:, :, idx]
        
    # Quay ảnh 90 độ thay vì transpose (.T) để não đứng thẳng chuẩn y khoa
    return np.rot90(slice_2d)

def visualize_comparison(case_id, mri_data, gt_data, pred_data, uncertainty_data, config, 
                         slice_idx=None, view_axis=0, suffix_name="best"):
    """
    view_axis: 0, 1, hoặc 2 (Trục không gian muốn cắt). Thường 0 hoặc 2 là Axial (tùy file gốc).
    suffix_name: Tên hậu tố để lưu file (vd: 'axial_50pct', 'sagittal_worst').
    """
    
    # --- 1. AUTO-SELECT SLICE (Fallback) ---
    if slice_idx is None:
        sums_gt = np.sum(gt_data, axis=tuple([i for i in (0,1,2,3) if i != view_axis+1]))
        slice_idx = np.argmax(sums_gt) if sums_gt.max() > 0 else gt_data.shape[view_axis+1] // 2 

    # --- 2. PREPARE BASIC DATA ---
    img_slice = extract_2d_slice(mri_data, view_axis, slice_idx)
    gt_slice = extract_2d_slice(gt_data, view_axis, slice_idx)
    pred_slice = extract_2d_slice(pred_data, view_axis, slice_idx)
    
    dice = calculate_dice_2d(pred_slice, gt_slice)
    ratio = (np.sum(pred_slice>0) / np.sum(gt_slice>0) * 100) if np.sum(gt_slice>0) > 0 else 0

    # --- 3. DETERMINE UNCERTAINTY MODE ---
    is_decomposition = False
    if isinstance(uncertainty_data, dict):
        # Dùng .get() is not None để đảm bảo giá trị bên trong thực sự tồn tại
        if uncertainty_data.get("aleatoric") is not None and uncertainty_data.get("epistemic") is not None:
            is_decomposition = True
            aleatoric_slice = extract_2d_slice(uncertainty_data["aleatoric"], view_axis, slice_idx)
            epistemic_slice = extract_2d_slice(uncertainty_data["epistemic"], view_axis, slice_idx)
            total_slice = extract_2d_slice(uncertainty_data.get("total", np.zeros_like(pred_slice)), view_axis, slice_idx)
        elif uncertainty_data.get("total") is not None:
            unc_slice = extract_2d_slice(uncertainty_data["total"], view_axis, slice_idx)
        else:
            unc_slice = np.zeros_like(pred_slice)
    else:
        unc_slice = extract_2d_slice(uncertainty_data, view_axis, slice_idx)

    # --- 4. TẠO ẢNH OVERLAY ---
    gt_overlay = generate_nnunet_overlay(img_slice, gt_slice)
    pred_overlay = generate_nnunet_overlay(img_slice, pred_slice)

    # --- 5. PLOTTING ---
    if is_decomposition:
        fig, ax = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'wspace': 0.1, 'hspace': 0.2})
        
        plt.suptitle(
            f"EDL Analysis: {case_id} ({suffix_name})\n"
            f"Dice Score: {dice:.1%} | Tumor Area Ratio: {ratio:.0f}%", 
            fontsize=18, fontweight='bold', y=0.98
        )

        ax[0, 0].imshow(img_slice, cmap='gray', origin='lower', interpolation='nearest')
        ax[0, 0].set_title("MRI Input", fontsize=15, fontweight='bold')
        ax[0, 0].axis('off')

        ax[0, 1].imshow(gt_overlay, origin='lower', interpolation='nearest')
        ax[0, 1].set_title("Ground Truth", fontsize=15, fontweight='bold', color='green')
        ax[0, 1].axis('off')

        ax[0, 2].imshow(pred_overlay, origin='lower', interpolation='nearest')
        ax[0, 2].set_title("AI Prediction", fontsize=15, fontweight='bold', color='blue')
        ax[0, 2].axis('off')

        im0 = ax[1, 0].imshow(total_slice, cmap='hot', origin='lower', interpolation='nearest')
        ax[1, 0].set_title("Total Uncertainty", fontsize=15, fontweight='bold', color='red')
        ax[1, 0].axis('off')
        plt.colorbar(im0, ax=ax[1, 0], fraction=0.046, pad=0.04)

        im1 = ax[1, 1].imshow(aleatoric_slice, cmap='hot', origin='lower', interpolation='nearest') 
        ax[1, 1].set_title("Aleatoric Uncertainty", fontsize=15, fontweight='bold', color='darkorange')
        ax[1, 1].axis('off')
        plt.colorbar(im1, ax=ax[1, 1], fraction=0.046, pad=0.04)

        im2 = ax[1, 2].imshow(epistemic_slice, cmap='hot', origin='lower', interpolation='nearest')
        ax[1, 2].set_title("Epistemic Uncertainty", fontsize=15, fontweight='bold', color='purple')
        ax[1, 2].axis('off')
        plt.colorbar(im2, ax=ax[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    else:
        fig, ax = plt.subplots(1, 4, figsize=(20, 6), gridspec_kw={'wspace': 0.1})
        plt.suptitle(
            f"Segmentation Result: {case_id} ({suffix_name})\n"
            f"Dice: {dice:.1%} | Area: {ratio:.0f}%", 
            fontsize=16, fontweight='bold', y=1.05
        )

        ax[0].imshow(img_slice, cmap='gray', origin='lower', interpolation='nearest')
        ax[0].set_title("MRI Input", fontsize=14, fontweight='bold')
        ax[0].axis('off')

        ax[1].imshow(gt_overlay, origin='lower', interpolation='nearest')
        ax[1].set_title("Ground Truth", fontsize=14, fontweight='bold', color='green')
        ax[1].axis('off')

        ax[2].imshow(pred_overlay, origin='lower', interpolation='nearest')
        ax[2].set_title("AI Prediction", fontsize=14, fontweight='bold', color='blue')
        ax[2].axis('off')

        im = ax[3].imshow(unc_slice, cmap='hot', origin='lower', vmin=0, vmax=1.0, interpolation='nearest')
        ax[3].set_title("Uncertainty Map", fontsize=14, fontweight='bold', color='red')
        ax[3].axis('off')
        plt.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()

    # --- 6. TẠO FOLDER ĐỘC LẬP & LƯU FILE ---
    try:
        if config.get("save_2d_snapshot", False):
            # Tạo thư mục con có tên là case_id (VD: output/BRATS_004/)
            case_folder = os.path.join(config["output_folder"], case_id)
            os.makedirs(case_folder, exist_ok=True)
            
            # Lưu file vào thư mục đó
            save_path = os.path.join(case_folder, f"{suffix_name}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=200, facecolor='white')
            print(f"    ✅ Saved: {save_path}")
        
        if config.get("show_on_screen", False):
            plt.show()
    except Exception as e:
        print(f"⚠️ Error saving/showing image: {e}")
    finally:
        plt.close()