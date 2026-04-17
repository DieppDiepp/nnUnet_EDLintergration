"""
🎨 PLOTTING FOR NOISE EXPERIMENT (UPDATED V2 - PAPER READY)
Vẽ biểu đồ lưới (Grid) so sánh độ kháng nhiễu.
Tự động áp dụng nnU-Net Overlay. Tự động chuyển đổi 6 cột (EDL) hoặc 4 cột (Baseline).
"""
import matplotlib.pyplot as plt
import numpy as np

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

def extract_2d_slice(data, axis, idx):
    if data is None: return None
    if data.ndim == 4: data = data[0] 
    if axis == 0: slice_2d = data[idx, :, :]
    elif axis == 1: slice_2d = data[:, idx, :]
    else: slice_2d = data[:, :, idx]
    return np.rot90(slice_2d)

def plot_combined_noise_levels(all_results, case_id, slice_idx, output_path, view_axis=0):
    noise_levels = sorted(all_results.keys())
    rows = len(noise_levels)
    
    # Kiểm tra xem Model có Decomposition không (EDL vs Baseline)
    first_res = all_results[noise_levels[0]]
    is_decomp = first_res.get("aleatoric") is not None
    
    cols = 6 if is_decomp else 4
    figsize = (30, 4.5 * rows) if is_decomp else (20, 4.5 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    if rows == 1: axes = np.expand_dims(axes, axis=0)
    
    plt.suptitle(f"Robustness Analysis: Noise Impact - {case_id} (Slice {slice_idx})", fontsize=24, y=0.95, fontweight='bold')
    
    headers = ["Input MRI (Noisy)", "Ground Truth", "Prediction", "Total Uncertainty"]
    if is_decomp: headers.extend(["Aleatoric (Data Noise)", "Epistemic (Model Uncertainty)"])

    for idx, sigma in enumerate(noise_levels):
        res = all_results[sigma]
        
        dice_info = f"Noise = {sigma}\n\nWT: {res['dices']['WT']:.2f}\nTC: {res['dices']['TC']:.2f}\nET: {res['dices']['ET']:.2f}"
        
        if idx == 0:
            for ax, title in zip(axes[0], headers):
                ax.set_title(title, fontsize=16, fontweight='bold')

        # 1. Trích xuất 2D Slice theo đúng trục
        mri_slice = extract_2d_slice(res["mri"], view_axis, slice_idx)
        gt_slice = extract_2d_slice(res["gt"], view_axis, slice_idx)
        pred_slice = extract_2d_slice(res["pred"], view_axis, slice_idx)
        total_slice = extract_2d_slice(res["total"], view_axis, slice_idx)
        
        # 2. Tạo ảnh Overlay tuyệt đẹp
        gt_overlay = generate_nnunet_overlay(mri_slice, gt_slice)
        pred_overlay = generate_nnunet_overlay(mri_slice, pred_slice)

        # Cột 1: MRI
        axes[idx, 0].imshow(mri_slice, cmap='gray', origin='lower', interpolation='nearest')
        axes[idx, 0].set_ylabel(dice_info, fontsize=15, fontweight='bold', rotation=0, labelpad=75, va='center')

        # Cột 2 & 3: GT & Prediction
        axes[idx, 1].imshow(gt_overlay, origin='lower', interpolation='nearest')
        axes[idx, 2].imshow(pred_overlay, origin='lower', interpolation='nearest')

        # Cột 4: Total Uncertainty
        im0 = axes[idx, 3].imshow(total_slice, cmap='hot', origin='lower', interpolation='nearest')
        plt.colorbar(im0, ax=axes[idx, 3], fraction=0.046, pad=0.04)

        if is_decomp:
            alea_slice = extract_2d_slice(res["aleatoric"], view_axis, slice_idx)
            epis_slice = extract_2d_slice(res["epistemic"], view_axis, slice_idx)
            
            # Cột 5 & 6
            im1 = axes[idx, 4].imshow(alea_slice, cmap='hot', origin='lower', interpolation='nearest')
            plt.colorbar(im1, ax=axes[idx, 4], fraction=0.046, pad=0.04)
            
            im2 = axes[idx, 5].imshow(epis_slice, cmap='hot', origin='lower', interpolation='nearest')
            plt.colorbar(im2, ax=axes[idx, 5], fraction=0.046, pad=0.04)

        for ax in axes[idx]: 
            ax.set_xticks([])
            ax.set_yticks([])

    fig.subplots_adjust(top=0.9, bottom=0.05, left=0.1, right=0.95)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    ✅ Saved Plot: {output_path}")