"""
🎨 PLOTTING FOR NOISE EXPERIMENT (COMBINED VIEW - OVERLAY FIX)
Vẽ biểu đồ tổng hợp: Prediction và GT đều được vẽ chồng (Overlay) lên MRI nền.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_combined_noise_levels(all_results, case_id, slice_idx, output_path):
    """
    Vẽ ảnh tổng hợp dạng lưới (Grid).
    Rows: Các mức nhiễu.
    Cols: Input MRI | Ground Truth | Prediction | Total | Aleatoric | Epistemic.
    """
    # Sắp xếp mức nhiễu tăng dần
    noise_levels = sorted(all_results.keys())
    
    rows = len(noise_levels)
    cols = 6 
    
    fig, axes = plt.subplots(rows, cols, figsize=(30, 4.5 * rows))
    plt.suptitle(f"Robustness Analysis: Gaussian Noise Impact - {case_id} (Slice {slice_idx})", fontsize=24, y=0.99)
    
    headers = ["Input MRI (Noisy)", "Ground Truth", "Prediction", "Total Unc", "Aleatoric (Data)", "Epistemic (Model)"]
    
    # Helper xoay ảnh
    def get_slice(vol):
        if vol.ndim == 4: return vol[0, :, :, slice_idx].T
        return vol[:, :, slice_idx].T

    for idx, sigma in enumerate(noise_levels):
        res = all_results[sigma]
        
        dice_info = f"Dice\nWT: {res['dices']['WT']:.2f}\nTC: {res['dices']['TC']:.2f}\nET: {res['dices']['ET']:.2f}"
        
        if idx == 0:
            for ax, title in zip(axes[0], headers):
                ax.set_title(title, fontsize=16, fontweight='bold')

        # 1. MRI
        mri_slice = get_slice(res["mri"])
        axes[idx, 0].imshow(mri_slice, cmap='gray', origin='lower')
        axes[idx, 0].set_ylabel(f"Level = {sigma}\n\n{dice_info}", fontsize=14, fontweight='bold', rotation=0, labelpad=60, va='center')

        # 2. Ground Truth (Overlay)
        gt_slice = get_slice(res["gt"])
        # [FIX] Làm trong suốt các pixel nền (giá trị 0)
        gt_masked = np.ma.masked_where(gt_slice == 0, gt_slice)
        
        axes[idx, 1].imshow(mri_slice, cmap='gray', origin='lower') # Vẽ nền MRI trước
        axes[idx, 1].imshow(gt_masked, cmap='Greens', origin='lower', interpolation='nearest', alpha=0.7) # Vẽ lớp màu lên trên
        
        # 3. Prediction (Overlay - [NEW])
        pred_slice = get_slice(res["pred"])
        # [FIX] Làm trong suốt nền dự đoán để thấy MRI bên dưới
        pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
        
        axes[idx, 2].imshow(mri_slice, cmap='gray', origin='lower') # Vẽ nền MRI trước
        axes[idx, 2].imshow(pred_masked, cmap='jet', origin='lower', interpolation='nearest', alpha=0.6) # Vẽ lớp màu
        
        # 4. Total Uncertainty
        axes[idx, 3].imshow(get_slice(res["total"]), cmap='hot', origin='lower', vmin=0, vmax=1.0)
        
        # 5. Aleatoric
        axes[idx, 4].imshow(get_slice(res["aleatoric"]), cmap='hot', origin='lower')
        
        # 6. Epistemic
        axes[idx, 5].imshow(get_slice(res["epistemic"]), cmap='hot', origin='lower')
        
        for ax in axes[idx]: 
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Saved combined plot: {output_path}")