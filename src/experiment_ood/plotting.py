"""
🎨 PLOTTING FOR OOD EXPERIMENT (FIXED SQUEEZE BUG)
Sửa lỗi crash khi chỉ có 1 biến thể (rows=1).
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_ood_results(all_results, group_name, case_id, slice_idx, output_path):
    """
    Vẽ Grid so sánh các biến thể trong cùng 1 nhóm OOD.
    """
    variants = sorted(all_results.keys())
    rows = len(variants)
    cols = 6 # MRI, GT, Pred, Total, Aleatoric, Epistemic
    
    # [FIX] squeeze=False đảm bảo axes luôn là mảng 2 chiều [rows, cols]
    # Kể cả khi rows=1, nó vẫn trả về mảng shape (1, 6) thay vì (6,)
    fig, axes = plt.subplots(rows, cols, figsize=(30, 5 * rows), squeeze=False)
    
    plt.suptitle(f"OOD Analysis: {group_name} - {case_id} (Slice {slice_idx})", fontsize=24, y=0.99)
    
    headers = ["OOD Input", "Ground Truth", "Prediction", "Total Unc", "Aleatoric (Data)", "Epistemic (Model)"]
    
    def get_slice(vol):
        if vol.ndim == 4: return vol[0, :, :, slice_idx].T
        return vol[:, :, slice_idx].T

    for idx, variant in enumerate(variants):
        res = all_results[variant]
        
        # Format tiêu đề Dice
        dice_info = f"Dice\nWT: {res['dices']['WT']:.2f}\nTC: {res['dices']['TC']:.2f}\nET: {res['dices']['ET']:.2f}"
        
        # Set tiêu đề cột cho hàng đầu tiên
        if idx == 0:
            # axes[0] bây giờ luôn là 1 mảng các axes (hàng đầu tiên), an toàn để zip
            for ax, title in zip(axes[0], headers):
                ax.set_title(title, fontsize=16, fontweight='bold')

        # 1. MRI (OOD)
        mri_slice = get_slice(res["mri"])
        axes[idx, 0].imshow(mri_slice, cmap='gray', origin='lower')
        axes[idx, 0].set_ylabel(f"{variant}\n\n{dice_info}", fontsize=12, fontweight='bold', rotation=0, labelpad=60, va='center')

        # 2. GT (Overlay)
        gt_slice = get_slice(res["gt"])
        gt_masked = np.ma.masked_where(gt_slice == 0, gt_slice)
        
        axes[idx, 1].imshow(mri_slice, cmap='gray', origin='lower') 
        axes[idx, 1].imshow(gt_masked, cmap='Greens', origin='lower', alpha=0.7, interpolation='nearest')

        # 3. Prediction (Overlay)
        pred_slice = get_slice(res["pred"])
        pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
        
        axes[idx, 2].imshow(mri_slice, cmap='gray', origin='lower')
        axes[idx, 2].imshow(pred_masked, cmap='jet', origin='lower', alpha=0.6, interpolation='nearest')

        # 4. Total Uncertainty
        axes[idx, 3].imshow(get_slice(res["total"]), cmap='hot', origin='lower', vmin=0, vmax=1.0)

        # 5. Aleatoric
        axes[idx, 4].imshow(get_slice(res["aleatoric"]), cmap='hot', origin='lower')

        # 6. Epistemic
        axes[idx, 5].imshow(get_slice(res["epistemic"]), cmap='hot', origin='lower')

        # Tắt trục tọa độ cho đẹp
        for ax in axes[idx]: 
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Saved OOD plot: {output_path}")