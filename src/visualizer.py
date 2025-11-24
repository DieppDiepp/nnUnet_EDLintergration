"""
🎨 VISUALIZER MODULE
Chuyên trách việc vẽ biểu đồ và lưu ảnh.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_dice_2d(pred_slice, gt_slice):
    """Tính Dice Score 2D đơn giản cho visualizer"""
    p = (pred_slice > 0).astype(np.float32)
    g = (gt_slice > 0).astype(np.float32)
    intersection = np.sum(p * g)
    sum_areas = np.sum(p) + np.sum(g)
    if sum_areas == 0: return 1.0
    return (2.0 * intersection) / sum_areas

def visualize_comparison(case_id, mri_data, gt_data, pred_data, uncertainty, config, slice_idx=None):
    """Vẽ 4 hình: MRI | GT | Pred | Uncertainty"""
    
    # 1. Tự động chọn Slice nếu không chỉ định
    if slice_idx is None:
        sums_gt = np.sum(gt_data, axis=(0, 1, 2))
        sums_pred = np.sum(pred_data, axis=(0, 1))
        
        if sums_gt.max() > 0: slice_idx = np.argmax(sums_gt)
        elif sums_pred.max() > 0: slice_idx = np.argmax(sums_pred)
        else: slice_idx = gt_data.shape[3] // 2

    print(f"    📸 Drawing Slice: {slice_idx}")

    # 2. Chuẩn bị dữ liệu (Xoay .T)
    # mri, gt: [C, X, Y, Z] -> lấy channel 0
    img_slice = mri_data[0, :, :, slice_idx].T
    gt_slice = gt_data[0, :, :, slice_idx].T 
    # pred, unc: [X, Y, Z]
    pred_slice = pred_data[:, :, slice_idx].T
    unc_slice = uncertainty[:, :, slice_idx].T

    # 3. Tính chỉ số hiển thị
    dice = calculate_dice_2d(pred_slice, gt_slice)
    ratio = (np.sum(pred_slice>0) / np.sum(gt_slice>0) * 100) if np.sum(gt_slice>0) > 0 else 0

    # 4. Vẽ
    fig, ax = plt.subplots(1, 4, figsize=config.get("figsize", (24, 6)))
    plt.suptitle(f"Case: {case_id} | Slice: {slice_idx}", fontsize=16, y=0.98)

    # MRI
    ax[0].imshow(img_slice, cmap='gray', origin='lower')
    ax[0].set_title("MRI Input", fontsize=12, fontweight='bold')
    ax[0].axis('off')

    # GT
    ax[1].imshow(img_slice, cmap='gray', origin='lower', alpha=0.6)
    if np.any(gt_slice): 
        ax[1].imshow(gt_slice, cmap='Greens', origin='lower', alpha=0.6, interpolation='nearest')
    ax[1].set_title("Ground Truth", fontsize=12, fontweight='bold', color='green')
    ax[1].axis('off')

    # Pred
    ax[2].imshow(img_slice, cmap='gray', origin='lower', alpha=0.6)
    if np.any(pred_slice): 
        ax[2].imshow(pred_slice, cmap='jet', origin='lower', alpha=0.5, interpolation='nearest')
    ax[2].set_title(f"AI Pred\\nDice: {dice:.1%} | Area: {ratio:.0f}%", fontsize=12, fontweight='bold', color='blue')
    ax[2].axis('off')

    # Uncertainty
    im = ax[3].imshow(unc_slice, cmap='hot', origin='lower', vmin=0, vmax=1.0)
    ax[3].set_title("Uncertainty Map", fontsize=12, fontweight='bold', color='red')
    ax[3].axis('off')
    plt.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)

    # 5. Lưu và Hiện
    if config.get("save_2d_snapshot", False):
        os.makedirs(config["output_folder"], exist_ok=True)
        save_path = os.path.join(config["output_folder"], f"{case_id}_slice{slice_idx}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        print(f"    ✅ Saved Snapshot: {save_path}")

    if config.get("show_on_screen", False):
        plt.show()
    plt.close()