"""
🎨 VISUALIZER MODULE (UPDATED V5 - HYBRID & ROBUST)
Module chuyên trách vẽ biểu đồ, hỗ trợ cả chế độ Uncertainty đơn (cũ) và phân rã (mới).
Tự động thích ứng dựa trên dữ liệu đầu vào.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_dice_2d(pred_slice, gt_slice):
    """Tính Dice Score 2D nhanh để hiển thị trên tiêu đề ảnh"""
    p = (pred_slice > 0).astype(np.float32)
    g = (gt_slice > 0).astype(np.float32)
    intersection = np.sum(p * g)
    sum_areas = np.sum(p) + np.sum(g)
    if sum_areas == 0: return 1.0
    return (2.0 * intersection) / sum_areas

def visualize_comparison(case_id, mri_data, gt_data, pred_data, uncertainty_data, config, slice_idx=None):
    """
    Hàm vẽ đa năng:
    - Nếu uncertainty_data là dict (có aleatoric/epistemic) -> Vẽ 5 hình.
    - Nếu uncertainty_data là array (hoặc dict chỉ có total) -> Vẽ 4 hình (tương thích ngược).
    """
    
    # --- 1. AUTO-SELECT SLICE (LOGIC CŨ - ROBUST) ---
    # Tự động chọn lát cắt có khối u lớn nhất để hiển thị
    if slice_idx is None:
        # Tính tổng pixel theo các trục để tìm slice có nhiều thông tin nhất
        sums_gt = np.sum(gt_data, axis=(0, 1, 2))
        sums_pred = np.sum(pred_data, axis=(0, 1))
        
        if sums_gt.max() > 0: slice_idx = np.argmax(sums_gt)
        elif sums_pred.max() > 0: slice_idx = np.argmax(sums_pred)
        else: slice_idx = gt_data.shape[3] // 2 # Fallback: Lấy giữa não

    print(f"    📸 Drawing Slice: {slice_idx}")

    # --- 2. PREPARE BASIC DATA ---
    # Xoay .T để ảnh hiển thị đúng chiều (người nhìn thẳng vào mặt)
    img_slice = mri_data[0, :, :, slice_idx].T
    gt_slice = gt_data[0, :, :, slice_idx].T 
    pred_slice = pred_data[:, :, slice_idx].T
    
    dice = calculate_dice_2d(pred_slice, gt_slice)
    ratio = (np.sum(pred_slice>0) / np.sum(gt_slice>0) * 100) if np.sum(gt_slice>0) > 0 else 0

    # --- 3. DETERMINE MODE (LOGIC MỚI) ---
    # Kiểm tra xem dữ liệu uncertainty là loại nào
    is_decomposition = False
    if isinstance(uncertainty_data, dict):
        if "aleatoric" in uncertainty_data and "epistemic" in uncertainty_data:
            is_decomposition = True
            aleatoric_slice = uncertainty_data["aleatoric"][:, :, slice_idx].T
            epistemic_slice = uncertainty_data["epistemic"][:, :, slice_idx].T
        elif "total" in uncertainty_data:
            # Trường hợp dict nhưng chỉ có total
            unc_slice = uncertainty_data["total"][:, :, slice_idx].T
        else:
            # Fallback
            unc_slice = np.zeros_like(pred_slice)
    else:
        # Trường hợp legacy (numpy array)
        unc_slice = uncertainty_data[:, :, slice_idx].T

    # --- 4. PLOTTING ---
    if is_decomposition:
        # === CHẾ ĐỘ 5 CỘT (EDL MỚI) ===
        fig, ax = plt.subplots(1, 5, figsize=config.get("figsize", (30, 6)))
        plt.suptitle(f"EDL Decomposition: {case_id} (Slice {slice_idx})", fontsize=18, y=0.98)
    else:
        # === CHẾ ĐỘ 4 CỘT (CŨ/BASIC) ===
        fig, ax = plt.subplots(1, 4, figsize=config.get("figsize", (24, 6)))
        plt.suptitle(f"Segmentation Result: {case_id} (Slice {slice_idx})", fontsize=16, y=0.98)

    # --- Cột 1: MRI ---
    ax[0].imshow(img_slice, cmap='gray', origin='lower')
    ax[0].set_title("MRI Input", fontsize=14, fontweight='bold')
    ax[0].axis('off')

    # --- Cột 2: Ground Truth ---
    ax[1].imshow(img_slice, cmap='gray', origin='lower', alpha=0.6)
    if np.any(gt_slice): 
        ax[1].imshow(gt_slice, cmap='Greens', origin='lower', alpha=0.6, interpolation='nearest')
    ax[1].set_title("Ground Truth", fontsize=14, fontweight='bold', color='green')
    ax[1].axis('off')

    # --- Cột 3: Prediction ---
    ax[2].imshow(img_slice, cmap='gray', origin='lower', alpha=0.6)
    if np.any(pred_slice): 
        ax[2].imshow(pred_slice, cmap='jet', origin='lower', alpha=0.5, interpolation='nearest')
    ax[2].set_title(f"AI Prediction\nDice: {dice:.1%} | Area: {ratio:.0f}%", fontsize=14, fontweight='bold', color='blue')
    ax[2].axis('off')

    if is_decomposition:
        # --- Cột 4: Aleatoric (Nhiễu dữ liệu) ---
        # Không set vmin/vmax cứng để thấy rõ độ tương phản
        im1 = ax[3].imshow(aleatoric_slice, cmap='hot', origin='lower') 
        ax[3].set_title("Aleatoric (Data Noise)\n(Viền khối u, ảnh mờ)", fontsize=14, fontweight='bold', color='orange')
        ax[3].axis('off')
        plt.colorbar(im1, ax=ax[3], fraction=0.046, pad=0.04)

        # --- Cột 5: Epistemic (Mô hình không biết) ---
        im2 = ax[4].imshow(epistemic_slice, cmap='hot', origin='lower')
        ax[4].set_title("Epistemic (Model Uncertainty)\n(Vùng lạ, hiếm gặp)", fontsize=14, fontweight='bold', color='red')
        ax[4].axis('off')
        plt.colorbar(im2, ax=ax[4], fraction=0.046, pad=0.04)
    else:
        # --- Cột 4 (Cũ): Total Uncertainty ---
        # Code cũ set vmax=1.0, giữ nguyên để tương thích
        im = ax[3].imshow(unc_slice, cmap='hot', origin='lower', vmin=0, vmax=1.0)
        ax[3].set_title("Uncertainty Map", fontsize=14, fontweight='bold', color='red')
        ax[3].axis('off')
        plt.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)

    # --- 5. SAVE & SHOW ---
    try:
        if config.get("save_2d_snapshot", False):
            os.makedirs(config["output_folder"], exist_ok=True)
            save_path = os.path.join(config["output_folder"], f"{case_id}_slice{slice_idx}_viz.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            print(f"    ✅ Saved Snapshot: {save_path}")
        
        if config.get("show_on_screen", False):
            plt.show()
    except Exception as e:
        print(f"⚠️ Error saving/showing image: {e}")
    finally:
        plt.close() # Quan trọng: Giải phóng bộ nhớ để không bị tràn RAM khi chạy nhiều ảnh