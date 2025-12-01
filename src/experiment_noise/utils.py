"""
🛠️ UTILITIES FOR NOISE EXPERIMENT (FIXED MASKING)
Các hàm phụ trợ: Thêm nhiễu (có Mask), Lưu file NIfTI.
"""
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

def add_gaussian_noise(data, sigma):
    """Nhiễu hạt (Cũ)"""
    if sigma == 0: return data
    noise = np.random.normal(0, sigma, data.shape)
    noisy_data = data + noise
    mask_background = data < 1e-5 
    noisy_data[mask_background] = 0
    return noisy_data

def add_gaussian_blur(data, sigma):
    """
    Làm mờ ảnh (Blur).
    Đây là 'sát thủ' của Segmentation vì nó xóa nhòa ranh giới u/nền.
    sigma: Độ mờ (thường từ 0.5 đến 2.0 là đã rất mờ rồi).
    """
    if sigma == 0: return data
    # Làm mờ trên từng kênh không gian (x, y, z)
    return gaussian_filter(data, sigma=sigma)

def add_motion_ghosting(data, num_ghosts=2, intensity=0.5, axis=1):
    """
    Giả lập nhiễu chuyển động (Ghosting Artifact) trong k-space.
    Cơ chế: Biến đổi Fourier -> Xóa/Lệch pha -> Biến đổi ngược.
    """
    if num_ghosts == 0: return data
    
    # Chuyển sang miền tần số (k-space)
    k_space = np.fft.fftn(data)
    
    # Tạo ghosting bằng cách điều biến amplitude
    indices = np.arange(k_space.shape[axis])
    # Chỉ giữ lại các tần số tạo ghost
    mask = (indices % num_ghosts) == 0
    
    # Tạo slice index để áp dụng mask cho đúng trục
    slice_obj = [slice(None)] * k_space.ndim
    slice_obj[axis] = mask
    
    # Áp dụng nhiễu
    k_space_corrupted = k_space.copy()
    k_space_corrupted[tuple(slice_obj)] *= (1 + intensity)
    
    # Chuyển về miền không gian
    data_ghosted = np.abs(np.fft.ifftn(k_space_corrupted))
    
    # Khôi phục nền đen (để tránh artifact lan ra ngoài vùng background quá nhiều)
    mask_background = data < 1e-5
    data_ghosted[mask_background] = 0
    
    return data_ghosted

def save_temp_nifti(data, affine, path):
    try:
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        nib.save(img, path)
    except Exception as e:
        print(f"⚠️ Error saving: {e}")