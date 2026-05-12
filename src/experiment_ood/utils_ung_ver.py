# src/experiment_ood/utils.py
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

def get_roi_center(data):
    coords = np.argwhere(data > 0)
    if coords.size == 0: return np.array(data.shape) // 2
    return (coords.min(axis=0) + coords.max(axis=0)) // 2

def add_artifact(data, gt_data=None, type='Box_White'): # chèn dị vật
    modified_data = data.copy()
    shape = data.shape
    
    # 1. Targeting (Giữ nguyên logic của bạn)
    if gt_data is not None and np.sum(gt_data) > 0:
        center = get_roi_center(gt_data)
    else:
        center = get_roi_center(data)

    cx, cy, cz = center[0], center[1], center[2] # Lấy cả tâm trục Z
    
    # 2. Xử lý Cường độ an toàn (Dùng Percentile thay vì Max)
    # Lấy các pixel trong não (>0) để tính toán
    brain_pixels = data[data > 0]
    if len(brain_pixels) == 0: return modified_data
    
    p99_val = np.percentile(brain_pixels, 99)
    mean_val = np.mean(brain_pixels)
    
    if "White" in type:
        val = p99_val * 1.2  # Sáng hơn não bình thường một chút, nhưng không phá vỡ dải màu
    else:
        val = mean_val       # Xám như mô não
        
    # 3. Kích thước 3D thực thụ
    xy_size = 15     # Bán kính X, Y (tổng chiều dài 30px)
    z_size = 5       # Bán kính Z (tổng độ dày 10 slices, phù hợp với spacing MRI)
    
    # Giới hạn tọa độ để không bị văng ra ngoài mảng
    x_start, x_end = max(0, cx - xy_size), min(shape[0], cx + xy_size)
    y_start, y_end = max(0, cy - xy_size), min(shape[1], cy + xy_size)
    z_start, z_end = max(0, cz - z_size), min(shape[2], cz + z_size)

    # 4. Vẽ Dị Vật (Bỏ cái viền đen border_size=4 đi)
    if "Box" in type:
        # Tạo một mask cục bộ
        artifact_mask = np.zeros_like(data, dtype=bool)
        artifact_mask[x_start:x_end, y_start:y_end, z_start:z_end] = True
        
        # Ghi đè giá trị
        modified_data[artifact_mask] = val

    elif "Sphere" in type:
        x_grid, y_grid, z_grid = np.ogrid[:shape[0], :shape[1], :shape[2]]
        # Tính khoảng cách 3D (Có scale trục Z vì Z thường thưa hơn)
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2 + ((z_grid - cz) * (xy_size/z_size))**2
        
        mask_core = dist_sq <= xy_size**2
        modified_data[mask_core] = val
        
    # 5. [Trick quan trọng] Soft Blend để không kích hoạt Aleatoric Edge
    # Áp dụng Gaussian Blur NHẸ lên toàn bộ ảnh, nhưng CHỈ lấy phần rìa của dị vật
    blurred_data = gaussian_filter(modified_data, sigma=0.5)
    
    # Tìm vùng viền của dị vật (Dilation mask) để blend
    if "Box" in type or "Sphere" in type:
        # Cách đơn giản: Trộn đè (Blend) lại bản blur vào bản gốc
        modified_data = modified_data * 0.8 + blurred_data * 0.2

    # Đảm bảo phần ngoài não vẫn là 0 đen tuyệt đối
    modified_data[data == 0] = 0.0

    return modified_data


def add_mirror_artifact(data, gt_data):
    """
    Cấy ghép chéo mô não từ bán cầu đối diện với HÌNH DÁNG ĐỘNG (chuẩn khít 100%).
    Lật ngược não khỏe mạnh để đắp đè lên chính xác vị trí và hình dáng khối u.
    """
    # Rào chắn an toàn: Nếu không có file nhãn hoặc ca này không có khối u thì bỏ qua
    if gt_data is None or np.sum(gt_data) == 0:
        return data 
        
    modified_data = data.copy()
    
    # 1. Tạo "chiếc khuôn" cắt dán
    # Dù u có hình thù kỳ dị (amoeba, sao, giọt nước...), mask này sẽ ôm khít 100%
    tumor_mask = gt_data > 0
    
    # 2. Lật ngược toàn bộ ma trận não (Mirror Trái/Phải)
    # Với ảnh BraTS, trục 0 (Sagittal) phân chia 2 bán cầu. 
    # Khi lật, vùng não khỏe sẽ lật sang đúng vị trí của vùng u.
    flipped_data = np.flip(data, axis=0)
    
    # 3. Phẫu thuật cắt dán (Thay thế hoàn toàn bước tính toán tọa độ phức tạp)
    # Lấy tín hiệu từ não đã lật, nhét khít vào chiếc khuôn của khối u
    modified_data[tumor_mask] = flipped_data[tumor_mask]
    
    # 4. Che giấu "vết khâu" phẫu thuật (Soft Blend) để đánh lừa Aleatoric
    # Tạo một bản làm mờ toàn cục
    blurred_data = gaussian_filter(modified_data, sigma=1.0)
    
    # [QUAN TRỌNG] CHỈ trộn blend ở khu vực miếng dán (tumor_mask)
    # Giữ nguyên vẹn độ sắc nét 100% cho toàn bộ phần não gốc còn lại
    modified_data[tumor_mask] = (modified_data[tumor_mask] * 0.7) + (blurred_data[tumor_mask] * 0.3)
    
    # 5. Dọn dẹp phông nền (đảm bảo không có viền mờ tràn ra ngoài hộp sọ)
    modified_data[data == 0] = 0.0

    return modified_data

# ... (Giữ nguyên các hàm khác)
def apply_structural_mutation(data, type='Flip_Horizontal'):
    if type == 'Flip_Horizontal': return np.flip(data, axis=0) 
    elif type == 'Flip_Vertical': return np.flip(data, axis=1)
    return data

def apply_intensity_shift(data, factor):
    return data * float(factor)

def save_temp_nifti(data, affine, path):
    try:
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        nib.save(img, path)
    except: pass