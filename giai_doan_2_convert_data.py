# File này chỉ cần chạy local - bằng CPU
# Mục đích: Chuyển đổi dữ liệu BraTS 2020 sang định dạng nnU-Net yêu cầu
# Đồng thời sửa lỗi nhãn (remap label) từ 4 lớp về 3 lớp như sau:
# - Lớp 0: Background
# - Lớp 1: NCR/NET (Non-Enhancing Tumor + Necrotic Core)
# - Lớp 2: ED (Edema)
# - Lớp 3: ET (Enhancing Tumor) [Gồm cả lớp 4 cũ được remap về đây]

import os
import json
import glob
from tqdm import tqdm
import re
import shutil
import time
import numpy as np
import nibabel as nib
import sys

print("--- GIAI DOAN 2 (FIX 12 - REMAP LABEL): BAT DAU CHUYEN DOI DU LIEU ---")
start_time = time.time()

# 1. Đường dẫn NGUỒN
SOURCE_DATA_DIR_TRAIN = r"D:\Brats2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"

# 2. Lấy đường dẫn ĐÍCH từ biến môi trường
try:
    NNUNET_RAW = os.environ.get('nnUNet_raw')
    if NNUNET_RAW is None: raise EnvironmentError
except EnvironmentError:
    print("LOI: Bien moi truong 'nnUNet_raw' chua duoc thiet lap!")
    print("Ban da chay file 'activate_env.bat' trong terminal nay chua?")
    sys.exit(1)

print(f"Thu muc NGUON: {SOURCE_DATA_DIR_TRAIN}")
print(f"Thu muc DICH (nnUNet_raw): {NNUNET_RAW}")

# 3. Đặt tên Task (FIX: Dung 'Dataset' thay vi 'Task')
TASK_ID_NUM = 101
TASK_NAME_FULL = "Dataset101_BraTS2020" # <-- DA SUA THANH 'Dataset'
TASK_DIR = os.path.join(NNUNET_RAW, TASK_NAME_FULL)

# 4. Tạo các thư mục con
DIR_IMAGES_TR = os.path.join(TASK_DIR, "imagesTr")
DIR_LABELS_TR = os.path.join(TASK_DIR, "labelsTr")
os.makedirs(DIR_IMAGES_TR, exist_ok=True)
os.makedirs(DIR_LABELS_TR, exist_ok=True)
print(f"Da tao thu muc tac vu: {TASK_DIR}")

# 5. Định nghĩa nội dung cho file `dataset.json`
dataset_info = {
    "channel_names": { "0": "flair", "1": "t1", "2": "t1ce", "3": "t2" },
    "labels": {
        "background": 0,
        "NCR/NET": 1,
        "ED": 2,
        "ET": 3  # <-- FIX: Bay gio chung ta se tao ra nhan 3
    },
    "numTraining": 0,
    "file_ending": ".nii",
    "dataset_name": "BraTS2020",
    "description": "Brain Tumor Segmentation 2020"
}

json_path = os.path.join(TASK_DIR, "dataset.json")
with open(json_path, 'w') as f:
    json.dump(dataset_info, f, indent=4)
print("Da tao dataset.json (voi nhan da sua doi 0,1,2,3)")

# 6. Bắt đầu quá trình đổi tên, COPY (cho ảnh) và REMAP (cho nhãn)
print("\nBat dau sao chep (anh) va remap (nhan)...")
print("(Qua trinh nay se mat 10-30 phut, vui long kien nhan...)")

search_pattern = os.path.join(SOURCE_DATA_DIR_TRAIN, "*", "*_seg.nii")
label_files = glob.glob(search_pattern)
patient_id_re = re.compile(r'BraTS20_Training_(\d+)')
patient_count = 0

for label_file_path in tqdm(label_files, desc="Dang xu ly cac ca benh"):
    file_name = os.path.basename(label_file_path)
    patient_dir = os.path.dirname(label_file_path)
    patient_prefix = file_name.replace("_seg.nii", "")
    match = patient_id_re.search(patient_prefix)
    if not match: continue
    patient_id_num = match.group(1)
    patient_count += 1

    # Đường dẫn NGUỒN
    src_flair = os.path.join(patient_dir, f"{patient_prefix}_flair.nii")
    src_t1 = os.path.join(patient_dir, f"{patient_prefix}_t1.nii")
    src_t1ce = os.path.join(patient_dir, f"{patient_prefix}_t1ce.nii")
    src_t2 = os.path.join(patient_dir, f"{patient_prefix}_t2.nii")
    src_seg = label_file_path
    
    # Đường dẫn ĐÍCH
    target_prefix = f"BRATS_{patient_id_num}"
    dest_flair = os.path.join(DIR_IMAGES_TR, f"{target_prefix}_0000.nii")
    dest_t1 = os.path.join(DIR_IMAGES_TR, f"{target_prefix}_0001.nii")
    dest_t1ce = os.path.join(DIR_IMAGES_TR, f"{target_prefix}_0002.nii")
    dest_t2 = os.path.join(DIR_IMAGES_TR, f"{target_prefix}_0003.nii")
    dest_seg = os.path.join(DIR_LABELS_TR, f"{target_prefix}.nii")
    
    # --- Xu ly ANH (IMAGE): Chi can copy ---
    if os.path.exists(src_flair) and not os.path.exists(dest_flair): shutil.copy(src_flair, dest_flair)
    if os.path.exists(src_t1) and not os.path.exists(dest_t1): shutil.copy(src_t1, dest_t1)
    if os.path.exists(src_t1ce) and not os.path.exists(dest_t1ce): shutil.copy(src_t1ce, dest_t1ce)
    if os.path.exists(src_t2) and not os.path.exists(dest_t2): shutil.copy(src_t2, dest_t2)
    
    # --- Xu ly NHAN (LABEL): Load, Remap 4->3, va Save ---
    if os.path.exists(src_seg) and not os.path.exists(dest_seg):
        try:
            # 1. Load file .nii
            seg_img = nib.load(src_seg)
            # 2. Lay du lieu ra array numpy
            seg_data = seg_img.get_fdata()
            
            # 3. FIX QUAN TRONG: Doi tat ca nhan 4 thanh 3
            seg_data[seg_data == 4] = 3
            
            # 4. Dam bao kieu du lieu la integer (quan trong cho file nhan)
            seg_data = seg_data.astype(np.uint8)
            
            # 5. Tao file Nifti moi tu du lieu da sua
            new_seg_img = nib.Nifti1Image(seg_data, seg_img.affine, seg_img.header)
            # 6. Set kieu du lieu trong header
            new_seg_img.set_data_dtype(np.uint8)
            
            # 7. Luu file MOI vao DICH
            nib.save(new_seg_img, dest_seg)
            
        except Exception as e:
            print(f"\nLOI khi dang xu ly file {src_seg}: {e}")

# 7. Cập nhật `numTraining`
if patient_count > 0:
    dataset_info["numTraining"] = patient_count
    with open(json_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)
    print(f"\nDa cap nhat numTraining trong dataset.json thanh: {patient_count}")

end_time = time.time()
print(f"\nChuyen doi du lieu Giai doan 2 (REMAP) thanh cong ({patient_count} ca benh).")
print(f"Tong thoi gian thuc hien: {int(end_time - start_time)} giay.")
print(">>> GIAI DOAN 2 HOAN TAT: Du lieu da duoc chuyen doi VA SUA LOI.")