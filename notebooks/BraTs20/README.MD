# BraTS 2020 Uncertainty Quantification with nnU-Net v2

**Please cite the following paper when using nnU-Net:**
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature methods*, 18(2), 203-211.

---
**Note:** English version is available below the Vietnamese section.
---

## [Tiếng Việt] Tổng quan

Thư mục này chứa các Jupyter Notebooks thực hiện quy trình huấn luyện và đánh giá mô hình phân đoạn khối u não (Brain Tumor Segmentation) trên dữ liệu BraTS 2020. Dự án sử dụng khung làm việc nnU-Net v2 làm nền tảng (Baseline) và mở rộng với phương pháp Evidential Deep Learning (EDL) để định lượng độ bất định (Uncertainty Quantification).

Môi trường thực thi được tối ưu hóa cho **Google Colab**, sử dụng **Google Drive** làm nơi lưu trữ dữ liệu vĩnh viễn.

### Cấu trúc và Quy trình xử lý

Các notebook được đánh số theo thứ tự thực hiện để đảm bảo tính nhất quán của dữ liệu.

#### 1. Chuẩn bị dữ liệu
**File:** `2_Convert_to_nnUNet_Raw.ipynb`

* **Mục đích:** Chuyển đổi dữ liệu gốc BraTS 2020 sang định dạng chuẩn của nnU-Net v2.
* **Chi tiết kỹ thuật:**
    * Định danh Dataset ID là `Dataset101_BraTS2020` (tránh trùng lặp với các ID mặc định 001-010 của MSD).
    * Đổi tên file theo quy tắc `{Case_ID}_{ChannelID}.nii.gz`.
    * Tự động tạo file `dataset.json` với cấu trúc metadata v2, bao gồm ánh xạ kênh (modality) và nhãn (labels).

#### 2. Lập kế hoạch và Tiền xử lý
**File:** `3_Plan_and_Preprocess.ipynb`

* **Mục đích:** Phân tích đặc trưng dữ liệu (Fingerprint) và tạo các phương án huấn luyện (Plans).
* **Chi tiết kỹ thuật:**
    * Thực hiện trích xuất đặc trưng dataset mà nnU-Net cung cấp (spacing, intensity distribution, shape) để tạo file `dataset_fingerprint.json` và `nnUNetPlans.json`.
    * Chạy training thử nghiệm 1 epoch để hệ thống tự sinh ra file `splits_final.json`. File này được lưu lại để đảm bảo tính nhất quán trong việc chia tập train/val cho các lần huấn luyện sau (mặc dù nnU-Net cố định random_seed=12345).
    * **Thử nghiệm đa cấu hình:** Tạo configuration cho nhiều kiến trúc khác nhau (2d, 3d_fullres) và thử nghiệm các plan khác biệt so với mặc định (như nnU-Net ResEnc L, nnU-Net ResEnc XL). Chi tiết về các preset này có thể tham khảo tại tài liệu gốc của nnU-Net.
    * **Cấu hình hiện tại:** Sử dụng `3d_fullres` và `old nnU-Net default plans` (chưa áp dụng các kiến trúc ResEnc mới).

#### 3. Huấn luyện Baseline (Chuẩn)
**File:** `4_Train_nnU-Net_Baseline_(BraTS2020).ipynb`

* **Mục đích:** Huấn luyện mô hình nnU-Net tiêu chuẩn để làm mốc so sánh (Benchmark).
* **Lệnh thực thi:**
    * Huấn luyện 5-fold Cross Validation:
      `!nnUNetv2_train Dataset101_BraTS2020 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainer_50epochs`
    * Tìm kiếm cấu hình/ensemble tối ưu nhất:
      `!nnUNetv2_find_best_configuration Dataset101_BraTS2020 -f 0 1 2 3 4 -tr nnUNetTrainer_50epochs -c 3d_fullres`

* **Chi tiết kỹ thuật:**
    * Sử dụng lớp custom trainer `nnUNetTrainer_50epochs` (phiên bản rút gọn số epochs so với mặc định 1000 epochs) để thực hiện kiểm chứng chéo.
    * **Hậu xử lý Metrics:** Tự động tính toán Dice Score trên các vùng lâm sàng gộp: Whole Tumor (WT), Tumor Core (TC), và Enhancing Tumor (ET) thay vì các nhãn rời rạc.
    * Kết quả được lưu trữ tại: `/data/nnUNet_results/Dataset101_BraTS2020/nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres`.
    * Mô hình dùng để inference là một ensemble được tạo ra từ lệnh `find_best_configuration`, tuân thủ phương pháp của bài báo gốc.

**Kết quả thực nghiệm (Baseline - 50 Epochs):**

| Metric | Dice ET | Dice TC | Dice WT |
| :--- | :--- | :--- | :--- |
| **Mean** | **0.7648** | **0.8647** | **0.9114** |
| Std | 0.2690 | 0.1682 | 0.0899 |
| Median | 0.8709 | 0.9256 | 0.9338 |

**So sánh với State-of-the-Art (BraTS 2020):**
*Lưu ý: Kết quả hiện tại dựa trên version chưa tối ưu hóa hoàn toàn (50 epochs) so với chuẩn 1000 epochs.*

| Method | Whole (WT) | Core (TC) | Enh. (ET) | Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Our Baseline (50 ep)** | **91.14** | **86.47** | **76.48** | **84.70** |
| nnU-Net (Isensee et al.) | 91.17 | 87.97 | 81.37 | 87.07 |
| H2NF-Net (Jia et al.) | 91.30 | 85.50 | 78.80 | 86.20 |
| nnUnetEnsemble (Fidon) | 91.00 | 84.40 | 77.60 | 84.80 |
| dResU-Net (Raza et al.) | 86.60 | 83.57 | 80.04 | 83.40 |

**Kết quả HD95 (mm) - Thấp hơn là tốt hơn:**
* Mean HD95 WT: 5.0697
* Mean HD95 TC: 7.2368
* Mean HD95 ET: 32.0350

#### 4. Huấn luyện EDL (Độ bất định)
**File:** `5_Train_nnU-Net_EDL-Trainer_(BraTS2020).ipynb`

* **Mục đích:** Huấn luyện mô hình tích hợp Evidential Deep Learning để định lượng độ bất định.
* **Lệnh thực thi:**
    * Huấn luyện:
      `!nnUNet_EPOCHS=50 nnUNetv2_train Dataset101_BraTS2020 3d_fullres -f 0 1 2 3 4 -tr EDLTrainer --npz`
    * Tìm kiếm cấu hình tốt nhất:
      `!nnUNetv2_find_best_configuration Dataset101_BraTS2020 -f 0 1 2 3 4 -tr EDLTrainer -c 3d_fullres`

* **Chi tiết kỹ thuật:**
    * **Code Injection:** Sử dụng kỹ thuật sao chép động file `EDLTrainer.py` (từ thư mục `src`) vào thư viện hệ thống nnU-Net tại runtime. Trainer này cài đặt hàm mất mát Evidential Loss và điều chỉnh kiến trúc mạng đầu ra.
    * Cấu hình tham số Epoch thông qua biến môi trường để tái hiện điều kiện huấn luyện giống hệt Baseline (Plan, Fingerprint, 5 folds, 50 epochs).
    * Kết quả được lưu tại thư mục riêng: `/data/nnUNet_results/Dataset101_BraTS2020/EDLTrainer__nnUNetPlans__3d_fullres/`.

**Kết quả thực nghiệm (EDL Engine):**

| Metric | Dice ET | Dice TC | Dice WT |
| :--- | :--- | :--- | :--- |
| **Mean** | **0.7557** | **0.8546** | **0.9145** |
| Std | 0.2708 | 0.1782 | 0.0709 |
| Median | 0.8596 | 0.9210 | 0.9345 |

**Kết quả HD95 (mm) - Thấp hơn là tốt hơn:**
*Lưu ý: HD95 được tính trên CPU. Với các trường hợp dự đoán sai ET (có dự đoán nhưng thực tế không có), áp dụng mức phạt 373.13 mm theo bài báo gốc.*

| Region | Baseline HD95 | EDL HD95 |
| :--- | :--- | :--- |
| **Whole Tumor (WT)** | 5.0697 | 5.1461 |
| **Tumor Core (TC)** | 7.2368 | 6.2760 |
| **Enhancing Tumor (ET)**| 32.0350 | 29.3577 |

---

## [English] Overview

This directory contains Jupyter Notebooks for the training and evaluation pipeline of Brain Tumor Segmentation on the BraTS 2020 dataset. The project utilizes nnU-Net v2 as the Baseline framework and extends it with Evidential Deep Learning (EDL) for Uncertainty Quantification.

The execution environment is optimized for **Google Colab**, leveraging **Google Drive** for persistent data storage.

### Structure and Workflow

Notebooks are numbered sequentially to ensure data consistency throughout the pipeline.

#### 1. Data Preparation
**File:** `2_Convert_to_nnUNet_Raw.ipynb`

* **Goal:** Convert raw BraTS 2020 data into the strict nnU-Net v2 format.
* **Technical Details:**
    * Assigns Dataset ID as `Dataset101_BraTS2020` (to avoid conflicts with default MSD IDs 001-010).
    * Renames files strictly following the `{Case_ID}_{ChannelID}.nii.gz` convention.
    * Automatically generates `dataset.json` with v2 metadata structure, including channel mapping and label definitions.

#### 2. Planning and Preprocessing
**File:** `3_Plan_and_Preprocess.ipynb`

* **Goal:** Analyze dataset fingerprints and generate training plans.
* **Technical Details:**
    * Extracts dataset characteristics provided by nnU-Net (spacing, intensity distribution, shape) to generate `dataset_fingerprint.json` and `nnUNetPlans.json`.
    * Performs a 1-epoch trial run to generate `splits_final.json`. This file is saved to ensure consistent train/val splits across future training sessions (although nnU-Net defaults to random_seed=12345).
    * **Multi-Configuration Testing:** Generates configurations for various architectures (2d, 3d_fullres) and experiments with alternative plans (e.g., nnU-Net ResEnc L, nnU-Net ResEnc XL). Refer to the original documentation for preset details.
    * **Current Configuration:** Using `3d_fullres` with `old nnU-Net default plans`.

#### 3. Baseline Training
**File:** `4_Train_nnU-Net_Baseline_(BraTS2020).ipynb`

* **Goal:** Train a standard nnU-Net model to serve as a performance benchmark.
* **Commands:**
    * 5-fold Cross Validation:
      `!nnUNetv2_train Dataset101_BraTS2020 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainer_50epochs`
    * Find best model/ensemble:
      `!nnUNetv2_find_best_configuration Dataset101_BraTS2020 -f 0 1 2 3 4 -tr nnUNetTrainer_50epochs -c 3d_fullres`

* **Technical Details:**
    * Uses a custom trainer class `nnUNetTrainer_50epochs` (a variant of the standard trainer with reduced epochs) to perform 5-fold Cross Validation.
    * **Post-processing Metrics:** Automatically calculates Dice Scores on clinically relevant regions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) instead of raw discrete labels.
    * Results stored at: `/data/nnUNet_results/Dataset101_BraTS2020/nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres`.
    * The inference model is an ensemble generated from `find_best_configuration`, following the original paper's methodology.

**Experimental Results (Baseline - 50 Epochs):**

| Metric | Dice ET | Dice TC | Dice WT |
| :--- | :--- | :--- | :--- |
| **Mean** | **0.7648** | **0.8647** | **0.9114** |

**Comparison with State-of-the-Art (BraTS 2020):**
*Note: Current results are based on a non-optimized version (50 epochs) compared to the standard 1000 epochs.*

| Method | Whole (WT) | Core (TC) | Enh. (ET) | Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Our Baseline (50 ep)** | **91.14** | **86.47** | **76.48** | **84.70** |
| nnU-Net (Isensee et al.) | 91.17 | 87.97 | 81.37 | 87.07 |
| H2NF-Net (Jia et al.) | 91.30 | 85.50 | 78.80 | 86.20 |

**HD95 Results (mm) - Lower is better:**
* Mean HD95 WT: 5.0697
* Mean HD95 TC: 7.2368
* Mean HD95 ET: 32.0350

#### 4. EDL Training (Uncertainty)
**File:** `5_Train_nnU-Net_EDL-Trainer_(BraTS2020).ipynb`

* **Goal:** Train the model integrated with Evidential Deep Learning.
* **Commands:**
    * Train: `!nnUNet_EPOCHS=50 nnUNetv2_train Dataset101_BraTS2020 3d_fullres -f 0 1 2 3 4 -tr EDLTrainer --npz`
    * Find best config: `!nnUNetv2_find_best_configuration Dataset101_BraTS2020 -f 0 1 2 3 4 -tr EDLTrainer -c 3d_fullres`

* **Technical Details:**
    * **Code Injection:** Implements dynamic injection of the custom `EDLTrainer.py` (from the `src` directory) into the system nnU-Net library at runtime. This trainer incorporates a new loss function and slightly modifies the output architecture.
    * Epoch parameters are configured via environment variables to replicate the exact training conditions of the Baseline (Plan, Fingerprint, 5 folds, etc.).
    * **Post-processing Metrics:** Calculates Dice Scores on merged clinical regions (WT, TC, ET).
    * Results are stored in a separate directory (`/data/nnUNet_results/Dataset101_BraTS2020/EDLTrainer__nnUNetPlans__3d_fullres/`) and backed up independently from the Baseline.

**Experimental Results (EDL Engine):**

| Metric | Dice ET | Dice TC | Dice WT |
| :--- | :--- | :--- | :--- |
| **Mean** | **0.7557** | **0.8546** | **0.9145** |

**HD95 Comparison (mm):**
*Note: HD95 calculated on CPU. A penalty of 373.13 mm is applied for False Positive ET predictions (predicted but not present).*

| Region | Baseline HD95 | EDL HD95 |
| :--- | :--- | :--- |
| **Whole Tumor (WT)** | 5.0697 | 5.1461 |
| **Tumor Core (TC)** | 7.2368 | 6.2760 |
| **Enhancing Tumor (ET)**| 32.0350 | 29.3577 |