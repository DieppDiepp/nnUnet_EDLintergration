# Phân rã độ bất định bảo toàn độ chính xác cho phân đoạn khối u não với Evidential nnU-Net

Repository này chứa mã nguồn và tài liệu thực nghiệm cho phương pháp tích hợp **Evidential Deep Learning (EDL)** vào khung làm việc **nnU-Net v2** nhằm giải quyết bài toán phân đoạn khối u não đa phương thức trên tập dữ liệu **BraTS 2020**.

Mục tiêu chính của dự án là cung cấp khả năng ước lượng độ bất định (Uncertainty Quantification) tin cậy và phân rã chúng thành hai thành phần: **Aleatoric** (do nhiễu dữ liệu) và **Epistemic** (do sự thiếu hụt tri thức của mô hình), trong khi vẫn duy trì hoặc cải thiện hiệu năng phân đoạn so với mô hình Baseline SOTA.

## 1. Giới thiệu

Phân đoạn khối u não tự động đạt được hiệu suất cao với các mô hình CNN 3D như nnU-Net. Tuy nhiên, việc thiếu thông tin về độ tin cậy của dự đoán hạn chế khả năng ứng dụng lâm sàng. Các phương pháp ước lượng bất định phổ biến như Monte Carlo Dropout hay Deep Ensemble thường yêu cầu chi phí tính toán lớn hoặc làm giảm độ chính xác phân đoạn.

Dự án này đề xuất **Evidential nnU-Net**, thay thế lớp Softmax truyền thống bằng phân phối Dirichlet để mô hình hóa trực tiếp mật độ xác suất của các lớp phân đoạn. Phương pháp này cho phép ước lượng độ bất định trong một lần truyền xuôi (single forward pass) mà không tốn chi phí lấy mẫu.

## 2. Phương pháp luận

### 2.1. Kiến trúc nền tảng

* **Framework:** nnU-Net v2 (Self-configuring method).
* **Mạng Backbone:** 3D U-Net (Full resolution).
* **Cấu hình:** Tự động thích nghi dựa trên đặc trưng dataset (spacing, median shape, intensity properties) của BraTS 2020.

### 2.2. Evidential Deep Learning (EDL)

Chúng tôi tích hợp EDL vào quy trình huấn luyện của nnU-Net thông qua `EDLTrainer` tùy chỉnh:

* **Hàm kích hoạt:** Sử dụng hàm Softplus để đảm bảo bằng chứng (evidence) luôn dương, thay vì Softmax.
* **Hàm mất mát (Custom Loss):** Tổng hợp của ba thành phần:
1. **Bayes Risk:** Thay thế Cross-Entropy, tối ưu hóa kỳ vọng của hàm log-likelihood trên phân phối Dirichlet.
2. **KL Divergence (với Annealing):** Đóng vai trò regularization, phạt mô hình nếu nó quá tự tin vào các dự đoán sai lệch so với phân phối Uniform.
3. **Dice Loss:** Đảm bảo hiệu năng phân đoạn trên các vùng biên và giải quyết mất cân bằng lớp (foreground/background).



### 2.3. Phân rã độ bất định

Dựa trên lý thuyết Dempster-Shafer, độ bất định được phân rã thành:

* **Aleatoric Uncertainty:** Độ bất định dữ liệu, phản ánh nhiễu, chuyển động hoặc ranh giới mờ giữa các mô.
* **Epistemic Uncertainty:** Độ bất định mô hình, phản ánh những vùng dữ liệu nằm ngoài phân phối huấn luyện (OOD).

## 3. Dữ liệu thực nghiệm

Dự án sử dụng tập dữ liệu **BraTS 2020** (Multimodal Brain Tumor Segmentation Challenge 2020).

* **Modalities:** T1, T1ce, T2, FLAIR.
* **Nhãn gốc:** Background (0), NCR/NET (1), ED (2), ET (4).
* **Vùng đánh giá lâm sàng:**
* **Whole Tumor (WT):** Hợp của tất cả các nhãn (1 + 2 + 4).
* **Tumor Core (TC):** Hợp của NCR/NET và ET (1 + 4).
* **Enhancing Tumor (ET):** Chỉ nhãn ET (4).



## 4. Kết quả thực nghiệm

Mô hình được huấn luyện và đánh giá bằng phương pháp 5-fold Cross Validation. Kết quả dưới đây so sánh giữa Baseline (nnU-Net chuẩn) và phương pháp đề xuất (EDL).

**Lưu ý:** Kết quả thực nghiệm sơ bộ trên 50 epochs (để kiểm chứng tính khả thi).

### 4.1. Hiệu năng phân đoạn (Dice Score)

| Phương pháp | WT Mean | TC Mean | ET Mean | Trung bình |
| --- | --- | --- | --- | --- |
| **Baseline (Standard nnU-Net)** | 0.9114 | **0.8647** | **0.7648** | 0.8470 |
| **Evidential nnU-Net (Proposed)** | **0.9145** | 0.8546 | 0.7557 | 0.8416 |

*Nhận xét:* Phương pháp EDL duy trì hiệu năng phân đoạn tương đương với Baseline, thậm chí cải thiện nhẹ ở vùng Whole Tumor (WT), chứng minh rằng việc thêm khả năng ước lượng bất định không làm giảm độ chính xác phân đoạn ("Accuracy-Preserving").

### 4.2. Độ đo khoảng cách (HD95 - mm)

*Thấp hơn là tốt hơn.*

| Phương pháp | WT | TC | ET |
| --- | --- | --- | --- |
| **Baseline** | **5.0697** | 7.2368 | 32.0350 |
| **Evidential nnU-Net** | 5.1461 | **6.2760** | **29.3577** |

*Nhận xét:* EDL cho thấy sự cải thiện đáng kể về chỉ số HD95 ở vùng TC và ET, cho thấy biên dạng dự đoán ít xuất hiện các vùng nhiễu xa (outliers) hơn so với Baseline.

## 5. Trích dẫn

Nếu sử dụng mã nguồn hoặc tham khảo phương pháp này, vui lòng trích dẫn báo cáo đồ án gốc và nnU-Net:

```bibtex
@article{nguyen2024accuracy,
  title={Accuracy-Preserving Uncertainty Decomposition for Brain Tumor Segmentation with Evidential nnU-Net},
  author={Nguyen, Luong Dac and Son, Bui Truong Thai and Long, Ung Hoang and Nguyen-Tat, Thien B.},
  journal={Preprint submitted to Elsevier},
  year={2024}
}

@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}

```