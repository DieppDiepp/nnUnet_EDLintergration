# Phân rã độ bất định bảo toàn độ chính xác cho phân đoạn khối u não với Evidential nnU-Net

*(Accuracy-Preserving Uncertainty Decomposition for Brain Tumor Segmentation with Evidential nnU-Net)*

Repository này chứa mã nguồn và tài liệu thực nghiệm cho phương pháp tích hợp **Evidential Deep Learning (EDL)** vào khung làm việc **nnU-Net v2** nhằm giải quyết bài toán phân đoạn khối u não đa phương thức trên tập dữ liệu **BraTS 2020**.

Mục tiêu chính của dự án là cung cấp khả năng **Ước lượng độ bất định (Uncertainty Quantification)** có độ tin cậy cao và phân rã chúng thành hai thành phần riêng biệt: **Aleatoric** (do nhiễu dữ liệu) và **Epistemic** (do thiếu hụt tri thức của mô hình). Đặc biệt, phương pháp đề xuất duy trì và vượt qua hiệu năng phân đoạn của mô hình Baseline SOTA mà không cần tăng chi phí tính toán (single forward pass).


## Trực quan hóa kết quả phân đoạn
Chú thích:

- <b>Hàng trên</b>: MRI Input, Ground Truth, AI Prediction. 

- <b>Hàng dưới</b>: Total Uncertainty, Aleatoric Uncertainty (nhiễu dữ liệu viền khối u) và Epistemic Uncertainty (vùng mô hình thiếu tự tin).

<br>

![BraTS axial 50](img/BraTs_012_axial-50.png)

<b>Hình 1.</b> Minh họa dự đoán, quá trình tạo bản đồ bất định và phân rã độ bất định 

<br>

![BraTS axial 25 uncertainty](img/BraTs_012_axial-25_uncertainty-hightlight.png)

<b>Hình 2.</b> Minh họa độ bất định, dễ thấy dù mô hình thất bại hoàn toàn trong dự đoán khối U nhưng độ bất định đã sáng lên và bắt rất tốt.



---

## 1. Giới thiệu

Phân đoạn khối u não tự động đã đạt được hiệu suất cực kỳ ấn tượng với các kiến trúc CNN 3D tiên tiến như nnU-Net. Tuy nhiên, trong y khoa, việc chỉ đưa ra dự đoán "cứng" (hard prediction) là chưa đủ. Các bác sĩ cần biết **mô hình tự tin đến mức nào** vào dự đoán đó để tránh các can thiệp sai lầm.

Các phương pháp ước lượng bất định truyền thống như Monte Carlo Dropout hay Deep Ensemble thường mắc phải hai nhược điểm chí mạng:

1. Chi phí tính toán khổng lồ (phải chạy suy luận nhiều lần hoặc train nhiều mô hình).
2. Thường đánh đổi độ chính xác phân đoạn (Accuracy) để lấy độ hiệu chuẩn (Calibration).

Dự án này đề xuất **Evidential nnU-Net**, một giải pháp thay thế lớp Softmax truyền thống bằng phân phối Dirichlet. Kỹ thuật này trực tiếp mô hình hóa mật độ xác suất của các lớp phân đoạn, cho phép tính toán độ bất định chỉ trong **một lần truyền xuôi (single forward pass)** với độ chính xác vượt trội. Đồng thời hiệu chuẩn ECE (Expected Calibration Error), điểm số QuBraTS vượt các phương pháp SOTA (State of the Art) hiện tại.

---

## 2. Phương pháp luận

### 2.1. Kiến trúc nền tảng (nnU-Net v2)

* **Framework:** Kế thừa sự tự động hóa cấu hình xuất sắc của nnU-Net v2.
* **Mạng Backbone:** 3D U-Net (Full resolution) xử lý đa phương thức (T1, T1ce, T2, FLAIR).
* **Chiến lược huấn luyện:** Chuẩn mực 5-fold Cross Validation để đảm bảo tính khách quan và công bằng cao nhất khi so sánh. Chúng tôi sử dụng seed chia tập huấn luyện và kiểm thử theo mặc định của nnUnet (`split_seed = 12345`)

### 2.2. Evidential Deep Learning (EDL)

Chúng tôi tùy biến quy trình huấn luyện của nnU-Net thông qua class `EDLLoss` và `EDLTrainer`:

$$
\mathcal{L}_{\text{Total}}
=
\underbrace{\mathcal{L}_{\text{Bayes}}}_{\text{Accuracy}}
+
\lambda_t \cdot
\underbrace{\mathcal{L}_{\text{KL}}}_{\text{Reliability}}
+
\underbrace{\mathcal{L}_{\text{Dice}}(\alpha / S)}_{\text{Segmentation}}
$$

* **Hàm kích hoạt bằng chứng:** Thay thế Softmax bằng Softplus để đảm bảo bằng chứng (evidence) thu thập được luôn là số dương.
* **Hàm mất mát (Custom Loss) đa thành phần:**
1. **Expected Cross-Entropy (Data Fit Loss):** Tối ưu hóa dự đoán sát với Ground Truth dựa trên kỳ vọng của phân phối Dirichlet.
2. **KL Divergence Penalty (với Annealing Step):** Phạt nghiêm khắc các dự đoán "tự tin thái quá" ở những vùng mô hình đoán sai, ép mô hình trở về phân phối Uniform khi gặp dữ liệu lạ.
3. **Dice Loss:** Khắc phục tình trạng mất cân bằng lớp cực đoan trong ảnh y tế, tối ưu hóa trực tiếp độ trùng lặp không gian (spatial overlap).



### 2.3. Cơ chế Phân rã độ bất định

Dựa trên lý thuyết độ tin cậy Dempster-Shafer, tổng độ bất định (Total Uncertainty) được tách bạch rõ ràng thành:


$$
\underbrace{\mathcal{I}[y,\theta]}_{\text{Epistemic}}
=
\underbrace{H[\mathbb{E}[p]]}_{\text{Total}}
-
\underbrace{\mathbb{E}[H[p]]}_{\text{Aleatoric}}
$$

### a. Total Uncertainty
Entropy của xác suất trung bình.

$$
\mathcal{U}_{Total}
=
-\sum_k \hat{p}_k \ln \hat{p}_k
$$

*(Sự hỗn loạn của quyết định cuối)*


### b. Aleatoric Uncertainty
Trung bình entropy của từng điểm (sử dụng hàm Digamma $\psi$).

$$
\mathcal{U}_{Aleatoric}
=
\sum_k \hat{p}_k
\left(
\psi(S+1) - \psi(\alpha_k+1)
\right)
$$

*(Độ nhiễu của dữ liệu)*


### c. Epistemic Uncertainty
Phần dư của phép trừ:

$$
\mathcal{U}_{Epistemic}
=
\mathcal{U}_{Total}
-
\mathcal{U}_{Aleatoric}
$$

- Đo lường sự **phân tán của đám mây xác suất** -> Có thể dùng để **phát hiện dữ liệu lạ (OOD)**

* **Aleatoric Uncertainty (Nhiễu dữ liệu):** Bắt nguồn từ chất lượng ảnh MRI kém, ranh giới mờ nhạt giữa khối u và mô lành. Sự bất định này không thể giảm bớt dù có train thêm dữ liệu.
* **Epistemic Uncertainty (Độ thiếu hiểu biết của mô hình):** Bắt nguồn từ việc mô hình chưa gặp dạng khối u này bao giờ (Out-of-Distribution). Có thể được khắc phục bằng cách thu thập thêm dữ liệu huấn luyện.



## 3. Phân tích Hiện tượng "Overconfidence" và Sự ưu việt của EDL

Trong các bài toán phân vùng y tế, các mô hình học sâu thường mắc phải hội chứng **Tự tin thái quá (Overconfidence)**. Khi cố gắng tối ưu hóa Dice Score, mô hình có xu hướng đẩy xác suất dự đoán lên tiệm cận 1.0 (100%) ngay cả ở những vùng biên mờ mịt. 

Lý do là các mô hình phân loại tối ưu độ chính xác (trong ngữ cảnh bài toán này là dice), cố xuất ra giá trị xác suất cao nhất ở lớp mà mô hình tin là đúng với mục tiêu đẩy hàm loss về 0. Điều này vô tình làm cho chỉ số ECE của các mô hình này rất cao (tệ), bởi vì các mẫu tự tin nhất đều đã bị đẩy vô bin cuối cùng, các mẫu kém tự tin hơn ở **các** bin còn lại có nhiều xác suất là dự đoán sai. Điều này làm tăng sai số hiệu chuẩn kỳ vọng (**Expected Calibration Error - ECE**).

EDL giải quyết triệt để vấn đề này bằng cách hấp thụ phần "xác suất ảo" vào một đại lượng bất định (Uncertainty), giúp mô hình giữ được sự khiêm tốn cần thiết.


![Baseline nnU-Net (50 Epochs)](img\reliability_diagram_10bins_nnUnet_base.png)

*> Hình 3a: Baseline nnU-Net (50 Epochs). Mặc dù mới ở giai đoạn đầu huấn luyện, mô hình đã tỏ ra quá tự tin, tạo ra các "Calibration Gap" (vùng gạch chéo đỏ) lớn ở các bin xác suất cao. ECE lên tới 1.536%.*


![Evidential nnU-Net (50 Epochs)](img\reliability_diagram_10bins_EDL50.png)

*> Hình 3b: Evidential nnU-Net (50 Epochs). Nhờ cơ chế phạt KL Divergence, EDL kiểm soát độ tự tin cực tốt, bám sát đường Perfect Calibration (đứt nét). ECE giảm mạnh chỉ còn 0.616%.*


![Evidential nnU-Net (250 Epochs)](img\reliability_diagram_10bins_EDL250.png)

*> Hình 3c: Evidential nnU-Net (250 Epochs). Khi huấn luyện dài hạn, mô hình đạt đỉnh cao về độ chính xác (Dice). Dù có xu hướng tự tin hơn, EDL vẫn gồng gánh xuất sắc để giữ ECE ở mức an toàn 0.805% (thấp hơn một nửa so với Baseline 50 epochs).*

---

## 4. Kết quả Thực nghiệm & Benchmark SOTA

Mô hình được huấn luyện và đánh giá nghiêm ngặt bằng phương pháp **5-fold Cross Validation** trên toàn bộ tập dữ liệu huấn luyện BraTS 2020 (đây là chuẩn đánh giá khắt khe nhất được MICCAI đề xuất, thay vì tự chia tỷ lệ train/test cố định).

### 4.1. Hiệu năng tổng thể (Dice & HD95)

*Cấu hình: TTA (Test Time Augmentation), trượt 50%. HD95 càng thấp càng tốt, Dice càng cao càng tốt.*

| Phương pháp | Epochs | Mean Dice (%) | WT | TC | ET | Mean HD95 (mm) | WT | TC | ET |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline nnU-Net| 50 | 87.17 | 92.52 | 89.92 | 79.07 | 11.59 | 3.75 | 3.69 | 27.34 |
| Evidential nnU-Net | 50 | 85.34 | 92.05 | 87.97 | 76.00 | 12.32 | 3.87 | 4.20 | 28.88 |
| **Evidential nnU-Net (Ours)** | **250** | **88.96** | **93.63** | **92.00** | **81.25** | **10.71** | **2.71** | **2.82** | **26.59** |

*> Nhận xét: Khi được huấn luyện đủ số epoch (250), Evidential nnU-Net không những không bị tụt hậu mà còn bứt phá mạnh mẽ, đạt Mean Dice xấp xỉ 89% và giảm đáng kể khoảng cách HD95.*

### 4.2. So sánh với các phương pháp SOTA (BraTS 2020)

Kết quả của phương pháp đề xuất được đặt lên bàn cân với các kiến trúc tiên tiến nhất hiện nay trên cùng bộ dữ liệu.

| Phương pháp | Whole Tumor (WT) | Tumor Core (TC) | Enhancing Tumor (ET) | Mean Dice |
| --- | --- | --- | --- | --- |
| dResU-Net | 86.60 | 83.57 | 80.04 | 83.40 |
| ADHDC-Net | 89.75 | 83.31 | 78.01 | 83.69 |
| nnU-Net Ensemble | 91.00 | 84.40 | 77.60 | 84.80 |
| H2NF-Net | 91.30 | 85.50 | 78.80 | 86.20 |
| Standard nnU-Net v1 | 91.07 | 87.97 | 81.37 | 87.07 |
| *Energy-Based Attention* | *89.13* | *88.97* | **87.47** | *88.52* |
| **Evidential nnU-Net (Ours)** | **93.63** | **92.00** | 81.25 | **88.96** |

*> Nhận xét: Phương pháp đề xuất thiết lập mức SOTA mới về Mean Dice (88.96%), đặc biệt vượt trội ở khả năng khoanh vùng Toàn bộ khối u (WT: 93.63%) và Lõi khối u (TC: 92.00%).*

### 4.3. Đánh giá chất lượng Độ bất định (Uncertainty Benchmark)

**Lưu ý:** Đa số các bài báo tham chiếu dưới đây sử dụng tệp dữ liệu tự chia train dev test của họ (ví dụ: EDL Regioned-base chia 236 train / 59 validation/ 74 test), trong khi kết quả của chúng tôi (Ours) là trung bình của **5-fold CV (đánh giá toàn diện trên toàn bộ dữ liệu)** để đảm bảo tính minh bạch.

| Phương pháp | Mean Dice ↑ | ECE (%) ↓ | BraS* ↑ | Ghi chú |
|---|---|---|---|---|
| Deep Ensemble | 0.807 | 1.000% | 0.873 | Tốn gấp $N$ lần chi phí tính toán |
| MC Dropout | 0.804 | 0.900% | 0.869 | Phải lấy mẫu nhiều lần (Multiple passes) |
| EDL (TBraTS, 200 epochs) | 0.790 | 1.900% | 0.862 | Single pass |
| EDL (Region-based, wDICE, 200 epochs) | 0.793 | 1.600% | 0.874 | Single pass |
| **Our EDL (250 epochs)** | **0.889** | **0.805%** | **0.930** | **Single pass, 5-Fold Cross Validation** |

*> Nhận xét: Phương pháp của chúng tôi không chỉ đạt mức Dice vượt trội hoàn toàn so với các biến thể EDL khác (0.889 vs 0.790), mà còn ép ECE xuống mức kỷ lục (0.805%), vượt qua cả phương pháp đắt đỏ như MC Dropout.*

---

**\*BraS (QU-BraTS Score).** Đây là thước đo chính thức được sử dụng trong **BraTS 2020 – Quantification of Uncertainty in Brain Tumor Segmentation (QU-BraTS) challenge** để đánh giá chất lượng của các bản đồ độ bất định (uncertainty maps).

Khác với các metric segmentation truyền thống chỉ đo độ chính xác (như Dice), QU-BraTS đánh giá liệu **độ bất định do mô hình dự đoán có thực sự tương quan với lỗi phân đoạn hay không**. Nói cách khác, một mô hình tốt không chỉ dự đoán chính xác mà còn phải **biết khi nào mình không chắc chắn**.

Metric này tổng hợp ba thành phần:

- **AUC Dice**: đo mức độ tương quan giữa độ tin cậy và chất lượng phân đoạn  
- **AUC FTP (False Tumor Positive)**: phạt nếu mô hình tự tin sai ở vùng nền  
- **AUC FTN (False Tumor Negative)**: phạt nếu mô hình tự tin bỏ sót khối u  

Điểm số cuối cùng nằm trong khoảng **[0,1]**, với **giá trị càng cao càng tốt**.

![QuBraTS Score ET](img\qu_brats_curve_ET.png)

*>Hình 4: Minh họa các đường cong AUC của class ET với mô hình EDL 250 epochs*

## 5. Trích dẫn (Citation)

Nếu bạn sử dụng mã nguồn hoặc lấy cảm hứng từ phương pháp này cho nghiên cứu của mình, vui lòng trích dẫn các tài liệu sau:

```bibtex
@article{nguyen2026uncertainty,
  title={Evidential nnU-Net: Highly Accurate Brain Tumor Segmentation with Reliable Uncertainty Decomposition},
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

@inproceedings{isensee2020nnu,
  title={nnU-Net for brain tumor segmentation},
  author={Isensee, Fabian and J{\"a}ger, Paul F and Full, Peter M and Vollmuth, Philipp and Maier-Hein, Klaus H},
  booktitle={International MICCAI Brainlesion Workshop},
  pages={118--132},
  year={2020},
  organization={Springer}
}
