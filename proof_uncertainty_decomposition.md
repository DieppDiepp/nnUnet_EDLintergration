# Chứng Minh Phân Rã Độ Bất Định Trong EDL

Tài liệu này chứng minh công thức phân rã độ bất định mà repo đang dùng trong mô hình **Evidential Deep Learning (EDL)** cho bài toán phân đoạn đa lớp.

Trong repo, mạng không xuất trực tiếp softmax probability. Thay vào đó, mạng sinh ra **evidence** dương, rồi biến thành tham số Dirichlet:

$$
e_k = \operatorname{softplus}(z_k), \qquad \alpha_k = e_k + 1, \qquad S = \sum_{k=1}^K \alpha_k
$$

Sau đó ta xét phân phối:

$$
\mathbf{p} \sim \operatorname{Dir}(\boldsymbol{\alpha}),
 \qquad y \mid \mathbf{p} \sim \operatorname{Cat}(\mathbf{p})
$$

Mục tiêu là chứng minh:

$$
I(y,\mathbf{p}\mid \boldsymbol{\alpha})
= H\!\left[\mathbb{E}[\mathbf{p}]\right] - \mathbb{E}\!\left[H(\mathbf{p})\right]
$$

và nhận diện ba thành phần:

$$
\mathcal{U}_{\text{Total}} = H\!\left[\mathbb{E}[\mathbf{p}]\right],
 \qquad
\mathcal{U}_{\text{Aleatoric}} = \mathbb{E}\!\left[H(\mathbf{p})\right],
 \qquad
\mathcal{U}_{\text{Epistemic}} = \mathcal{U}_{\text{Total}} - \mathcal{U}_{\text{Aleatoric}}
$$

---

## 1. Ký hiệu và giả thiết

Xét bài toán phân loại/phan đoạn với $K$ lớp.

- $\mathbf{p} = (p_1,\dots,p_K)$ là vector xác suất ngẫu nhiên.
- $\boldsymbol{\alpha} = (\alpha_1,\dots,\alpha_K)$ là tham số Dirichlet, với $\alpha_k > 0$.
- $S = \sum_{k=1}^K \alpha_k$.
- $H(\mathbf{q}) = -\sum_{k=1}^K q_k \log q_k$ là entropy Shannon của vector xác suất $\mathbf{q}$.

Trong EDL, ta không dùng một vector xác suất cố định, mà dùng một **phân phối trên xác suất**:

$$
\mathbf{p} \sim \operatorname{Dir}(\boldsymbol{\alpha})
$$

Vì vậy, độ bất định có hai tầng:

1. **Bất định do nhiễu dữ liệu**: ngay cả khi biết $\mathbf{p}$, nhãn vẫn còn ngẫu nhiên.
2. **Bất định do mô hình chưa biết đủ**: chính $\mathbf{p}$ còn là ngẫu nhiên vì ta chưa chắc hoàn toàn về tham số.

Đây chính là nền tảng để phân rã aleatoric và epistemic.

---

## 2. Kỳ vọng của xác suất dự đoán

### Mệnh đề 1

Với $\mathbf{p} \sim \operatorname{Dir}(\boldsymbol{\alpha})$, ta có:

$$
\mathbb{E}[p_k] = \frac{\alpha_k}{S}
$$

### Chứng minh

Đây là tính chất cơ bản của phân phối Dirichlet. Có thể suy ra từ mật độ:

$$
f(\mathbf{p}\mid \boldsymbol{\alpha})
= \frac{1}{B(\boldsymbol{\alpha})}
\prod_{k=1}^K p_k^{\alpha_k-1},
\qquad
B(\boldsymbol{\alpha}) = \frac{\prod_{k=1}^K \Gamma(\alpha_k)}{\Gamma(S)}
$$

Dirichlet có đối xứng theo từng thành phần và tổng của các thành phần bằng 1, nên kỳ vọng của từng thành phần phải tỷ lệ với $\alpha_k$. Công thức chuẩn của Dirichlet cho ta:

$$
\mathbb{E}[p_k] = \frac{\alpha_k}{S}
$$

Đặt:

$$
\hat p_k := \mathbb{E}[p_k] = \frac{\alpha_k}{S}
$$

thì vector $\hat{\mathbf{p}}$ là xác suất dự đoán trung bình mà repo dùng khi tính uncertainty.

---

## 3. Total uncertainty là entropy của kỳ vọng

### Định nghĩa

Ta định nghĩa:

$$
\mathcal{U}_{\text{Total}}
= H\!\left[\mathbb{E}[\mathbf{p}]\right]
= -\sum_{k=1}^K \hat p_k \log \hat p_k
$$

Thay $\hat p_k = \alpha_k/S$ vào, được:

$$
\mathcal{U}_{\text{Total}}
= -\sum_{k=1}^K \frac{\alpha_k}{S}
\log\!\left(\frac{\alpha_k}{S}\right)
$$

Đây đúng là **predictive entropy**: entropy của phân phối trung bình mà mô hình dự đoán.

---

## 4. Chứng minh công thức entropy kỳ vọng

Ta cần chứng minh:

$$
\mathbb{E}[H(\mathbf{p})]
= \sum_{k=1}^K \frac{\alpha_k}{S}
\left[\psi(S+1)-\psi(\alpha_k+1)\right]
$$

trong đó $\psi(\cdot)$ là hàm digamma:

$$
\psi(x) = \frac{d}{dx}\log \Gamma(x)
$$

### 4.1. Bổ đề moment một chiều của Dirichlet

Với $p_k$ là một thành phần của $\mathbf{p} \sim \operatorname{Dir}(\boldsymbol{\alpha})$, ta có moment:

$$
\mathbb{E}[p_k^r]
= \frac{\Gamma(S)\,\Gamma(\alpha_k+r)}
{\Gamma(\alpha_k)\,\Gamma(S+r)}
\qquad (r > -\alpha_k)
$$

#### Chứng minh

Ta có thể xem $p_k$ có phân phối biên Beta:

$$
p_k \sim \operatorname{Beta}(\alpha_k, S-\alpha_k)
$$

nên:

$$
\mathbb{E}[p_k^r]
= \frac{1}{B(\alpha_k, S-\alpha_k)}
\int_0^1 t^{\alpha_k+r-1}(1-t)^{S-\alpha_k-1}\,dt
$$

Dùng định nghĩa Beta function:

$$
B(x,y) = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}
$$

suy ra:

$$
\mathbb{E}[p_k^r]
= \frac{\Gamma(S)\Gamma(\alpha_k+r)}
{\Gamma(\alpha_k)\Gamma(S+r)}
$$

---

### 4.2. Từ moment sang $p_k \log p_k$

Ta dùng đẳng thức:

$$
\frac{d}{dr}p_k^r = p_k^r \log p_k
$$

nên:

$$
\mathbb{E}[p_k \log p_k]
= \left.\frac{d}{dr}\mathbb{E}[p_k^r]\right|_{r=1}
$$

Xét:

$$
M_k(r) := \mathbb{E}[p_k^r]
= \frac{\Gamma(S)\Gamma(\alpha_k+r)}
{\Gamma(\alpha_k)\Gamma(S+r)}
$$

lấy log:

$$
\log M_k(r)
= \log \Gamma(S) - \log \Gamma(\alpha_k)
+ \log \Gamma(\alpha_k+r) - \log \Gamma(S+r)
$$

Đạo hàm theo $r$:

$$
\frac{M_k'(r)}{M_k(r)}
= \psi(\alpha_k+r) - \psi(S+r)
$$

nên:

$$
M_k'(r)
= M_k(r)\big[\psi(\alpha_k+r)-\psi(S+r)\big]
$$

Thay $r=1$:

$$
\mathbb{E}[p_k \log p_k]
= \frac{\alpha_k}{S}\big[\psi(\alpha_k+1)-\psi(S+1)\big]
$$

Vì $H(\mathbf{p}) = -\sum_k p_k \log p_k$, suy ra:

$$
\mathbb{E}[H(\mathbf{p})]
= -\sum_{k=1}^K \mathbb{E}[p_k \log p_k]
$$

do đó:

$$
\mathbb{E}[H(\mathbf{p})]
= \sum_{k=1}^K \frac{\alpha_k}{S}
\big[\psi(S+1)-\psi(\alpha_k+1)\big]
$$

Đây chính là công thức aleatoric uncertainty mà repo đang dùng.

---

## 5. Phân rã epistemic bằng mutual information

Ta xét mutual information giữa nhãn $y$ và vector xác suất ngẫu nhiên $\mathbf{p}$, có điều kiện theo $\boldsymbol{\alpha}$:

$$
I(y;\mathbf{p}\mid \boldsymbol{\alpha})
= H(y\mid \boldsymbol{\alpha}) - \mathbb{E}_{\mathbf{p}\sim \operatorname{Dir}(\boldsymbol{\alpha})}
\!\left[H(y\mid \mathbf{p}, \boldsymbol{\alpha})\right]
$$

Theo mô hình:

$$
y \mid \mathbf{p} \sim \operatorname{Cat}(\mathbf{p})
$$

nên:

$$
H(y\mid \mathbf{p}, \boldsymbol{\alpha}) = H(\mathbf{p})
$$

Vì $P(y=k\mid \boldsymbol{\alpha})=\mathbb{E}[p_k]$, nên:

$$
H(y\mid \boldsymbol{\alpha})
= H\!\left(\mathbb{E}[\mathbf{p}]\right)
$$

Suy ra:

$$
I(y;\mathbf{p}\mid \boldsymbol{\alpha})
= H\!\left(\mathbb{E}[\mathbf{p}]\right)
- \mathbb{E}[H(\mathbf{p})]
$$

Đặt:

$$ 
\mathcal{U}_{\text{Epistemic}}
:= I(y;\mathbf{p}\mid \boldsymbol{\alpha})
$$

thì:

$$
\mathcal{U}_{\text{Epistemic}}
= \mathcal{U}_{\text{Total}} - \mathcal{U}_{\text{Aleatoric}}
$$

Đây là công thức phân rã mà repo hiển thị trong inference và trong README.

---

## 6. Vì sao epistemic không âm

Vì epistemic uncertainty được định nghĩa là mutual information, nên:

$$
\mathcal{U}_{\text{Epistemic}}
= I(y;\mathbf{p}\mid \boldsymbol{\alpha}) \ge 0
$$

Đây là hệ quả trực tiếp của tính chất cơ bản của mutual information:

$$
I(A;B) \ge 0
$$

Vì vậy:

- $\mathcal{U}_{\text{Total}}$ là entropy dự đoán trung bình.
- $\mathcal{U}_{\text{Aleatoric}}$ là phần bất định không thể loại bỏ hoàn toàn do nhiễu dữ liệu.
- $\mathcal{U}_{\text{Epistemic}}$ là phần bất định do mô hình còn chưa chắc chắn.

Do đó phân rã này không chỉ có ý nghĩa trực giác, mà còn có nền tảng thông tin học chặt chẽ.

---

## 7. Đối chiếu trực tiếp với implementation trong repo

Phần code của repo khớp đúng với công thức trên:

### 7.1. Ở trainer

Trong [`nnUNet/nnunetv2/training/nnUNetTrainer/EDLTrainer.py`](nnUNet/nnunetv2/training/nnUNetTrainer/EDLTrainer.py):

- `evidence = F.softplus(outputs)` tạo evidence dương.
- `alpha = evidence + 1` tạo tham số Dirichlet.
- `S = torch.sum(alpha, dim=1, keepdim=True)` tính tổng $S$.
- `p = alpha / S` chính là $\hat{\mathbf{p}} = \mathbb{E}[\mathbf{p}]$.
- `torch.digamma(S)` và `torch.digamma(alpha)` được dùng để tính expected cross-entropy và KL regularization.

Khớp với các dòng:

$$
\texttt{evidence = softplus(outputs)},\quad
\texttt{alpha = evidence + 1},\quad
\texttt{p = alpha / S}
$$

### 7.2. Ở inference engine

Trong [`src/edl_engine.py`](src/edl_engine.py):

- `evidence = F.softplus(pred_logits)`
- `alpha = evidence + 1`
- `S = torch.sum(alpha, dim=0, keepdim=True)`
- `probs = alpha / S`
- `total_unc = -torch.sum(probs * torch.log(probs + 1e-7), dim=0)`
- `digamma_S = torch.digamma(S + 1)`
- `digamma_alpha = torch.digamma(alpha + 1)`
- `aleatoric_unc = torch.sum(probs * (digamma_S - digamma_alpha), dim=0)`
- `epistemic_unc = total_unc - aleatoric_unc`

Đây chính là bản triển khai số học của:

$$
\mathcal{U}_{\text{Total}} = H\!\left[\mathbb{E}[\mathbf{p}]\right],
\qquad
\mathcal{U}_{\text{Aleatoric}} = \mathbb{E}[H(\mathbf{p})],
\qquad
\mathcal{U}_{\text{Epistemic}} = \mathcal{U}_{\text{Total}} - \mathcal{U}_{\text{Aleatoric}}
$$

### 7.3. Một lưu ý về KL loss

Term KL trong trainer:

$$
\mathcal{L}_{\text{KL}} = \operatorname{KL}\big(\operatorname{Dir}(\boldsymbol{\alpha}) \,\|\, \operatorname{Dir}(\mathbf{1})\big)
$$

không làm thay đổi chứng minh phân rã ở trên. Nó chỉ là một regularizer để ép mô hình giảm evidence ở các vùng không chắc chắn hoặc sai nhãn, giúp uncertainty map có ý nghĩa hơn trong thực nghiệm.

---

## 8. Kết luận

Ta đã chứng minh rằng, dưới giả thiết EDL với:

$$
\mathbf{p} \sim \operatorname{Dir}(\boldsymbol{\alpha}), \qquad
y \mid \mathbf{p} \sim \operatorname{Cat}(\mathbf{p})
$$

thì:

$$
\boxed{
\mathcal{U}_{\text{Total}}
= \mathcal{U}_{\text{Aleatoric}} + \mathcal{U}_{\text{Epistemic}}
}
$$

với:

$$
\mathcal{U}_{\text{Total}} = H\!\left[\mathbb{E}[\mathbf{p}]\right]
$$

$$
\mathcal{U}_{\text{Aleatoric}} = \mathbb{E}[H(\mathbf{p})]
= \sum_{k=1}^K \frac{\alpha_k}{S}
\big[\psi(S+1)-\psi(\alpha_k+1)\big]
$$

$$
\mathcal{U}_{\text{Epistemic}}
= I(y,\mathbf{p}\mid \boldsymbol{\alpha})
= \mathcal{U}_{\text{Total}} - \mathcal{U}_{\text{Aleatoric}}
\ge 0
$$

Vì vậy công thức phân rã độ bất định mà repo sử dụng là đúng cả về mặt toán học lẫn triển khai số học.

## 8. Vì sao epistemic thường nhỏ và giống bản mờ của aleatoric

Hiện tượng bạn quan sát được là hợp lý về mặt toán học. Trong EDL, cả ba đại lượng đều là hàm của cùng một vector tham số Dirichlet $\boldsymbol{\alpha}$. Điểm khác nhau là:

- $\mathcal{U}_{\text{Total}}$ chỉ phụ thuộc vào tỉ lệ chuẩn hóa $\alpha_k / S$.
- $\mathcal{U}_{\text{Aleatoric}}$ và $\mathcal{U}_{\text{Epistemic}}$ còn phụ thuộc vào độ tập trung của Dirichlet, tức là mức lớn nhỏ của $S$.

Để thấy rõ hơn, xét một tham số hóa:

$$
\alpha_k = c\,\mu_k,
\qquad
\sum_{k=1}^K \mu_k = 1,
\qquad
c = S
$$

Khi đó:

$$
\mathcal{U}_{\text{Total}} = H(\boldsymbol{\mu})
$$

Nghĩa là total uncertainty chỉ nhìn thấy hình dạng của phân phối trung bình $\boldsymbol{\mu}$, không nhìn thấy mức độ chắc chắn của Dirichlet.

Với $\psi(x) = \log x - \frac{1}{2x} + O(x^{-2})$ khi $x$ lớn, ta có:

$$
\mathcal{U}_{\text{Aleatoric}}
= \sum_{k=1}^K \mu_k
\big[\psi(c+1)-\psi(c\mu_k+1)\big]
$$

và do đó, với $c$ đủ lớn:

$$
\mathcal{U}_{\text{Aleatoric}}
\approx H(\boldsymbol{\mu}) - \frac{K-1}{2c}
$$

Suy ra:

$$
\mathcal{U}_{\text{Epistemic}}
= \mathcal{U}_{\text{Total}} - \mathcal{U}_{\text{Aleatoric}}
\approx \frac{K-1}{2c}
$$

Đây là lý do cốt lõi cho hiện tượng bạn thấy:

1. Khi $c=S$ lớn, hậu nghiệm Dirichlet rất sắc, nên epistemic nhỏ.
2. Total và aleatoric cùng được sinh từ cùng một $\boldsymbol{\alpha}$, nên bản đồ epistemic thường mang cùng cấu trúc không gian nhưng chỉ còn biên độ thấp hơn.
3. Nếu mô hình học ra một quy luật gần như đồng nhất cho $S$ trên phần lớn voxel, thì epistemic không thể tự nhiên trở thành một bản đồ hoàn toàn khác biệt về hình thái.

Nói ngắn gọn, epistemic không phải là một kênh thông tin độc lập được vẽ riêng từ ảnh, mà là phần dư của một đại lượng thông tin học. Vì vậy, nó có thể trông như “phiên bản sáng nhẹ” hoặc “phiên bản mờ” của aleatoric mà không hề mâu thuẫn với công thức.

Điều này cũng giải thích vì sao việc chèn dị vật, nhiễu hoặc khối u ghép thêm vào ảnh chưa chắc tạo ra một epistemic map hoàn toàn mới: nếu mô hình vẫn giải thích vùng đó bằng cùng cơ chế tăng/giảm evidence, thì cả $\mathcal{U}_{\text{Total}}$ lẫn $\mathcal{U}_{\text{Aleatoric}}$ đều dịch theo cùng một hướng, và phần dư $\mathcal{U}_{\text{Epistemic}}$ chỉ phản ánh chênh lệch còn lại.

## 9. Kiểm tra bằng số liệu để xem epistemic có thật sự tách biệt không

Để xác nhận xem epistemic có mang thông tin riêng hay chỉ đang đồng biến với aleatoric, nên kiểm tra bằng thống kê thay vì chỉ nhìn ảnh.

### 9.1. So sánh phân phối theo vùng

Tính trung bình, độ lệch chuẩn, median và percentile cho từng map trên các nhóm voxel khác nhau:

- voxel đúng dự đoán
- voxel sai dự đoán
- voxel biên u
- voxel nền

Nếu epistemic thật sự có thông tin riêng, phân phối của nó sẽ tách tốt hơn theo các nhóm này, không chỉ bám theo aleatoric.

### 9.2. Kiểm tra tương quan giữa các bản đồ

Tính tương quan Pearson hoặc Spearman giữa:

$$
\operatorname{corr}(\mathcal{U}_{\text{Epistemic}}, \mathcal{U}_{\text{Aleatoric}}),
\qquad
\operatorname{corr}(\mathcal{U}_{\text{Epistemic}}, \mathcal{U}_{\text{Total}})
$$

Nếu tương quan quá cao trên hầu hết case, điều đó cho thấy epistemic đang gần như là phiên bản co giãn của aleatoric.

### 9.3. Đo khả năng phát hiện lỗi phân đoạn

Xem mỗi voxel sai dự đoán là nhãn dương, rồi dùng từng uncertainty map làm điểm số để tính:

- AUROC
- AUPRC
- accuracy ở một ngưỡng cố định

So sánh riêng cho:

- $\mathcal{U}_{\text{Total}}$
- $\mathcal{U}_{\text{Aleatoric}}$
- $\mathcal{U}_{\text{Epistemic}}$
- tỷ lệ $\mathcal{U}_{\text{Epistemic}} / \mathcal{U}_{\text{Total}}$

Nếu epistemic thật sự phân biệt được, nó phải có khả năng phát hiện lỗi tốt hơn chỉ một bản sao mờ của aleatoric.

### 9.4. Kiểm tra với nhiễu và OOD giả lập

Chạy cùng một ảnh qua các biến thể:

- thêm Gaussian noise
- làm mờ
- ghép dị vật
- chèn khối u giả lập
- crop hoặc perturb vùng ngoài phân bố huấn luyện

Sau đó so sánh độ thay đổi:

$$
\Delta \mathcal{U} = \mathcal{U}_{\text{test}} - \mathcal{U}_{\text{in-distribution}}
$$

Nếu epistemic chỉ là bản sao của aleatoric, thì $\Delta \mathcal{U}_{\text{Epistemic}}$ sẽ gần như đi cùng chiều với $\Delta \mathcal{U}_{\text{Aleatoric}}$ mà không mang tách biệt rõ.

### 9.5. Kiểm tra trực tiếp concentration map

Nên xuất thêm hoặc quan sát bản đồ:

$$
S = \sum_k \alpha_k
$$

vì $S$ là biến điều khiển độ tập trung của Dirichlet. Nếu $S$ cao ở phần lớn voxel, epistemic sẽ bị nén xuống. Nếu $S$ biến thiên có ý nghĩa theo vùng OOD hoặc vùng sai, khi đó epistemic mới có cơ hội tách tốt hơn.

### 9.6. Lưu ý khi hiển thị

Khi so sánh bằng ảnh, không nên min-max normalize từng map riêng lẻ rồi kết luận về độ lớn tương đối. Cách đó có thể làm bản đồ trông rất khác nhau dù giá trị gốc thực tế không khác nhiều. Nên kiểm tra:

- thống kê trên thang gốc
- thống kê sau chuẩn hóa chung
- histogram theo case
