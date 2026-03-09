*Note: Vietnamese version in README_Vietnamese-version.MD.*

# Evidential nnU-Net: Highly Accurate Brain Tumor Segmentation with Reliable Uncertainty Decomposition

This repository contains the source code and experimental documentation for integrating **Evidential Deep Learning (EDL)** into the **nnU-Net v2** framework to solve the multimodal brain tumor segmentation problem on the **BraTS 2020** dataset.

The main objective of the project is to provide highly reliable **Uncertainty Quantification** capabilities and decompose them into two distinct components: **Aleatoric** (due to data noise) and **Epistemic** (due to the model's lack of knowledge). Specifically, the proposed method maintains and surpasses the segmentation performance of the SOTA Baseline model without increasing computational costs (single forward pass).

## Segmentation Result Visualization

Note:

* Top row: MRI Input, Ground Truth, AI Prediction.
* Bottom row: Total Uncertainty, Aleatoric Uncertainty (tumor border data noise), and Epistemic Uncertainty (regions where the model lacks confidence).

<br>

![BraTS axial 50](img/BraTs_012_axial-50.png)
**Figure 1**. Illustration of prediction, uncertainty map generation process, and uncertainty decomposition

![BraTS axial 25 uncertainty](img/BraTs_012_axial-25_uncertainty-hightlight.png)
**Figure 2**. Illustration of uncertainty. It is easy to see that even though the model completely fails in predicting the tumor, the uncertainty map lights up and captures it very well.

---

## 1. Introduction

Automated brain tumor segmentation has achieved extremely impressive performance with advanced 3D CNN architectures like nnU-Net. However, in medicine, simply providing a "hard prediction" is not enough. Doctors need to know **how confident the model is** in that prediction to avoid erroneous interventions.

Traditional uncertainty estimation methods like Monte Carlo Dropout or Deep Ensemble often suffer from two fatal drawbacks:

1. Enormous computational cost (must run inference multiple times or train multiple models).
2. Often trade-off segmentation accuracy (Accuracy) for calibration.

This project proposes **Evidential nnU-Net**, a solution that replaces the traditional Softmax layer with a Dirichlet distribution. This technique directly models the probability density of segmentation classes, allowing uncertainty computation in just **a single forward pass** with superior accuracy. Simultaneously, the Expected Calibration Error (ECE) and QuBraTS scores surpass current SOTA (State of the Art) methods.


## 2. Methodology

### 2.1. Foundation Architecture (nnU-Net v2)

* **Framework:** Inherits the excellent self-configuring capabilities of nnU-Net v2.
* **Backbone Network:** 3D U-Net (Full resolution) processing multimodal inputs (T1, T1ce, T2, FLAIR).
* **Training Strategy:** Standard 5-fold Cross Validation to ensure the highest objectivity and fairness in comparison. We use nnU-Net's default seed for train/test splitting (`split_seed = 12345`).

### 2.2. Evidential Deep Learning (EDL)

We customize nnU-Net's training pipeline through the `EDLLoss` and `EDLTrainer` classes:

$$\mathcal{L}_{\text{Total}} = \underbrace{\mathcal{L}_{\text{Bayes}}}_{\text{Accuracy}} + \lambda_t \cdot \underbrace{\mathcal{L}_{\text{KL}}}_{\text{Reliability}} + \underbrace{\mathcal{L}_{\text{Dice}}(\alpha / S)}_{\text{Segmentation}}$$

* **Evidence Activation Function:** Replaces Softmax with Softplus to ensure the collected evidence is always positive.
* **Multi-component Custom Loss:**

1. **Expected Cross-Entropy (Data Fit Loss):** Optimizes the prediction closely to the Ground Truth based on the expectation of the Dirichlet distribution.
2. **KL Divergence Penalty (with Annealing Step):** Strictly penalizes "overconfident" predictions in regions where the model guesses incorrectly, forcing the model to return to a Uniform distribution when encountering unfamiliar data.
3. **Dice Loss:** Overcomes extreme class imbalance in medical imaging, directly optimizing spatial overlap.

### 2.3. Uncertainty Decomposition Mechanism

Based on Dempster-Shafer evidence theory, the Total Uncertainty is clearly separated into:

$$\underbrace{\mathcal{I}[y,\theta]}_{\text{Epistemic}} = \underbrace{H[\mathbb{E}[p]]}_{\text{Total}} - \underbrace{\mathbb{E}[H[p]]}_{\text{Aleatoric}}$$

### a. Total Uncertainty

Entropy of the expected probability.

$$\mathcal{U}_{Total} = -\sum_k \hat{p}_k \ln \hat{p}_k$$

*(The chaos of the final decision)*

### b. Aleatoric Uncertainty

Expected entropy of each point (using Digamma function $\psi$).

$$\mathcal{U}_{Aleatoric} = \sum_k \hat{p}_k \left( \psi(S+1) - \psi(\alpha_k+1) \right)$$

*(Data noise)*

### c. Epistemic Uncertainty

The remainder of the subtraction:

$$\mathcal{U}_{Epistemic} = \mathcal{U}_{Total} - \mathcal{U}_{Aleatoric}$$

* Measures the **dispersion of the probability cloud** -> Can be used to **detect Out-of-Distribution (OOD) data**

* **Aleatoric Uncertainty (Data Noise):** Originates from poor MRI image quality, blurred boundaries between the tumor and healthy tissue. This uncertainty cannot be reduced even with more training data.
* **Epistemic Uncertainty (Model Ignorance):** Originates from the model never having encountered this type of tumor before (Out-of-Distribution). It can be mitigated by collecting more training data.

## 3. Analysis of the "Overconfidence" Phenomenon and the Superiority of EDL

In medical segmentation problems, deep learning models often suffer from **Overconfidence** syndrome. When trying to optimize the Dice Score, the model tends to push the prediction probability approaching 1.0 (100%) even in ambiguous boundary regions.

The reason is that classification models optimize for accuracy (in the context of this problem, Dice), trying to output the highest probability value in the class the model believes is correct with the goal of pushing the loss function to 0. This inadvertently causes the ECE score of these models to be very high (poor), because the most confident samples are all pushed into the last bin, while the less confident samples in the remaining bins have a high probability of being incorrect predictions. This increases the **Expected Calibration Error (ECE)**.

EDL thoroughly solves this problem by absorbing the "virtual probability" into an Uncertainty metric, helping the model maintain necessary humility.

![Baseline nnU-Net (50 Epochs)](img/reliability_diagram_10bins_nnUnet_base.png)

*> Figure 3a: Baseline nnU-Net (50 Epochs). Despite being in the early stages of training, the model already appears too confident, creating large "Calibration Gaps" (red hatched areas) in the high probability bins. ECE reaches up to 1.536%.*

![Evidential nnU-Net (50 Epochs)](img/reliability_diagram_10bins_EDL50.png)

*> Figure 3b: Evidential nnU-Net (50 Epochs). Thanks to the KL Divergence penalty mechanism, EDL controls confidence extremely well, closely following the Perfect Calibration line (dashed). ECE sharply drops to only 0.616%.*

![Evidential nnU-Net (250 Epochs)](img/reliability_diagram_10bins_EDL250.png)

*> Figure 3c: Evidential nnU-Net (250 Epochs). When trained long-term, the model reaches its peak in accuracy (Dice). Although tending to be more confident, EDL still excellently manages to keep ECE at a safe level of 0.805% (less than half of the 50-epoch Baseline).*

---

## 4. Experimental Results & SOTA Benchmark

The model is rigorously trained and evaluated using the **5-fold Cross Validation** method on the entire BraTS 2020 training dataset (this is the strictest evaluation standard proposed by MICCAI, instead of manually splitting fixed train/test ratios).

### 4.1. Overall Performance (Dice & HD95)

*Configuration: TTA (Test Time Augmentation), 50% sliding window. Lower HD95 is better, higher Dice is better.*

| Method | Epochs | Mean Dice (%) | WT | TC | ET | Mean HD95 (mm) | WT | TC | ET |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline nnU-Net | 50 | 87.17 | 92.52 | 89.92 | 79.07 | 11.59 | 3.75 | 3.69 | 27.34 |
| Evidential nnU-Net | 50 | 85.34 | 92.05 | 87.97 | 76.00 | 12.32 | 3.87 | 4.20 | 28.88 |
| **Evidential nnU-Net (Ours)** | **250** | **88.96** | **93.63** | **92.00** | **81.25** | **10.71** | **2.71** | **2.82** | **26.59** |

*> Observation: When trained for enough epochs (250), Evidential nnU-Net not only does not fall behind but also strongly breaks through, achieving a Mean Dice of approximately 89% and significantly reducing the HD95 gap.*

### 4.2. Comparison with SOTA methods (BraTS 2020)

The results of the proposed method are weighed against the most advanced architectures currently available on the same dataset.

| Method | Whole Tumor (WT) | Tumor Core (TC) | Enhancing Tumor (ET) | Mean Dice |
| --- | --- | --- | --- | --- |
| dResU-Net | 86.60 | 83.57 | 80.04 | 83.40 |
| ADHDC-Net | 89.75 | 83.31 | 78.01 | 83.69 |
| nnU-Net Ensemble | 91.00 | 84.40 | 77.60 | 84.80 |
| H2NF-Net | 91.30 | 85.50 | 78.80 | 86.20 |
| Standard nnU-Net v1 | 91.07 | 87.97 | 81.37 | 87.07 |
| *Energy-Based Attention* | *89.13* | *88.97* | **87.47** | *88.52* |
| **Evidential nnU-Net (Ours)** | **93.63** | **92.00** | 81.25 | **88.96** |

*> Observation: The proposed method establishes a new SOTA level for Mean Dice (88.96%), particularly excelling in segmenting the Whole Tumor (WT: 93.63%) and Tumor Core (TC: 92.00%).*

### 4.3. Uncertainty Quality Evaluation (Uncertainty Benchmark)

**Note:** Most of the reference papers below use their own custom-split train/dev/test datasets (e.g., Region-based EDL splits 236 train / 59 validation / 74 test), whereas our results (Ours) are the average of a **5-fold CV (comprehensive evaluation across the entire data)** to ensure transparency.

| Method | Mean Dice ↑ | ECE (%) ↓ | BraS* ↑ | Notes |
| --- | --- | --- | --- | --- |
| Deep Ensemble | 0.807 | 1.000% | 0.873 | Costs $N$ times more computational power |
| MC Dropout | 0.804 | 0.900% | 0.869 | Must sample multiple times (Multiple passes) |
| EDL (TBraTS, 200 epochs) | 0.790 | 1.900% | 0.862 | Single pass |
| EDL (Region-based, wDICE, 200 epochs) | 0.793 | 1.600% | 0.874 | Single pass |
| **Our EDL (250 epochs)** | **0.889** | **0.805%** | **0.930** | **Single pass, 5-Fold Cross Validation** |

*> Observation: Our method not only achieves an overwhelmingly superior Dice level compared to other EDL variants (0.889 vs 0.790), but also pushes ECE down to a record level (0.805%), surpassing even expensive methods like MC Dropout.*

---

***BraS (QU-BraTS Score).** This is the official metric used in the **BraTS 2020 – Quantification of Uncertainty in Brain Tumor Segmentation (QU-BraTS) challenge** to evaluate the quality of uncertainty maps.

Unlike traditional segmentation metrics that only measure accuracy (like Dice), QU-BraTS evaluates whether **the uncertainty predicted by the model truly correlates with segmentation errors**. In other words, a good model not only predicts accurately but must also **know when it is uncertain**.

This metric aggregates three components:

* **AUC Dice**: measures the correlation between confidence and segmentation quality  
* **AUC FTP (False Tumor Positive)**: penalizes if the model is falsely confident in background regions  
* **AUC FTN (False Tumor Negative)**: penalizes if the model is confidently missing a tumor  

The final score ranges from **[0,1]**, with **higher values being better**.

![QuBraTS Score ET](img/qu_brats_curve_ET.png)

*> Figure 4: Illustration of AUC curves of the ET class with the 250-epoch EDL model.*

## 5. Citation

If you use this source code or take inspiration from this method for your research, please cite the following documents:

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

```
