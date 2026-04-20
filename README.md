# 🩸 Blood Cell Classification & Detection — TXL-PBC

Automated hematological diagnosis using deep learning on the **TXL-PBC** dataset. This project implements and compares CNN-based pipelines for both **classification** (identifying individual cell types) and **object detection** (localizing and classifying cells within full microscopy images).

> **Dataset:** Gan et al., *Scientific Data* (Nature), 2025 — [GitHub](https://github.com/lugan113/TXL-PBC_Dataset)  
> **Platform:** Google Colab (GPU T4)

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Highlights](#technical-highlights)

---

## 🎯 Problem Statement

Manual blood cell counting (hémogramme / CBC) is a routine but time-consuming laboratory task. This project automates the identification and localization of three blood cell types from microscopy images, enabling:

- Faster and more reproducible hematological diagnosis
- Reduced workload in clinical laboratories
- Early detection of anomalies such as leukemia, anemia, and thrombocytopenia

---

## 📦 Dataset

**TXL-PBC** — 1,260 YOLO-annotated microscopy images with 3 cell classes:

| Class | Type | Distribution |
|---|---|---|
| **WBC** | Leukocytes (White Blood Cells) | ~7% |
| **RBC** | Erythrocytes (Red Blood Cells) | ~90% |
| **Platelet** | Thrombocytes | ~3% |

| Split | Images |
|---|---|
| Train | 882 |
| Val | 252 |
| Test | 126 |

> ⚠️ **Strong class imbalance**: RBC dominates at 90%. A naive model predicting RBC always would achieve 90% accuracy while learning nothing — motivating the use of `WeightedRandomSampler` and Focal Loss.

Annotations follow the YOLO format: `class_id cx cy w h` (normalized coordinates).

---

## 🗂️ Project Structure

```
TXL-PBC_Dataset/
├── images/
│   ├── train/   # 882 images
│   ├── val/     # 252 images
│   └── test/    # 126 images
└── labels/
    ├── train/   # YOLO .txt annotations
    ├── val/
    └── test/
```

---

## 🔄 Pipeline Overview

```
Raw Images (YOLO annotated)
        │
        ▼
1. EDA ─────────────── Class distribution · Split consistency · BBox statistics
        │
        ▼
2. Preprocessing ────── Resize · ImageNet normalization · Gaussian filter
                        CLAHE (L* channel) · Augmentation · WeightedRandomSampler
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
3. Classification (crop-based)        4. Detection (full image)
   ResNet-18 vs EfficientNet-B0          Faster R-CNN vs YOLOv8n
        │                                      │
        └──────────────────┬───────────────────┘
                           ▼
              5. Final Evaluation & Comparison
```

---

## 🧠 Models

### Classification

Each input is a **single-cell crop** from its bounding box (crop-based approach), eliminating the dominant-class-per-image bias.

| Model | Architecture | Fine-tuning Strategy |
|---|---|---|
| **ResNet-18** | Residual network with skip connections | `layer3` + `layer4` + `fc` unfrozen (~94% params) |
| **EfficientNet-B0** | Compound scaling (depth × width × resolution) | `features.5–8` + classifier unfrozen (~94% params) |

**Training setup:**
- Optimizer: AdamW (decoupled weight decay)
- Scheduler: CosineAnnealingLR
- Loss: Focal Loss (handles class imbalance)
- Gradient clipping: `max_norm=5`
- Early stopping: patience=5
- Seed: `torch.manual_seed(42)`

### Detection

| Model | Architecture | Speed |
|---|---|---|
| **Faster R-CNN** | 2-stage: RPN → classifier (ResNet-50-FPN, ~41M params) | Slower, higher accuracy |
| **YOLOv8n** | 1-stage: direct grid prediction (~3.2M params) | Real-time capable |

Metrics: **mAP@50** and **mAP@50-95** via `torchmetrics.MeanAveragePrecision`.

---

## 📊 Results

### Classification

| Model | Key Strength |
|---|---|
| ResNet-18 | Higher balanced accuracy — benefits from deep fine-tuning and skip connections |
| EfficientNet-B0 | Lighter footprint, competitive under equalized training conditions (V5 correction) |

### Detection

| Model | Key Strength |
|---|---|
| YOLOv8n | Very high mAP@50, real-time inference, single-pass architecture |
| Faster R-CNN | Comparable mAP (after V5 torchmetrics fix), strong 2-stage localization |

---

## ⚙️ Installation

```bash
pip install opencv-python-headless matplotlib seaborn scikit-learn torchvision tqdm
pip install ultralytics
pip install torchmetrics
```

Or run directly in **Google Colab** — the notebook handles all installation automatically.

**Clone the dataset:**
```bash
git clone https://github.com/lugan113/TXL-PBC_Dataset.git
```

---

## 🚀 Usage

Open and run the notebook sequentially in Google Colab (GPU recommended):

1. **Section 0** — Installs dependencies and downloads the dataset
2. **Section 1** — Exploratory data analysis
3. **Section 2** — Preprocessing pipeline setup
4. **Section 3** — Train and evaluate ResNet-18 and EfficientNet-B0
5. **Section 4** — Train and evaluate Faster R-CNN and YOLOv8n
6. **Section 5** — Final comparison and dashboard

---

## 🔬 Technical Highlights

### CLAHE Preprocessing
Contrast Limited Adaptive Histogram Equalization is applied on the **L\* channel** (LAB color space) to enhance local contrast without distorting cell hues — critical for discriminating WBC from RBC by color.

```python
class CLAHETransform:
    def __call__(self, img_pil):
        lab = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2LAB)
        lab[:,:,0] = self.clahe.apply(lab[:,:,0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
```

### Augmentation Pipeline
```
CLAHE → Resize 224×224 → RandomRotation(45°) → RandomHorizontalFlip
     → RandomVerticalFlip → ColorJitter → GaussianBlur → ToTensor → Normalize(ImageNet)
```

Justified biologically: blood cells have no fixed orientation under the microscope, and illumination varies between acquisitions.

### Interpretability — Grad-CAM
Gradient-weighted Class Activation Mapping is applied to ResNet-18 and EfficientNet-B0 to visualize which image regions drive each prediction. Dynamic sample selection per class (no hardcoded indices).

### Key V5 Corrections

| Issue | Fix |
|---|---|
| No mAP for Faster R-CNN | Added `torchmetrics.MeanAveragePrecision` |
| EfficientNet-B0 under-unfrozen (33.6%) | Unfroze `features.5→8` + classifier (~60%) |
| Double normalization in Faster R-CNN | Removed external normalization (torchvision handles internally) |
| Aggressive StepLR scheduler | Replaced with `CosineAnnealingLR` |
| No global seed | Added `SEED=42` across torch, numpy, random, YOLO |
| CLAHE only visualized | Integrated into `train_tfm` via `CLAHETransform` |

---

## 📚 Reference

Gan et al., "TXL-PBC: A large-scale peripheral blood cell dataset", *Scientific Data* (Nature), 2025.  
Dataset: https://github.com/lugan113/TXL-PBC_Dataset
