# TransEffiNet-14

**Explainable Hybrid EfficientNet-B4 + Transformer for Multi-Label Chest X-Ray Disease Classification**

A lightweight hybrid deep learning framework that combines EfficientNet-B4's local feature extraction with a compact Transformer encoder's global context modelling, trained on the NIH ChestX-ray14 dataset to detect 14 thoracic diseases simultaneously — with built-in Grad-CAM explainability and a prototype confidence-report generator.

> 📄 Based on our paper *"Explainable Hybrid EfficientNet-B4 and Transformer Model for Multi-Label Lung Disease Detection from Chest X-Ray Images"* (submitted to IEEE CEECT 2026)

---

## 📋 Project Details

| | |
|---|---|
| **Supervisor** | Ms. M. Sowjanya |
| **Email** | [sowjanya1637@vardhaman.org](mailto:sowjanya1637@vardhaman.org) |
| **Contact** | 6300890083 |
| **Project Type** | Mini Project / EPBL Internship |
| **Domain** | Artificial Intelligence, Machine Learning, AI |

---

## ✨ Highlights

- **Mean AUC-ROC of 0.7680** across all 14 disease classes on held-out validation data
- **Only ~1.9M extra parameters** over the EfficientNet-B4 backbone (21.2M total) — the Transformer encoder is deliberately lightweight
- **CLAHE preprocessing + Focal Loss + weighted sampling** to handle NIH ChestX-ray14's severe class imbalance (Hernia and Pneumonia together make up <1% of positive labels)
- **Grad-CAM explainability** built in — every prediction comes with a class-specific heatmap, not just a probability
- **Prototype confidence-report module** that turns raw probabilities into a structured, readable clinical-style summary
- Live demo deployed on Hugging Face Spaces

---

## 🏗️ Architecture

```
Chest X-ray (224×224×3)
        │
        ▼
CLAHE Enhancement (clip limit 2.0, tile grid 8×8)
        │
        ▼
EfficientNet-B4 Backbone  →  feature map (B, 448, 7, 7)
        │
        ▼
1×1 Conv + BatchNorm + GELU  →  448 → 512 channels
        │
        ▼
Flatten to 49 tokens (7×7 → 49 × 512)
        │
        ▼
Transformer Encoder (2 layers, 8-head self-attention, dropout 0.3)
        │
        ▼
Global Average Pooling → 512-d vector
        │
        ▼
Dropout(0.3) → Linear(512 → 14) → Sigmoid
        │
        ▼
14 independent disease probabilities
        │
        ├──→ Grad-CAM heatmap (class-specific)
        └──→ Confidence-level report (High / Moderate / Low / No significant finding)
```

The Transformer treats the CNN's spatial feature map as a sequence of 49 tokens, letting every region attend to every other region — so the model can directly relate, say, the cardiac silhouette to thoracic width for Cardiomegaly, or both costophrenic angles for bilateral Effusion, which a convolution-only model can't do without much deeper stacking.

---

## 📊 Results

### Overall performance

| Metric | Value |
|---|---|
| Mean AUC-ROC (14 classes) | **0.7680** |
| Best class | Emphysema (0.881) |
| Weakest class | Pneumonia (0.605) |
| Model parameters | 21.2M |
| Params added over EfficientNet-B4 backbone | ~1.9M |

### Per-class AUC-ROC

| Disease | AUC-ROC | Tier |
|---|---|---|
| Emphysema | 0.881 | High |
| Edema | 0.868 | High |
| Cardiomegaly | 0.861 | High |
| Effusion | 0.845 | High |
| Pneumothorax | 0.842 | High |
| Hernia | 0.826 | High |
| Mass | 0.766 | Moderate |
| Atelectasis | 0.746 | Moderate |
| Fibrosis | 0.744 | Moderate |
| Pleural Thickening | 0.739 | Moderate |
| Consolidation | 0.724 | Moderate |
| Nodule | 0.662 | Low |
| Infiltration | 0.642 | Low |
| Pneumonia | 0.605 | Low |

### Ablation study

| Configuration | Mean AUC |
|---|---|
| EfficientNet-B4 only (BCE loss) | 0.6124 |
| + CLAHE preprocessing | 0.6590 |
| + Focal loss (γ=2) | 0.7042 |
| + Transformer (1 layer) | 0.7315 |
| **+ Transformer (2 layers) — Full model** | **0.7680** |

Each component contributes a complementary, measurable gain — the Transformer encoder provides the single largest improvement, confirming that self-attention over spatial tokens captures information convolution alone misses.

---

## 🖼️ Demo

Try it live: **[Hugging Face Space](https://huggingface.co/spaces/Siripodishetti18/lung-disease-ai)**

Upload a chest X-ray and get:
- Per-class disease probabilities
- Grad-CAM heatmap overlay on the predicted region
- A structured confidence report (e.g. *"Infiltration — Confidence Score 0.72 — High confidence — Grad-CAM highlights the upper-left lung region"*)

---

## 📁 Repository Structure

```
.
├── data/
│   └── ...                  # NIH ChestX-ray14 (not included — see Dataset section)
├── notebooks/
│   ├── training.ipynb       # Model training pipeline
│   └── evaluation.ipynb     # Final evaluation, ROC curves, confusion matrix, Grad-CAM
├── src/
│   ├── model.py             # HybridEfficientNetTransformer architecture
│   ├── dataset.py           # ChestXrayDataset + CLAHE preprocessing
│   ├── losses.py            # Focal loss implementation
│   ├── train.py             # Training loop with AdamW + OneCycleLR
│   └── gradcam.py           # Grad-CAM explanation module
├── checkpoints/
│   └── best_hybrid_model.pth
├── app.py                   # Gradio demo app
├── requirements.txt
└── README.md
```

*(adjust this section to match your actual repo layout)*

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/TransEffiNet-14.git
cd TransEffiNet-14
pip install -r requirements.txt
```

**Key dependencies:**
```
torch
timm
opencv-python
scikit-learn
pandas
numpy
matplotlib
gradio
```

---

## 🚀 Usage

### Training

```python
from src.model import HybridEfficientNetTransformer
from src.train import train_model

model = HybridEfficientNetTransformer(num_classes=14)
train_model(model, train_loader, val_loader, epochs=20, patience=7)
```

Training uses AdamW with discriminative learning rates per sub-module (backbone: 1e-4, channel-reduction: 3e-4, Transformer: 3e-4, classifier: 5e-4), a OneCycleLR schedule (30% warm-up, cosine annealing), gradient-norm clipping at 1.0, and batch size 16, on dual NVIDIA T4 GPUs.

### Inference

```python
import torch
from src.model import HybridEfficientNetTransformer

model = HybridEfficientNetTransformer(num_classes=14)
model.load_state_dict(torch.load("checkpoints/best_hybrid_model.pth"))
model.eval()

with torch.no_grad():
    probs = torch.sigmoid(model(image_tensor))
```

---

## 🗂️ Dataset

This project uses the **[NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)** dataset:
- 112,120 frontal chest radiographs
- 30,805 unique patients
- 14 thoracic disease labels, NLP-extracted from radiology reports
- Multi-label, multi-class — a single image can show multiple co-occurring diseases

The dataset is not included in this repository due to size — download it from the [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) or via [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data).

---

## 🔬 Method Summary

| Component | Purpose |
|---|---|
| **CLAHE** (clip limit 2.0, tile grid 8×8) | Boosts visibility of low-contrast findings like faint infiltrations |
| **EfficientNet-B4** | Local feature extraction backbone, pretrained on ImageNet |
| **1×1 Conv bottleneck** | Projects 448-channel features into a 512-d token space |
| **Transformer Encoder** (2 layers, 8 heads) | Models long-range spatial dependencies between lung regions |
| **Focal Loss** (γ=2) | Down-weights easy negatives, focuses learning on hard/minority classes |
| **Weighted Random Sampler** | Oversamples rare-disease images during training |
| **Grad-CAM** | Class-specific visual explanation for every prediction |

---

## 🔮 Future Work

- Validate on external datasets (CheXpert, MIMIC-CXR) for cross-institution generalisation
- Quantitatively evaluate Grad-CAM localisation against expert-annotated bounding boxes
- Class-specific augmentation and explicit label-correlation modelling to improve low-AUC classes (Pneumonia, Infiltration)
- Replace the rule-based report module with a learned Transformer-based report generator
- Prospective clinical evaluation in real-world screening workflows

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@inproceedings{narayani2026transeffinet,
  title={Explainable Hybrid EfficientNet-B4 and Transformer Model for Multi-Label Lung Disease Detection from Chest X-Ray Images},
  author={Narayani, P. Sirilakshmi and Vasantha, S. V. and Rao, M. Sharvanth and Ram, B. Tulasi},
  booktitle={Proc. IEEE CEECT},
  year={2026},
  note={In press}
}
```

---

## ⚠️ Disclaimer

This is a research prototype and is **not a certified medical device**. Grad-CAM heatmaps and confidence reports are intended to support, not replace, professional clinical judgment. It has not been validated for real-world diagnostic use.

---

## 🙏 Acknowledgments

- Department of Computer Science and Engineering, Vardhaman College of Engineering, Hyderabad
- NIH Clinical Center for the ChestX-ray14 dataset
- Built with [PyTorch](https://pytorch.org/) and [timm](https://github.com/huggingface/pytorch-image-models)

---

## 📄 License

*(add your chosen license here — e.g. MIT, Apache 2.0)*

---

## 👥 Contributors

- P. Sirilakshmi Narayani — [siripodishetti18@gmail.com](mailto:siripodishetti18@gmail.com)
- S. V. Vasantha
- M. Sharvanth Rao
- B. Tulasi Ram
