# Alligator vs Crocodile — Image Classification

**Tilesview AI Interview Task**

Binary image classification using deep learning (PyTorch).
Two models trained and compared — best selected automatically by Weighted F1 Score.
Includes a live Streamlit dashboard for real-time inference.

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| MobileNetV2 | 65.86% | 0.6604 | 0.6586 | 0.6585 |
| **ResNet50** | **67.75%** | **0.6787** | **0.6775** | **0.6758** |

**Best Model: ResNet50** — selected by highest Weighted F1 Score on the validation set (n = 741).

---

## Project Overview

This project builds a complete end-to-end pipeline to classify images of alligators and crocodiles using transfer learning on pretrained ImageNet models.

**What it does:**
- Analyses the dataset for class imbalance and corrupted files
- Trains MobileNetV2 and ResNet50 with data augmentation and imbalance handling
- Evaluates both models with Accuracy, Precision, Recall, F1, and confusion matrices
- Selects the best model automatically
- Runs inference on new images with confidence overlay
- Provides an interactive Streamlit dashboard for live predictions

---

## Dataset

| Class | Images | Formats |
|-------|--------|---------|
| Alligator | 1,727 | jpg, jpeg, png, webp |
| Crocodile | 1,977 | jpg, jpeg, png, webp |
| **Total** | **3,704** | |

Mild class imbalance (1.14 : 1) handled with `WeightedRandomSampler` + weighted `CrossEntropyLoss`.

---

## Model Architecture

### MobileNetV2 — Lightweight Baseline
- Pretrained ImageNet backbone (fully frozen)
- Custom 2-layer classifier head
- ~300K trainable parameters
- Fast training, suitable for edge deployment

### ResNet50 — Deep Feature Extractor
- Pretrained ImageNet backbone (layers 1–3 frozen, layer4 fine-tuned)
- Custom 2-layer classifier head
- ~6.5M trainable parameters
- Stronger spatial feature representations

---

## Training Setup

| Component | Detail |
|-----------|--------|
| Framework | PyTorch |
| Input size | 224 × 224 |
| Normalisation | ImageNet mean/std |
| Augmentation | RandomFlip, RandomRotation, ColorJitter, RandomResizedCrop |
| Loss | CrossEntropyLoss with class weights |
| Optimiser | Adam (lr = 1e-3) |
| Scheduler | ReduceLROnPlateau (patience = 3) |
| Epochs | 12 |
| Train / Val split | 80% / 20% (seed = 42) |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/sanjaypatel2889/alligator-crocodile-classifier.git
cd alligator-crocodile-classifier

# Install dependencies
pip install -r requirements.txt
```

**GPU (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> Add your dataset folder as `dataset/alligator/` and `dataset/crocodile/` before running.

---

## Run the Pipeline

### Step 1 — Dataset Analysis
```bash
python dataset_analysis.py
```
Scans images, flags corrupted files, reports class distribution.

### Step 2 — Training
```bash
python train.py
```
Trains both models, evaluates, generates confusion matrices and training curves,
saves the best model to `models/best_model.pth`.

### Step 3 — Evaluate
```bash
python evaluate.py
```
Loads the best checkpoint and prints full metrics on the validation set.

### Step 4 — Predict
```bash
python predict.py                          # 40 random val images
python predict.py --input path/to/image.jpg   # single image
python predict.py --input path/to/folder      # entire folder
```
Each output image has an OpenCV confidence overlay:
```
Predicted: Crocodile  |  Confidence: 87%
```

### Step 5 — Dashboard
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## Streamlit Dashboard

6-page interactive dashboard:

| Page | Content |
|------|---------|
| Home | Project overview and best model metrics |
| Dataset Analysis | Class distribution chart and imbalance report |
| Live Prediction | Upload any image → instant Alligator / Crocodile result |
| Training Results | Training curves and confusion matrices |
| Error Analysis | Grid of misclassified images |
| Conclusion | Full conclusion document and model comparison chart |

---

## Project Structure

```
alligator-crocodile-classifier/
├── dataset_analysis.py       # Dataset scan and validation
├── train.py                  # Full training pipeline
├── evaluate.py               # Metrics and evaluation report
├── predict.py                # Inference with OpenCV overlay
├── app.py                    # Streamlit dashboard
├── requirements.txt
│
├── conclusion/
│   └── conclusion.txt        # One-page project conclusion
│
├── outputs/
│   ├── confusion_matrix/     # Confusion matrix PNGs
│   ├── training_plots/       # Loss / Accuracy / F1 curves
│   └── model_comparison.txt  # Side-by-side metric comparison
│
├── predictions/              # Sample inference output images
│
└── models/                   # Trained checkpoints (not tracked in git)
    ├── best_model.pth
    ├── best_mobilenetv2.pth
    └── best_resnet50.pth
```

---

## Key Observations

- **Snout geometry** (broad U = alligator, tapered V = crocodile) is the primary visual differentiator — invisible in side-angle shots
- **Mud/water coverage** erases skin texture cues that the backbone relies on
- **Juvenile specimens** share similar proportions across species, increasing confusion
- ResNet50 achieved 74% recall on crocodiles, showing stronger discrimination for the majority class

---

## Submission

| Deliverable | File |
|-------------|------|
| Conclusion document | `conclusion/conclusion.txt` |
| Prediction images | `predictions/` (28 annotated images) |
| Model comparison | `outputs/model_comparison.txt` |

---

*Sriram Sanjay — Tilesview AI Interview Task*
