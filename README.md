# Alligator vs Crocodile Image Classification
### Tilesview AI — Interview Task

Binary image classification pipeline using **MobileNetV2** and **ResNet50**
with PyTorch, comparing both models and selecting the best by Weighted F1 Score.

---

## Project Structure

```
TILESVIEW.AI/
├── dataset/
│   ├── alligator/          # 1,727 images  (jpg, jpeg, png, webp)
│   └── crocodile/          # 1,978 images  (jpg, jpeg, png, webp)
│
├── models/
│   ├── best_model.pth                    # Best model checkpoint (auto-selected)
│   ├── best_mobilenetv2.pth              # MobileNetV2 best checkpoint
│   └── best_resnet50.pth                 # ResNet50 best checkpoint
│
├── outputs/
│   ├── dataset_report.txt                # Dataset quality + imbalance report
│   ├── model_comparison.txt              # MobileNetV2 vs ResNet50 metrics
│   ├── evaluation_report.txt             # Full evaluation report
│   ├── val_indices.json                  # Reproducible validation split
│   ├── confusion_matrix/                 # Confusion matrix PNGs
│   ├── training_plots/                   # Loss / Accuracy / F1 curves
│   ├── misclassified/                    # Plain copies of wrong predictions
│   └── wrong_predictions/               # OpenCV-annotated wrong predictions
│
├── predictions/                          # Inference output (annotated images)
│   └── prediction_summary.txt
│
├── conclusion/
│   └── conclusion.txt                    # One-page professional summary
│
├── dataset_analysis.py                   # Step 0: dataset scan & validation
├── train.py                              # Step 1: full training pipeline
├── evaluate.py                           # Step 2: evaluation & metrics
├── predict.py                            # Step 3: inference on new images
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**GPU users — CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**GPU users — CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Execution Commands

### Step 0 — Dataset Analysis (optional but recommended)
```bash
python dataset_analysis.py
```
Scans every image, flags corrupted/unsupported files, reports class imbalance.
Output → `outputs/dataset_report.txt`

---

### Step 1 — Training
```bash
python train.py
```
- Downloads pretrained ImageNet weights for MobileNetV2 and ResNet50.
- Trains both models for 12 epochs with `ReduceLROnPlateau` scheduler.
- Saves best checkpoint per model (by Validation F1 Score).
- Generates confusion matrices, training curves, error analysis.
- Compares both models and copies the winner to `models/best_model.pth`.

Output artefacts:
```
models/best_model.pth
models/best_mobilenetv2.pth
models/best_resnet50.pth
outputs/confusion_matrix/*.png
outputs/training_plots/*.png
outputs/misclassified/
outputs/wrong_predictions/
outputs/model_comparison.txt
conclusion/conclusion.txt
```

---

### Step 2 — Evaluation
```bash
python evaluate.py
```
Loads `models/best_model.pth`, reconstructs the exact validation split,
and prints/saves a full evaluation report.

Output → `outputs/evaluation_report.txt`

---

### Step 3 — Prediction
```bash
# Predict on a folder of images
python predict.py --input path/to/folder

# Predict a single image
python predict.py --input path/to/image.jpg

# Default: 40 random validation images
python predict.py

# Control sample size
python predict.py --sample 60
```

Each output image has an OpenCV overlay:
```
Predicted: Crocodile  |  Confidence: 92%
```
Output → `predictions/`

---

## Pipeline Design

| Component             | Choice                                |
|-----------------------|---------------------------------------|
| Framework             | PyTorch                               |
| Model A               | MobileNetV2 (frozen backbone)         |
| Model B               | ResNet50 (layer4 + fc unfrozen)       |
| Input resolution      | 224 × 224                             |
| Normalisation         | ImageNet mean/std                     |
| Augmentations         | Flip, Rotation, ColorJitter, Crop     |
| Loss function         | CrossEntropyLoss with class weights   |
| Optimiser             | Adam (lr = 1e-3)                      |
| LR scheduler          | ReduceLROnPlateau (mode=max, pat=3)   |
| Imbalance handling    | WeightedRandomSampler + weighted loss |
| Model selection       | Highest Weighted F1 Score             |
| Epochs                | 12                                    |
| Train / Val split     | 80 % / 20 % (fixed seed = 42)         |

---

## Dataset Summary

| Class      | Count  | Format(s)              |
|------------|--------|------------------------|
| Alligator  | 1,727  | jpg, jpeg, png, webp   |
| Crocodile  | 1,978  | jpg, jpeg, png, webp   |
| **Total**  | **3,705** |                     |

Mild imbalance (~1.14 : 1) — handled with `WeightedRandomSampler`
and `CrossEntropyLoss(weight=...)`.

---

## Key Observations

- **Snout geometry** (broad U = alligator, tapered V = crocodile) is the
  primary visual differentiator but is invisible in pure side-angle shots.
- **Mud/water coverage** erases skin texture cues used by the backbone.
- **Juvenile specimens** share similar proportions across species.
- ResNet50 (unfrozen layer4) typically outperforms MobileNetV2 on this task
  due to richer feature representations, at the cost of ~8× more parameters.

---

## Batch Scripts (Windows)

| Script             | Action                |
|--------------------|-----------------------|
| `1_dataset.bat`    | Run dataset analysis  |
| `2_train.bat`      | Run training          |
| `3_evaluate.bat`   | Run evaluation        |
| `4_predict.bat`    | Run prediction        |

---

*Tilesview AI Interview Task — Professional Deep Learning Pipeline*
