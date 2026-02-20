"""
evaluate.py
============
Alligator vs Crocodile Classification — Tilesview AI Interview Task

Loads the saved best model, reconstructs the exact validation split,
then computes and prints full evaluation metrics:
  • Accuracy, Precision, Recall, F1 Score
  • Per-class classification report
  • Confusion matrix (saved to outputs/confusion_matrix/)

Run:
    python evaluate.py
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
MODEL_DIR   = ROOT / "models"
OUTPUTS     = ROOT / "outputs"
CM_DIR      = OUTPUTS / "confusion_matrix"
VAL_IDX_FILE = OUTPUTS / "val_indices.json"
BEST_CKPT    = MODEL_DIR / "best_model.pth"

IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VALID_EXT     = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
BATCH_SIZE    = 32

CM_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────
class ImageDataset(torch.utils.data.Dataset):
    """Identical to the one in train.py — keeps evaluate.py self-contained."""

    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.classes      = sorted([d.name for d in root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: list[tuple[str, int]] = []
        self.targets: list[int]             = []

        for cls in self.classes:
            cls_dir = root / cls
            for fp in sorted(cls_dir.iterdir()):
                if fp.suffix.lower() not in VALID_EXT:
                    continue
                label = self.class_to_idx[cls]
                self.samples.append((str(fp), label))
                self.targets.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device) -> tuple[nn.Module, list[str], str]:
    """
    Load the best model from a checkpoint created by train.py.

    Supports MobileNetV2 and ResNet50.
    Returns (model, class_names, model_name_str).
    """
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print("        Run  python train.py  first.")
        sys.exit(1)

    ckpt        = torch.load(ckpt_path, map_location=device)
    model_name  = ckpt.get("model_name", "ResNet50")
    class_names = ckpt.get("classes",    ["alligator", "crocodile"])
    num_classes = len(class_names)

    if "MobileNetV2" in model_name:
        net = models.mobilenet_v2(weights=None)
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(net.last_channel, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )
    else:
        net = models.resnet50(weights=None)
        in_feat = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_feat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    net.load_state_dict(ckpt["model_state"])
    net.eval()
    net.to(device)
    print(f"[Model] Loaded  : {model_name}")
    print(f"[Model] Classes : {class_names}")
    return net, class_names, model_name


# ── Evaluation ────────────────────────────────────────────────────────────────
def run_evaluation(model, val_loader, device, class_names):
    """
    Full evaluation pass.
    Returns (all_preds, all_labels, all_probs).
    """
    model.eval()
    all_preds:  list[int]   = []
    all_labels: list[int]   = []
    all_probs:  list[float] = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.max(dim=1).values.cpu().numpy())

    preds_np  = np.array(all_preds)
    labels_np = np.array(all_labels)
    probs_np  = np.array(all_probs)

    acc  = accuracy_score(labels_np, preds_np)
    prec = precision_score(labels_np, preds_np, average="weighted", zero_division=0)
    rec  = recall_score(labels_np, preds_np, average="weighted", zero_division=0)
    f1   = f1_score(labels_np, preds_np, average="weighted", zero_division=0)
    cm   = confusion_matrix(labels_np, preds_np)
    report = classification_report(labels_np, preds_np,
                                   target_names=class_names, zero_division=0)

    return {
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1":         f1,
        "cm":         cm,
        "report":     report,
        "preds":      preds_np,
        "labels":     labels_np,
        "probs":      probs_np,
    }


def save_confusion_matrix(cm, class_names, model_name, save_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, annot_kws={"size": 16},
    )
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label",      fontsize=13)
    ax.set_title(f"Confusion Matrix — {model_name} (Evaluation)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    fname = f"eval_confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    path  = save_dir / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Confusion matrix -> {path}")
    return path


def save_confidence_histogram(probs, preds, labels, class_names, model_name, save_dir):
    """Histogram of prediction confidence — useful for calibration check."""
    correct   = probs[preds == labels]
    incorrect = probs[preds != labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(correct,   bins=20, alpha=0.7, color="steelblue", label=f"Correct   (n={len(correct)})")
    ax.hist(incorrect, bins=20, alpha=0.7, color="salmon",    label=f"Incorrect (n={len(incorrect)})")
    ax.set_xlabel("Prediction Confidence", fontsize=12)
    ax.set_ylabel("Count",                 fontsize=12)
    ax.set_title(f"Confidence Distribution — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"eval_confidence_{model_name.lower().replace(' ', '_')}.png"
    path  = save_dir / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Confidence hist  -> {path}")


def save_eval_report(metrics: dict, model_name: str, class_names: list[str]) -> None:
    """Write a plain-text evaluation report to outputs/evaluation_report.txt."""
    from datetime import datetime
    sep = "=" * 62

    lines = [
        sep,
        "  EVALUATION REPORT",
        "  Alligator vs Crocodile — Tilesview AI",
        f"  Model    : {model_name}",
        f"  Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep, "",
        "  METRICS (Weighted Average — Validation Set)",
        "-" * 44,
        f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f} %)",
        f"  Precision : {metrics['precision']:.4f}",
        f"  Recall    : {metrics['recall']:.4f}",
        f"  F1 Score  : {metrics['f1']:.4f}",
        "",
        "  CLASSIFICATION REPORT",
        "-" * 44,
        metrics["report"],
        "",
        "  CONFUSION MATRIX",
        "-" * 44,
    ]

    cm = metrics["cm"]
    header = "  " + "".join(f"{c:>15s}" for c in class_names)
    lines.append(header)
    for i, row in enumerate(cm):
        row_str = "  " + f"{class_names[i]:15s}" + "".join(f"{v:>15d}" for v in row)
        lines.append(row_str)

    lines += ["", sep]

    report_text = "\n".join(lines)
    path = OUTPUTS / "evaluation_report.txt"
    path.write_text(report_text, encoding="utf-8")
    print(f"\n  [Saved] Evaluation report -> {path}")
    print(report_text)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model, class_names, model_name = load_model(BEST_CKPT, device)

    # ── Reconstruct validation split ─────────────────────────────────────────
    if not VAL_IDX_FILE.exists():
        print("[ERROR] Val-index file not found:", VAL_IDX_FILE)
        print("        Run  python train.py  first to create it.")
        sys.exit(1)

    with open(VAL_IDX_FILE) as fh:
        idx_data = json.load(fh)

    val_indices = idx_data["indices"]
    print(f"[Split ] Val indices loaded  : {len(val_indices)} samples")

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    base_dataset = ImageDataset(DATASET_DIR, transform=val_tf)
    val_subset   = Subset(base_dataset, val_indices)
    val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[Evaluating] Please wait ...")
    metrics = run_evaluation(model, val_loader, device, class_names)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  EVALUATION RESULTS — {model_name}")
    print("=" * 62)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f} %)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"\n  Classification Report:\n{metrics['report']}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    save_confusion_matrix(metrics["cm"], class_names, model_name, CM_DIR)
    save_confidence_histogram(
        metrics["probs"], metrics["preds"], metrics["labels"],
        class_names, model_name, CM_DIR,
    )
    save_eval_report(metrics, model_name, class_names)

    print("\n[Done] Evaluation complete.")


if __name__ == "__main__":
    main()
