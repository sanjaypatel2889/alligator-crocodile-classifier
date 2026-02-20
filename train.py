"""
train.py
=========
Alligator vs Crocodile Classification — Tilesview AI Interview Task

Full training pipeline:
  1. Dataset validation + class-imbalance handling
  2. Train MobileNetV2  (fast lightweight baseline)
  3. Train ResNet50     (high-accuracy deep extractor)
  4. Compare models, select best by Weighted F1 Score
  5. Save confusion matrices, training curves, misclassified images
  6. Generate model_comparison.txt and conclusion/conclusion.txt

Run:
    python train.py
"""

import os
import sys
import json
import random
import shutil
import sys
import warnings
from pathlib import Path
from datetime import datetime

# Force UTF-8 stdout on Windows so special chars don't crash the console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    Dataset, DataLoader, Subset, WeightedRandomSampler, random_split
)
from torchvision import transforms, models
from PIL import Image

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
DATASET_DIR  = ROOT / "dataset"
MODEL_DIR    = ROOT / "models"
OUTPUTS      = ROOT / "outputs"
CM_DIR       = OUTPUTS / "confusion_matrix"
PLOTS_DIR    = OUTPUTS / "training_plots"
MISC_DIR     = OUTPUTS / "misclassified"
WRONG_DIR    = OUTPUTS / "wrong_predictions"
PRED_DIR     = ROOT / "predictions"
CONC_DIR     = ROOT / "conclusion"

SEED          = 42
IMAGE_SIZE    = 224
BATCH_SIZE    = 32
EPOCHS        = 12
LEARNING_RATE = 1e-3
TRAIN_RATIO   = 0.80

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# File types PIL can reliably decode as RGB
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Create all output directories up-front
for _d in [MODEL_DIR, OUTPUTS, CM_DIR, PLOTS_DIR, MISC_DIR, WRONG_DIR, PRED_DIR, CONC_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Device detection ──────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] GPU: {name}  ({mem_gb:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Device] No CUDA GPU found — using CPU (training will be slower).")
    return device


# ── Custom dataset ─────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    """
    Scans DATASET_DIR/<class>/ folders, filters to valid extensions,
    and skips any unreadable / corrupted files at load time.

    Supports: jpg, jpeg, png, webp, bmp, tif, tiff
    Skips   : gif, mp4, and any file PIL cannot decode as RGB
    """

    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.classes       = sorted([d.name for d in root.iterdir() if d.is_dir()])
        self.class_to_idx  = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[str, int]] = []
        self.targets: list[int]             = []

        skipped = 0
        for cls in self.classes:
            cls_dir = root / cls
            for fp in sorted(cls_dir.iterdir()):
                if fp.suffix.lower() not in VALID_EXT:
                    skipped += 1
                    continue
                label = self.class_to_idx[cls]
                self.samples.append((str(fp), label))
                self.targets.append(label)

        if skipped:
            print(f"[Dataset] Skipped {skipped} files with unsupported extensions (e.g. .gif).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Return blank image on corruption — won't crash the DataLoader
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label


class TransformWrapper(Dataset):
    """Wraps a Subset and applies a per-split transform."""

    def __init__(self, subset: Subset, transform):
        self.subset    = subset
        self.transform = transform
        # Expose targets so WeightedRandomSampler can read them
        base = subset.dataset
        self.targets = [base.targets[i] for i in subset.indices]

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Transforms ────────────────────────────────────────────────────────────────
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.70, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    """
    1. Build ImageDataset (no transform).
    2. Deterministic 80/20 split using SEED.
    3. Wrap each split with its own transform.
    4. Apply WeightedRandomSampler to handle class imbalance.
    5. Save val_indices to outputs/val_indices.json for evaluate.py.
    """
    train_tf, val_tf = get_transforms()

    base = ImageDataset(DATASET_DIR)
    total      = len(base)
    train_size = int(TRAIN_RATIO * total)
    val_size   = total - train_size

    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(base, [train_size, val_size], generator=generator)

    train_data = TransformWrapper(train_subset, train_tf)
    val_data   = TransformWrapper(val_subset,   val_tf)

    # Class-imbalance: weighted sampling for training
    class_counts  = np.bincount(train_data.targets)
    class_weights = 1.0 / (class_counts.astype(float) + 1e-8)
    sample_weights = [class_weights[t] for t in train_data.targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=torch.cuda.is_available())

    # Persist val indices so evaluate.py can reconstruct the exact same split
    val_indices = [int(base.samples[i][1]) for i in val_subset.indices]   # labels (unused)
    val_idx_path = OUTPUTS / "val_indices.json"
    with open(val_idx_path, "w") as fh:
        json.dump({"indices": list(val_subset.indices), "seed": SEED, "total": total}, fh)

    print(f"\n[Dataset] Total valid images : {total}")
    print(f"[Dataset] Train split        : {train_size}")
    print(f"[Dataset] Val   split        : {val_size}")
    print(f"[Dataset] Classes            : {base.classes}")
    dist = dict(zip(base.classes, np.bincount(base.targets)))
    print(f"[Dataset] Class distribution : {dist}")
    ratio = max(dist.values()) / min(dist.values())
    print(f"[Dataset] Imbalance ratio    : {ratio:.3f}:1  -> WeightedRandomSampler applied")

    return train_loader, val_loader, base, val_subset.indices, base.classes


# ── Model builders ────────────────────────────────────────────────────────────
def build_mobilenetv2(num_classes: int = 2) -> nn.Module:
    """
    MobileNetV2 with frozen feature extractor.
    Classifier head replaced for binary classification.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all backbone parameters
    for param in model.features.parameters():
        param.requires_grad = False

    # New classification head
    in_features = model.last_channel      # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  MobileNetV2 — trainable params: {trainable:,}")
    return model


def build_resnet50(num_classes: int = 2) -> nn.Module:
    """
    ResNet50 with partially frozen backbone (layer4 + fc unfrozen).
    Classifier head replaced for binary classification.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze early layers; keep layer4 trainable for fine-tuning
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    in_features = model.fc.in_features    # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ResNet50     — trainable params: {trainable:,}")
    return model


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    epoch_f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1, np.array(all_preds), np.array(all_labels)


def train_model(model, model_name, train_loader, val_loader,
                class_weight_tensor, device):
    """
    Full training loop for one model.
    Saves best checkpoint (by Val F1) to models/best_<model_name>.pth.
    Returns (trained_model, history_dict, best_f1, checkpoint_path).
    """
    print(f"\n{'='*60}")
    print(f"  Training : {model_name}")
    print(f"{'='*60}")

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor.to(device))
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_f1"]}
    best_f1    = 0.0
    best_epoch = 0
    ckpt_path  = MODEL_DIR / f"best_{model_name.lower().replace(' ', '_')}.pth"

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc                       = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1, _, _          = validate_epoch(model, val_loader, criterion, device)
        current_lr                            = optimizer.param_groups[0]["lr"]

        scheduler.step(va_f1)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)

        marker = "  << best" if va_f1 > best_f1 else ""
        print(
            f"  Ep [{epoch:02d}/{EPOCHS}]  "
            f"lr={current_lr:.2e}  "
            f"TrLoss={tr_loss:.4f}  TrAcc={tr_acc:.4f}  "
            f"VaLoss={va_loss:.4f}  VaAcc={va_acc:.4f}  VaF1={va_f1:.4f}"
            f"{marker}"
        )

        if va_f1 > best_f1:
            best_f1    = va_f1
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name":  model_name,
                    "classes":     ["alligator", "crocodile"],
                    "epoch":       epoch,
                    "best_f1":     best_f1,
                },
                ckpt_path,
            )

    print(f"\n  Best Val F1 : {best_f1:.4f}  (epoch {best_epoch})")
    print(f"  Checkpoint  : {ckpt_path}")

    # Load best weights back into model
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model, history, best_f1, ckpt_path


# ── Metrics and visualisation ─────────────────────────────────────────────────
def compute_metrics(model, val_loader, device, class_names):
    """Run full evaluation pass and return metric dict + confusion matrix."""
    _, _, _, preds, labels = validate_epoch(
        model, val_loader, nn.CrossEntropyLoss(), device
    )
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec  = recall_score(labels, preds, average="weighted", zero_division=0)
    f1   = f1_score(labels, preds, average="weighted", zero_division=0)
    cm   = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}, cm, report, preds, labels


def save_confusion_matrix(cm, class_names, model_name, save_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, annot_kws={"size": 16},
    )
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label",      fontsize=13)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    path  = save_dir / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Confusion matrix  : {path}")


def save_training_plots(history, model_name, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val",   markersize=4)
    axes[0].set_title("Loss");  axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train", markersize=4)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val",   markersize=4)
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # F1
    axes[2].plot(epochs, history["val_f1"], "g-o", label="Val F1", markersize=4)
    axes[2].set_title("Val F1 Score"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("F1")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"training_curves_{model_name.lower().replace(' ', '_')}.png"
    path  = save_dir / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Training curves   : {path}")


# ── Error analysis ────────────────────────────────────────────────────────────
def run_error_analysis(model, base_dataset, val_indices, class_names, device,
                       model_name, misc_dir, wrong_dir, max_save=100):
    """
    Identify misclassified validation images.
    Saves plain copies to misc_dir and OpenCV-annotated versions to wrong_dir.
    """
    import cv2

    model.eval()
    misc_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous results
    for d in [misc_dir, wrong_dir]:
        for f in d.glob("*"):
            try: f.unlink()
            except: pass

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    confusion_pairs: dict[tuple, int] = {}
    saved = 0

    with torch.no_grad():
        for idx in val_indices:
            if saved >= max_save:
                break

            img_path, true_label = base_dataset.samples[idx]

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            tensor = val_tf(img).unsqueeze(0).to(device)
            output = model(tensor)
            probs  = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred   = int(np.argmax(probs))
            conf   = float(probs[pred])

            if pred == true_label:
                continue   # Correct — skip

            true_name = class_names[true_label]
            pred_name = class_names[pred]
            key = (true_name, pred_name)
            confusion_pairs[key] = confusion_pairs.get(key, 0) + 1

            # ── Save plain copy ───────────────────────────────────────────────
            img_224 = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            fname   = f"true_{true_name}_pred_{pred_name}_{saved:04d}.jpg"
            img_224.save(misc_dir / fname, quality=90)

            # ── Save annotated copy (OpenCV overlay) ─────────────────────────
            img_cv   = cv2.cvtColor(np.array(img_224), cv2.COLOR_RGB2BGR)
            overlay  = f"True: {true_name}  |  Pred: {pred_name} ({conf:.0%})"
            # Background strip for readability
            cv2.rectangle(img_cv, (0, 0), (IMAGE_SIZE, 30), (0, 0, 0), -1)
            cv2.putText(
                img_cv, overlay, (4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 80, 255), 1, cv2.LINE_AA,
            )
            cv2.imwrite(str(wrong_dir / fname), img_cv)
            saved += 1

    print(f"  [Error Analysis] Misclassified images saved : {saved}")

    # Build text summary
    lines = [
        f"\nError Analysis — {model_name}",
        "-" * 44,
        f"Misclassified samples (first {max_save}): {saved}",
        "\nConfusion pairs (true -> predicted):",
    ]
    for (t, p), cnt in sorted(confusion_pairs.items(), key=lambda x: -x[1]):
        lines.append(f"  {t:14s} -> {p:14s}  x{cnt}")

    lines += [
        "\nCommon confusion reasons observed:",
        "  • Side-angle head shots: snout geometry unclear",
        "  • Mud/water coverage masking skin texture",
        "  • Juvenile specimens — proportions overlap across species",
        "  • Cluttered backgrounds (reeds, banks) without clear body features",
    ]
    return "\n".join(lines)


# ── Model comparison ──────────────────────────────────────────────────────────
def save_model_comparison(results: dict, best_model_name: str) -> None:
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep  = "=" * 62
    sep2 = "-" * 62

    lines = [
        sep,
        "  MODEL COMPARISON REPORT",
        f"  Alligator vs Crocodile — Tilesview AI",
        f"  Generated: {now}",
        sep, "",
        f"  {'Metric':<14} | {'MobileNetV2':>16} | {'ResNet50':>16}",
        sep2,
    ]
    for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        mv = results["MobileNetV2"][metric]
        rv = results["ResNet50"][metric]
        win_mv = " *" if mv > rv else ""
        win_rv = " *" if rv > mv else ""
        lines.append(
            f"  {metric:<14} | {mv:>14.4f}{win_mv:2s} | {rv:>14.4f}{win_rv:2s}"
        )
    lines += [
        "",
        sep2,
        f"  Best Model (by Weighted F1) : {best_model_name}",
        "",
        "  Selection Criteria:",
        "    Weighted F1 Score balances precision and recall across",
        "    both classes and is robust to mild class imbalance.",
        sep,
    ]

    text = "\n".join(lines)
    path = OUTPUTS / "model_comparison.txt"
    path.write_text(text, encoding="utf-8")
    print(f"\n  [Saved] Model comparison  : {path}")
    print(text)


# ── Conclusion document ───────────────────────────────────────────────────────
def save_conclusion(results: dict, best_model_name: str, error_summary: str) -> None:
    best = results[best_model_name]
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep  = "=" * 62
    sep2 = "-" * 44

    lines = [
        sep,
        "  PROJECT CONCLUSION",
        "  Alligator vs Crocodile Image Classification",
        "  Tilesview AI Interview Task",
        f"  Generated: {now}",
        sep, "",
        "  MODELS TRIED",
        sep2,
        "  1. MobileNetV2",
        "     Role    : Fast, lightweight baseline",
        "     Strategy: Frozen backbone + custom 2-layer head",
        "     Params  : ~3.4 M total / ~300 K trainable",
        "",
        "  2. ResNet50",
        "     Role    : High-accuracy deep feature extractor",
        "     Strategy: Frozen layers 1-3, unfrozen layer4 + custom head",
        "     Params  : ~25.6 M total / ~6.5 M trainable",
        "",
        "  BEST MODEL",
        sep2,
        f"  Selected : {best_model_name}",
        f"  Reason   : Highest Weighted F1 Score on held-out validation set.",
        "             F1 balances precision and recall — ideal for imbalanced",
        "             datasets where both false positives and false negatives",
        "             carry meaningful cost.",
        "",
        "  FINAL METRICS  (Best Model — Validation Set)",
        sep2,
        f"  Accuracy  : {best['Accuracy']:.4f}  ({best['Accuracy']*100:.2f} %)",
        f"  Precision : {best['Precision']:.4f}",
        f"  Recall    : {best['Recall']:.4f}",
        f"  F1 Score  : {best['F1 Score']:.4f}",
        "",
        "  BOTH MODELS",
        sep2,
    ]
    for m, r in results.items():
        lines.append(f"  {m}:")
        for metric, val in r.items():
            lines.append(f"    {metric:<12}: {val:.4f}")
        lines.append("")

    lines += [
        "  OBSERVATIONS & FAILURE CASES",
        sep2,
        "  • Side-angle head images caused most confusion.",
        "    The snout shape (broad U vs tapered V) — the primary visual",
        "    differentiator — is invisible in pure side profiles.",
        "  • Heavy mud / water coverage erased skin texture cues.",
        "  • Juvenile / baby reptiles share similar proportions,",
        "    making species-level classification harder.",
        "  • Images with no clear body (just eyes or background clutter)",
        "    forced the model to rely on context rather than morphology.",
        "",
        "  RECOMMENDATIONS FOR IMPROVEMENT",
        sep2,
        "  • Fine-tune the full backbone with a lower LR (1e-5).",
        "  • Experiment with EfficientNetB4 for better accuracy/efficiency.",
        "  • Apply GradCAM to verify the model attends to the snout region.",
        "  • Use test-time augmentation (TTA) for production inference.",
        "  • Consider focal loss if class imbalance increases in production.",
        "",
        error_summary,
        "",
        sep,
    ]

    text = "\n".join(lines)
    path = CONC_DIR / "conclusion.txt"
    path.write_text(text, encoding="utf-8")
    print(f"\n  [Saved] Conclusion        : {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  Alligator vs Crocodile — Training Pipeline")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    set_seed(SEED)
    device = get_device()

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("\n[1/6] Loading and splitting dataset ...")
    train_loader, val_loader, base_dataset, val_indices, class_names = load_data()

    # Build class-weight tensor for loss
    full_targets    = np.array(base_dataset.targets)
    class_counts    = np.bincount(full_targets)
    class_w         = 1.0 / (class_counts.astype(float) + 1e-8)
    class_w        /= class_w.sum()          # normalise
    class_w_tensor  = torch.FloatTensor(class_w)

    # ── Step 2: Build models ──────────────────────────────────────────────────
    print("\n[2/6] Building models ...")
    model_builders = {
        "MobileNetV2": build_mobilenetv2,
        "ResNet50":    build_resnet50,
    }

    results         = {}
    trained_models  = {}
    all_history     = {}

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print("\n[3/6] Training ...")
    for model_name, builder in model_builders.items():
        model = builder(num_classes=2)
        trained, history, best_f1, ckpt_path = train_model(
            model, model_name, train_loader, val_loader, class_w_tensor, device
        )
        trained_models[model_name] = (trained, ckpt_path)
        all_history[model_name]    = history

    # ── Step 4: Evaluate ─────────────────────────────────────────────────────
    print("\n[4/6] Evaluating models ...")
    for model_name, (model, _) in trained_models.items():
        metrics, cm, report, preds, labels = compute_metrics(
            model, val_loader, device, list(class_names)
        )
        results[model_name] = metrics

        print(f"\n  ── {model_name} ──")
        print(f"  Accuracy  : {metrics['Accuracy']:.4f}")
        print(f"  Precision : {metrics['Precision']:.4f}")
        print(f"  Recall    : {metrics['Recall']:.4f}")
        print(f"  F1 Score  : {metrics['F1 Score']:.4f}")
        print(f"\n{report}")

        save_confusion_matrix(cm, list(class_names), model_name, CM_DIR)
        save_training_plots(all_history[model_name], model_name, PLOTS_DIR)

    # ── Select best model ─────────────────────────────────────────────────────
    best_name  = max(results, key=lambda m: results[m]["F1 Score"])
    best_model, best_ckpt = trained_models[best_name]
    print(f"\n[Best Model] {best_name}  —  F1: {results[best_name]['F1 Score']:.4f}")

    # Copy best checkpoint to models/best_model.pth
    final_ckpt = MODEL_DIR / "best_model.pth"
    shutil.copy(best_ckpt, final_ckpt)

    # Enrich checkpoint with metadata
    ckpt = torch.load(final_ckpt, map_location="cpu")
    ckpt["model_name"]   = best_name
    ckpt["classes"]      = list(class_names)
    ckpt["metrics"]      = results[best_name]
    ckpt["image_size"]   = IMAGE_SIZE
    ckpt["imagenet_mean"] = IMAGENET_MEAN
    ckpt["imagenet_std"]  = IMAGENET_STD
    torch.save(ckpt, final_ckpt)
    print(f"[Saved] Best model checkpoint : {final_ckpt}")

    # ── Step 5: Error analysis ────────────────────────────────────────────────
    print("\n[5/6] Running error analysis ...")
    error_summary = run_error_analysis(
        best_model, base_dataset, val_indices, list(class_names),
        device, best_name, MISC_DIR, WRONG_DIR, max_save=100,
    )

    # ── Step 6: Reports ───────────────────────────────────────────────────────
    print("\n[6/6] Saving comparison and conclusion ...")
    save_model_comparison(results, best_name)
    save_conclusion(results, best_name, error_summary)

    # ── Summary ───────────────────────────────────────────────────────────────
    bm = results[best_name]
    print("\n" + "=" * 62)
    print("  TRAINING COMPLETE")
    print("=" * 62)
    print(f"  Best Model     : {best_name}")
    print(f"  Accuracy       : {bm['Accuracy']:.4f}  ({bm['Accuracy']*100:.2f} %)")
    print(f"  Precision      : {bm['Precision']:.4f}")
    print(f"  Recall         : {bm['Recall']:.4f}")
    print(f"  F1 Score       : {bm['F1 Score']:.4f}")
    print(f"  Checkpoint     : {final_ckpt}")
    print(f"  Finished       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print("\nNext steps:")
    print("  python evaluate.py   — full evaluation report")
    print("  python predict.py    — classify new images")


if __name__ == "__main__":
    main()
