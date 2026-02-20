"""
predict.py
===========
Alligator vs Crocodile Classification — Tilesview AI Interview Task

Loads the best trained model and classifies images.

Usage
-----
# Predict ALL images in a folder  (results saved to predictions/)
    python predict.py --input path/to/folder

# Predict a single image
    python predict.py --input path/to/image.jpg

# Default: predict on a random sample of validation images
    python predict.py

Outputs
-------
predictions/<filename>_pred.jpg   — copy with OpenCV overlay:
  "Predicted: Crocodile | Confidence: 92%"
predictions/prediction_summary.txt — text summary of all predictions
"""

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torchvision import transforms, models
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
MODEL_DIR   = ROOT / "models"
OUTPUTS     = ROOT / "outputs"
PRED_DIR    = ROOT / "predictions"
VAL_IDX_FILE = OUTPUTS / "val_indices.json"
BEST_CKPT    = MODEL_DIR / "best_model.pth"

IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VALID_EXT     = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

PRED_DIR.mkdir(parents=True, exist_ok=True)


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device):
    """Load best checkpoint, rebuild the model architecture, restore weights."""
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
    else:                                        # ResNet50 (default)
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

    print(f"[Model] {model_name}  loaded  ({device})")
    print(f"[Model] Classes : {class_names}")

    # Log stored training metrics if available
    if "metrics" in ckpt:
        m = ckpt["metrics"]
        print(
            f"[Model] Training metrics — "
            f"Acc: {m.get('Accuracy', 0):.4f}  "
            f"F1: {m.get('F1 Score', 0):.4f}"
        )

    return net, class_names, model_name


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_image(model, img_path: str | Path, device: torch.device,
                  class_names: list[str]) -> dict:
    """
    Run a single image through the model.

    Returns
    -------
    dict with keys: path, label, confidence, all_probs
    """
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    try:
        img = Image.open(str(img_path)).convert("RGB")
    except Exception as e:
        return {"path": str(img_path), "label": "ERROR", "confidence": 0.0,
                "error": str(e), "all_probs": {}}

    tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx   = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    all_probs  = {class_names[i]: float(p) for i, p in enumerate(probs)}

    return {
        "path":       str(img_path),
        "label":      pred_label,
        "confidence": confidence,
        "all_probs":  all_probs,
    }


# ── OpenCV overlay ────────────────────────────────────────────────────────────
def annotate_and_save(img_path: str | Path, result: dict, save_dir: Path) -> Path:
    """
    Load image, draw prediction overlay with OpenCV, save to save_dir.

    Overlay format:
      "Predicted: Crocodile | Confidence: 92%"
    """
    try:
        img_pil = Image.open(str(img_path)).convert("RGB")
        img_cv  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        img_cv = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    # Resize for consistent display (keep aspect ratio, pad to square)
    h, w = img_cv.shape[:2]
    scale = 400 / max(h, w)
    img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    h, w = img_cv.shape[:2]

    label      = result.get("label", "unknown")
    confidence = result.get("confidence", 0.0)
    all_probs  = result.get("all_probs", {})

    # ── Colour: green = alligator, orange = crocodile ─────────────────────────
    colour = (0, 180, 0) if label.lower() == "alligator" else (0, 140, 255)

    # ── Primary overlay bar (top) ─────────────────────────────────────────────
    bar_h = 38
    overlay = img_cv.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img_cv, 0.25, 0, img_cv)

    primary_text = f"Predicted: {label.capitalize()}  |  Confidence: {confidence:.0%}"
    cv2.putText(
        img_cv, primary_text,
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        colour, 2, cv2.LINE_AA,
    )

    # ── Per-class probability bar (bottom) ───────────────────────────────────
    if all_probs:
        bar_y_start = h - 44
        cv2.rectangle(img_cv, (0, bar_y_start), (w, h), (20, 20, 20), -1)

        x_offset = 8
        for cls, prob in sorted(all_probs.items()):
            c = (0, 180, 0) if cls.lower() == "alligator" else (0, 140, 255)
            text = f"{cls.capitalize()}: {prob:.0%}"
            cv2.putText(img_cv, text, (x_offset, bar_y_start + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
            x_offset += 220

    # ── Border colour coded by prediction ────────────────────────────────────
    cv2.rectangle(img_cv, (0, 0), (w - 1, h - 1), colour, 3)

    # ── Save ──────────────────────────────────────────────────────────────────
    stem  = Path(img_path).stem
    fname = f"{stem}_pred.jpg"
    out   = save_dir / fname
    cv2.imwrite(str(out), img_cv, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out


# ── Collect input images ──────────────────────────────────────────────────────
def collect_images(input_path: str | None, base_dataset, val_indices: list[int],
                   sample_n: int = 40) -> list[Path]:
    """
    Priority:
      1. --input folder -> all images in that folder
      2. --input file   -> that single file
      3. No input       -> random sample_n images from validation set
    """
    if input_path:
        p = Path(input_path)
        if p.is_dir():
            imgs = [f for f in sorted(p.iterdir())
                    if f.is_file() and f.suffix.lower() in VALID_EXT]
            print(f"[Input] Folder mode  — {len(imgs)} images found in {p}")
            return imgs
        elif p.is_file():
            print(f"[Input] Single file  — {p}")
            return [p]
        else:
            print(f"[ERROR] Path not found: {p}")
            sys.exit(1)

    # Default: sample from validation set
    if not val_indices:
        print("[WARN ] No validation indices found — sampling from full dataset.")
        all_paths = [Path(s[0]) for s in base_dataset.samples]
        sample_n  = min(sample_n, len(all_paths))
        return random.sample(all_paths, sample_n)

    indices   = random.sample(val_indices, min(sample_n, len(val_indices)))
    img_paths = [Path(base_dataset.samples[i][0]) for i in indices]
    print(f"[Input] Default mode — {len(img_paths)} random validation images")
    return img_paths


# ── Simple dataset for default mode ──────────────────────────────────────────
class MinimalDataset:
    """Lightweight class: just loads samples list — no full Dataset overhead."""

    def __init__(self, root: Path):
        self.samples: list[tuple[str, int]] = []
        classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls in classes:
            cls_dir = root / cls
            for fp in sorted(cls_dir.iterdir()):
                if fp.suffix.lower() in VALID_EXT:
                    self.samples.append((str(fp), class_to_idx[cls]))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Predict alligator vs crocodile with the trained model."
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to an image file or folder. Defaults to val-set sample.",
    )
    parser.add_argument(
        "--sample", "-n", type=int, default=40,
        help="Number of validation images to predict in default mode (default: 40).",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=str(PRED_DIR),
        help="Output folder for annotated images (default: predictions/).",
    )
    args = parser.parse_args()

    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}\n")

    # Load model
    model, class_names, model_name = load_model(BEST_CKPT, device)

    # Load val indices (optional — used in default mode)
    val_indices = []
    if VAL_IDX_FILE.exists():
        with open(VAL_IDX_FILE) as fh:
            val_indices = json.load(fh)["indices"]

    # Load minimal dataset
    base_dataset = MinimalDataset(DATASET_DIR)

    # Collect images to predict
    img_paths = collect_images(args.input, base_dataset, val_indices, args.sample)
    if not img_paths:
        print("[ERROR] No images to predict.")
        sys.exit(1)

    print(f"\n[Predict] Running inference on {len(img_paths)} images ...\n")

    summary_rows = []
    correct = 0
    total   = 0

    for i, img_path in enumerate(img_paths, 1):
        result = predict_image(model, img_path, device, class_names)

        if "error" in result:
            print(f"  [{i:03d}] SKIP  {img_path.name}  ({result['error']})")
            continue

        out_path = annotate_and_save(img_path, result, save_dir)

        # Check ground truth from folder name if available
        gt_label = img_path.parent.name.lower()
        is_correct = (gt_label == result["label"].lower()) if gt_label in class_names else None
        correct_str = ""
        if is_correct is not None:
            correct_str = "  OK" if is_correct else "  X"
            total += 1
            if is_correct:
                correct += 1

        probs_str = "  ".join(
            f"{c.capitalize()}: {p:.0%}"
            for c, p in sorted(result["all_probs"].items())
        )
        print(
            f"  [{i:03d}] {img_path.name:<45s}  "
            f"-> {result['label'].capitalize():<12s}  "
            f"({result['confidence']:.0%}){correct_str}"
        )
        print(f"           Probs: {probs_str}  |  Saved: {out_path.name}")

        summary_rows.append(
            f"{img_path.name}  |  {result['label'].capitalize()}  |  "
            f"{result['confidence']:.4f}  |  "
            f"{'GT:'+gt_label if gt_label in class_names else 'GT:unknown'}"
        )

    # ── Accuracy (only when GT is known) ─────────────────────────────────────
    if total > 0:
        acc = correct / total
        print(f"\n  Ground-truth accuracy on {total} labelled images: {acc:.2%}")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = save_dir / "prediction_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"Model : {model_name}\n")
        fh.write(f"Images: {len(summary_rows)}\n")
        if total > 0:
            fh.write(f"GT Accuracy: {correct}/{total} = {correct/total:.2%}\n")
        fh.write("\n")
        fh.write("Filename  |  Prediction  |  Confidence  |  Ground Truth\n")
        fh.write("-" * 70 + "\n")
        fh.write("\n".join(summary_rows))

    print(f"\n[Saved] {len(summary_rows)} annotated images -> {save_dir}")
    print(f"[Saved] Summary              -> {summary_path}")
    print("\n[Done] Prediction complete.")


if __name__ == "__main__":
    main()
