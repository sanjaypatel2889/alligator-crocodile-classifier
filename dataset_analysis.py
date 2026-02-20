"""
dataset_analysis.py
====================
Alligator vs Crocodile Classification — Tilesview AI Interview Task

Scans the dataset directory, validates image integrity, reports class
distribution and imbalance, then saves a full report to
outputs/dataset_report.txt.

Run:
    python dataset_analysis.py
"""

import os
import sys
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR  = ROOT / "outputs"
REPORT_PATH = OUTPUT_DIR / "dataset_report.txt"

# Formats PyTorch / PIL can reliably decode as RGB
VALID_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
INVALID_EXTENSIONS = {".gif", ".mp4", ".avi", ".mov"}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _try_open(path: Path) -> str:
    """
    Attempt to open and verify an image with PIL.
    Returns 'ok', 'corrupted', or 'unreadable'.
    """
    try:
        from PIL import Image, UnidentifiedImageError
        with Image.open(path) as img:
            img.verify()          # checks file integrity without decoding
        # Re-open to ensure it can actually be decoded to RGB
        with Image.open(path) as img:
            img.convert("RGB")
        return "ok"
    except Exception:
        return "corrupted"


def _get_image_dimensions(path: Path):
    """Return (width, height) or None on failure."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


# ── Core analysis ────────────────────────────────────────────────────────────
def analyze_dataset(verbose: bool = True) -> dict:
    """
    Scan DATASET_DIR, validate every file, compute statistics.

    Returns
    -------
    dict  mapping class_name -> {total, valid, corrupted, unsupported, unreadable,
                                  widths, heights}
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    class_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    if not class_dirs:
        print("[ERROR] No class sub-folders found inside dataset/")
        sys.exit(1)

    class_stats = {}

    for cls_dir in class_dirs:
        cls_name  = cls_dir.name
        all_files = list(cls_dir.iterdir())

        valid_files  = []
        corrupted    = []
        unsupported  = []
        unreadable   = []
        widths, heights = [], []

        print(f"[Scanning] {cls_name} ({len(all_files)} files) ...")

        for f in all_files:
            if not f.is_file():
                continue

            ext = f.suffix.lower()

            if ext in INVALID_EXTENSIONS:
                unsupported.append(f.name)
                continue

            if ext not in VALID_EXTENSIONS:
                unreadable.append(f.name)
                continue

            status = _try_open(f)
            if status == "ok":
                valid_files.append(f)
                dims = _get_image_dimensions(f)
                if dims:
                    widths.append(dims[0])
                    heights.append(dims[1])
            else:
                corrupted.append(f.name)

        class_stats[cls_name] = {
            "total":       len(all_files),
            "valid":       len(valid_files),
            "corrupted":   corrupted,
            "unsupported": unsupported,
            "unreadable":  unreadable,
            "widths":      widths,
            "heights":     heights,
        }

    # ── Build report ─────────────────────────────────────────────────────────
    lines = []
    sep60 = "=" * 60
    sep40 = "-" * 40

    lines += [sep60, "  DATASET ANALYSIS REPORT", "  Alligator vs Crocodile — Tilesview AI", sep60, ""]

    # Per-class summary
    for cls_name, s in class_stats.items():
        lines += [
            f"  Class : {cls_name.upper()}",
            f"  {sep40}",
            f"  Total files        : {s['total']}",
            f"  Valid images       : {s['valid']}",
            f"  Corrupted          : {len(s['corrupted'])}",
            f"  Unsupported format : {len(s['unsupported'])}",
            f"  Unreadable / other : {len(s['unreadable'])}",
        ]
        if s["widths"]:
            import statistics
            lines += [
                f"  Avg resolution     : {statistics.mean(s['widths']):.0f} x "
                f"{statistics.mean(s['heights']):.0f} px",
                f"  Min resolution     : {min(s['widths'])} x {min(s['heights'])} px",
                f"  Max resolution     : {max(s['widths'])} x {max(s['heights'])} px",
            ]
        if s["corrupted"]:
            lines.append(f"  [CORRUPTED FILES]  : {', '.join(s['corrupted'][:10])}")
        if s["unsupported"]:
            lines.append(f"  [UNSUPPORTED]      : {', '.join(s['unsupported'][:10])}")
        if s["unreadable"]:
            lines.append(f"  [UNREADABLE]       : {', '.join(s['unreadable'][:10])}")
        lines.append("")

    # Class distribution
    valid_counts = {cls: s["valid"] for cls, s in class_stats.items()}
    total_valid  = sum(valid_counts.values())

    lines += [sep60, "  CLASS DISTRIBUTION", sep60]
    for cls, cnt in valid_counts.items():
        pct = 100.0 * cnt / total_valid if total_valid else 0
        bar = "#" * int(pct / 2)
        lines.append(f"  {cls:20s}: {cnt:5d}  ({pct:5.1f}%)  {bar}")
    lines += [f"  {'TOTAL':20s}: {total_valid:5d}", ""]

    # Imbalance analysis
    counts = list(valid_counts.values())
    ratio  = max(counts) / min(counts) if min(counts) > 0 else float("inf")
    lines += [sep60, "  IMBALANCE ANALYSIS", sep60]
    lines.append(f"  Imbalance ratio      : {ratio:.3f}:1")

    if ratio > 2.0:
        level = "SEVERE"
        advice = "Apply class-weighted loss AND oversample minority class."
    elif ratio > 1.5:
        level = "MODERATE"
        advice = "Apply class-weighted CrossEntropyLoss."
    elif ratio > 1.1:
        level = "MILD"
        advice = "Weighted loss applied as precaution."
    else:
        level = "NONE"
        advice = "Dataset is well balanced — standard training applies."

    lines += [
        f"  Imbalance level      : {level}",
        f"  Recommendation       : {advice}",
        "",
    ]

    # Recommendations
    lines += [
        sep60,
        "  RECOMMENDATIONS",
        sep60,
        "  1. Resize all images to 224x224 before feeding the model.",
        "  2. Apply ImageNet mean/std normalisation.",
        "  3. Use RandomHorizontalFlip, RandomRotation, ColorJitter augmentations.",
        "  4. Apply weighted CrossEntropyLoss to counter class imbalance.",
        "  5. Corrupted / unsupported files are silently skipped by DataLoader.",
        "",
        sep60,
        "  EXTENSION BREAKDOWN",
        sep60,
    ]

    ext_counts = defaultdict(int)
    for cls_dir in class_dirs:
        for f in cls_dir.iterdir():
            if f.is_file():
                ext_counts[f.suffix.lower()] += 1
    for ext, cnt in sorted(ext_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {ext:10s}: {cnt}")

    lines += ["", sep60]

    report_text = "\n".join(lines)

    REPORT_PATH.write_text(report_text, encoding="utf-8")

    if verbose:
        print(report_text)
        print(f"\n[Saved] Report -> {REPORT_PATH}")

    return class_stats


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    stats = analyze_dataset(verbose=True)
    valid_total = sum(s["valid"] for s in stats.values())
    print(f"\n[Summary] {valid_total} valid images across {len(stats)} classes.")
    print("[Done] Dataset analysis complete.\n")
