"""
app.py
=======
Alligator vs Crocodile Classification — Tilesview AI Interview Task
Streamlit Dashboard

Launch:
    streamlit run app.py
"""

import io
import os
import warnings
from pathlib import Path

# Fix: OMP duplicate library error when Anaconda + PyTorch coexist
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Tilesview AI — Alligator vs Crocodile",
    page_icon="🐊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
DATASET_DIR  = ROOT / "dataset"
MODEL_DIR    = ROOT / "models"
OUTPUTS      = ROOT / "outputs"
BEST_CKPT    = MODEL_DIR / "best_model.pth"
CM_DIR       = OUTPUTS / "confusion_matrix"
PLOTS_DIR    = OUTPUTS / "training_plots"
WRONG_DIR    = OUTPUTS / "wrong_predictions"
MISC_DIR     = OUTPUTS / "misclassified"
PRED_DIR     = ROOT / "predictions"
CONC_FILE    = ROOT / "conclusion" / "conclusion.txt"
COMPARE_FILE = OUTPUTS / "model_comparison.txt"
REPORT_FILE  = OUTPUTS / "dataset_report.txt"
EVAL_FILE    = OUTPUTS / "evaluation_report.txt"

IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VALID_EXT     = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar title */
    .sidebar-title { font-size: 1.3rem; font-weight: 700; color: #1a7f3c; }
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #f8f9fa; border-radius: 10px;
        border: 1px solid #dee2e6; padding: 8px 12px;
    }
    /* Confidence bars */
    .conf-bar-wrap { margin: 4px 0; }
    /* Status badge */
    .badge-ok   { background:#d4edda; color:#155724; border-radius:6px; padding:3px 10px; font-weight:600; }
    .badge-warn { background:#fff3cd; color:#856404; border-radius:6px; padding:3px 10px; font-weight:600; }
    /* Section headers */
    h3 { color: #1a7f3c; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def is_trained() -> bool:
    return BEST_CKPT.exists()


def dataset_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    if DATASET_DIR.exists():
        for cls_dir in sorted(DATASET_DIR.iterdir()):
            if cls_dir.is_dir():
                counts[cls_dir.name] = sum(
                    1 for f in cls_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in VALID_EXT
                )
    return counts


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights …")
def load_model():
    """Rebuild and load the best model checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not BEST_CKPT.exists():
        return None, None, None, None, device

    ckpt        = torch.load(BEST_CKPT, map_location=device)
    model_name  = ckpt.get("model_name", "ResNet50")
    class_names = ckpt.get("classes",    ["alligator", "crocodile"])
    metrics     = ckpt.get("metrics",    {})
    num_classes = len(class_names)

    if "MobileNetV2" in model_name:
        net = models.mobilenet_v2(weights=None)
        net.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(net.last_channel, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    else:
        net = models.resnet50(weights=None)
        in_feat = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    net.load_state_dict(ckpt["model_state"])
    net.eval()
    net.to(device)
    return net, class_names, model_name, metrics, device


# ── Inference helper ──────────────────────────────────────────────────────────
def run_inference(model, pil_image: Image.Image, class_names: list[str],
                  device: torch.device) -> dict:
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = tf(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred  = int(np.argmax(probs))
    return {
        "label":      class_names[pred],
        "confidence": float(probs[pred]),
        "all_probs":  {class_names[i]: float(p) for i, p in enumerate(probs)},
    }


def annotate_pil(pil_image: Image.Image, result: dict) -> Image.Image:
    """Add OpenCV prediction overlay and return as PIL."""
    img_cv = cv2.cvtColor(np.array(pil_image.convert("RGB").resize((400, 400))),
                          cv2.COLOR_RGB2BGR)
    h, w   = img_cv.shape[:2]
    label  = result["label"]
    conf   = result["confidence"]
    colour = (0, 180, 0) if label.lower() == "alligator" else (0, 140, 255)

    # Top bar
    cv2.rectangle(img_cv, (0, 0), (w, 38), (20, 20, 20), -1)
    cv2.putText(img_cv,
                f"Predicted: {label.capitalize()}  |  Confidence: {conf:.0%}",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, colour, 2, cv2.LINE_AA)

    # Per-class bottom bar
    cv2.rectangle(img_cv, (0, h - 42), (w, h), (20, 20, 20), -1)
    x = 8
    for cls, prob in sorted(result["all_probs"].items()):
        c = (0, 180, 0) if cls.lower() == "alligator" else (0, 140, 255)
        cv2.putText(img_cv, f"{cls.capitalize()}: {prob:.0%}",
                    (x, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 1, cv2.LINE_AA)
        x += 210

    cv2.rectangle(img_cv, (0, 0), (w - 1, h - 1), colour, 3)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🐊 Tilesview AI</div>', unsafe_allow_html=True)
    st.caption("Alligator vs Crocodile — Image Classification")
    st.divider()

    # Training status
    if is_trained():
        st.markdown('<span class="badge-ok">✅ Model Trained</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">⚠️ Not Trained Yet</span>', unsafe_allow_html=True)
        st.caption("Run `python train.py` to train both models.")

    # Device
    device_str = "🖥️ CUDA GPU" if torch.cuda.is_available() else "💻 CPU"
    st.caption(f"Device: {device_str}")

    # Dataset counts
    counts = dataset_counts()
    if counts:
        st.divider()
        st.markdown("**Dataset**")
        for cls, cnt in counts.items():
            st.caption(f"  {cls.capitalize()}: {cnt:,} images")
        st.caption(f"  Total: {sum(counts.values()):,} images")

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "📊 Dataset Analysis",
            "🤖 Live Prediction",
            "📈 Training Results",
            "🔍 Error Analysis",
            "📋 Conclusion",
        ],
        label_visibility="collapsed",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🐊 Alligator vs Crocodile Classifier")
    st.subheader("Tilesview AI — Interview Task")
    st.markdown(
        """
        A professional deep learning pipeline that classifies images as
        **Alligator** or **Crocodile** using two pretrained CNN backbones
        compared side-by-side.
        """
    )

    # Dataset cards
    st.markdown("### Dataset Overview")
    counts = dataset_counts()
    total  = sum(counts.values())
    cols   = st.columns(len(counts) + 1)
    for i, (cls, cnt) in enumerate(counts.items()):
        pct = 100 * cnt / total if total else 0
        cols[i].metric(cls.capitalize(), f"{cnt:,}", f"{pct:.1f}% of total")
    cols[-1].metric("Total", f"{total:,}", "images")

    # Model metrics (if trained)
    st.markdown("### Model Performance")
    if is_trained():
        _, class_names, model_name, metrics, device = load_model()
        if metrics:
            st.markdown(f"**Best Model: `{model_name}`** *(selected by highest Weighted F1)*")
            m_cols = st.columns(4)
            for col, (metric, val) in zip(m_cols, metrics.items()):
                col.metric(metric, f"{val:.4f}", f"{val*100:.2f}%")
    else:
        st.info("Train the models first with `python train.py` to see performance metrics here.")

    # Pipeline overview
    st.markdown("### Pipeline Overview")
    st.markdown(
        """
        | Step | Command | Description |
        |------|---------|-------------|
        | 0 | `python dataset_analysis.py` | Validate dataset, detect corrupted files |
        | 1 | `python train.py`            | Train MobileNetV2 + ResNet50, compare, save best |
        | 2 | `python evaluate.py`         | Full evaluation on held-out validation set |
        | 3 | `python predict.py`          | CLI inference with OpenCV overlays |
        | 4 | `streamlit run app.py`       | This interactive dashboard |

        **80/20 train/val split** · **Fixed seed 42** · **Weighted loss for imbalance**
        """
    )

    # Architecture cards
    st.markdown("### Models Compared")
    a_col, b_col = st.columns(2)
    with a_col:
        st.markdown("#### MobileNetV2")
        st.markdown(
            """
            - Role: Fast, lightweight baseline
            - Backbone: Frozen (ImageNet weights)
            - Head: `Dropout → Linear(1280,256) → ReLU → Linear(256,2)`
            - Trainable params: ~300 K
            """
        )
    with b_col:
        st.markdown("#### ResNet50")
        st.markdown(
            """
            - Role: High-accuracy deep extractor
            - Backbone: Layer4 unfrozen for fine-tuning
            - Head: `Dropout → Linear(2048,512) → ReLU → Linear(512,2)`
            - Trainable params: ~6.5 M
            """
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Analysis":
    st.title("📊 Dataset Analysis")

    counts = dataset_counts()
    total  = sum(counts.values())

    # ── Count cards ──────────────────────────────────────────────────────────
    st.markdown("### Class Distribution")
    cols = st.columns(len(counts) + 1)
    for i, (cls, cnt) in enumerate(counts.items()):
        pct = 100 * cnt / total if total else 0
        cols[i].metric(cls.capitalize(), f"{cnt:,}", f"{pct:.1f}%")
    cols[-1].metric("Total", f"{total:,}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(
        [c.capitalize() for c in counts.keys()],
        list(counts.values()),
        color=["#2ecc71", "#e67e22"],
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{val:,}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.set_title("Image Count per Class", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(counts.values()) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=False)

    # ── Imbalance info ────────────────────────────────────────────────────────
    st.markdown("### Imbalance Analysis")
    if len(counts) >= 2:
        ratio = max(counts.values()) / min(counts.values())
        if   ratio > 2.0: level, colour = "SEVERE",   "🔴"
        elif ratio > 1.5: level, colour = "MODERATE", "🟠"
        elif ratio > 1.1: level, colour = "MILD",     "🟡"
        else:             level, colour = "BALANCED",  "🟢"

        st.markdown(
            f"{colour} **Imbalance Ratio:** `{ratio:.3f} : 1`  |  "
            f"**Level:** `{level}`"
        )
        st.markdown(
            "**Mitigation applied:** `WeightedRandomSampler` during training + "
            "`CrossEntropyLoss(weight=...)` with inverse class frequencies."
        )

    # ── Extension breakdown ───────────────────────────────────────────────────
    st.markdown("### File Format Breakdown")
    from collections import defaultdict
    ext_counts: dict[str, int] = defaultdict(int)
    if DATASET_DIR.exists():
        for cls_dir in DATASET_DIR.iterdir():
            if cls_dir.is_dir():
                for f in cls_dir.iterdir():
                    if f.is_file():
                        ext_counts[f.suffix.lower()] += 1

    if ext_counts:
        import pandas as pd
        df_ext = pd.DataFrame(
            [{"Extension": k, "Count": v, "Supported": k in VALID_EXT}
             for k, v in sorted(ext_counts.items(), key=lambda x: -x[1])]
        )
        st.dataframe(df_ext, use_container_width=True, hide_index=True)

    # ── Dataset report text ───────────────────────────────────────────────────
    if REPORT_FILE.exists():
        with st.expander("📄 Full Dataset Report (outputs/dataset_report.txt)"):
            st.text(REPORT_FILE.read_text(encoding="utf-8"))
    else:
        st.info("Run `python dataset_analysis.py` to generate a full dataset report.")

    # ── Augmentations ─────────────────────────────────────────────────────────
    st.markdown("### Augmentation Strategy (Training Only)")
    st.code(
        """transforms.Compose([
    Resize(256 × 256),
    RandomResizedCrop(224, scale=(0.70, 1.0)),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15),
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])""",
        language="python",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Live Prediction":
    st.title("🤖 Live Prediction")
    st.markdown(
        "**Step 1:** Upload your image below  →  "
        "**Step 2:** Click **Classify Image** to get the answer."
    )

    if not is_trained():
        st.warning("⚠️ No trained model found. Run `python train.py` first, then refresh.")
        st.stop()

    model, class_names, model_name, metrics, device = load_model()
    if model is None:
        st.error("Failed to load model. Check `models/best_model.pth`.")
        st.stop()

    # ── Model info strip ──────────────────────────────────────────────────────
    with st.expander("Model info", expanded=False):
        st.caption(f"Loaded: **{model_name}** · Device: `{device}`")
        if metrics:
            m_cols = st.columns(4)
            for col, (k, v) in zip(m_cols, metrics.items()):
                col.metric(k, f"{v:.4f}")

    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📁 Upload an image (jpg / jpeg / png / webp / bmp)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=False,
        key="pred_uploader",
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")

        # Show uploaded image centred
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.image(pil_img, caption=f"Uploaded: {uploaded_file.name}",
                     use_container_width=True)

        st.markdown("")

        # ── Classify button ───────────────────────────────────────────────────
        btn_col, _ = st.columns([1, 3])
        clicked = btn_col.button(
            "🔍 Classify Image",
            type="primary",
            use_container_width=True,
        )

        if clicked:
            with st.spinner("Running model inference ..."):
                result = run_inference(model, pil_img, class_names, device)

            label      = result["label"]
            conf       = result["confidence"]
            probs      = result["all_probs"]
            annotated  = annotate_pil(pil_img, result)

            st.markdown("---")
            st.markdown("### Result")

            # ── Big answer card ───────────────────────────────────────────────
            is_alligator = label.lower() == "alligator"
            colour_hex   = "#27ae60" if is_alligator else "#e67e22"
            emoji        = "🐊" if is_alligator else "🦎"

            st.markdown(
                f"""
                <div style="
                    background:{colour_hex}18;
                    border: 3px solid {colour_hex};
                    border-radius: 16px;
                    padding: 24px 32px;
                    margin: 12px 0 20px 0;
                    text-align: center;
                ">
                    <div style="font-size:4rem; margin-bottom:8px;">{emoji}</div>
                    <div style="font-size:2.4rem; font-weight:800; color:{colour_hex};
                                letter-spacing:2px; text-transform:uppercase;">
                        {label}
                    </div>
                    <div style="font-size:1.2rem; color:#555; margin-top:8px;">
                        Confidence : <strong style="color:{colour_hex};">{conf:.1%}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Side-by-side: original + annotated ───────────────────────────
            left, right = st.columns(2)
            with left:
                st.markdown("**Your Image**")
                st.image(pil_img, use_container_width=True)
            with right:
                st.markdown("**With Prediction Overlay**")
                st.image(annotated, use_container_width=True)

            # ── Probability bars ──────────────────────────────────────────────
            st.markdown("#### Class Probabilities")
            for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
                c1, c2, c3 = st.columns([2, 6, 2])
                c1.markdown(f"**{cls.capitalize()}**")
                c2.progress(prob)
                c3.markdown(f"**{prob:.1%}**")

            # ── Save annotated image to predictions/ ──────────────────────────
            PRED_DIR.mkdir(parents=True, exist_ok=True)
            out_name = f"{Path(uploaded_file.name).stem}_pred.jpg"
            out_path = PRED_DIR / out_name
            annotated.save(str(out_path), quality=95)
            st.success(f"Annotated image saved to `predictions/{out_name}`")

    else:
        # Placeholder instructions
        st.markdown(
            """
            <div style="
                border: 2px dashed #ccc; border-radius:12px;
                padding: 40px; text-align:center; color:#888;
            ">
                <div style="font-size:3rem;">🖼️</div>
                <div style="font-size:1.1rem; margin-top:8px;">
                    No image uploaded yet.<br>
                    Use the file picker above to upload a reptile image.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAINING RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Training Results":
    st.title("📈 Training Results")

    if not is_trained():
        st.warning("⚠️ No training artefacts found. Run `python train.py` first.")
        st.stop()

    # ── Hyperparameters ───────────────────────────────────────────────────────
    st.markdown("### Training Configuration")
    h_cols = st.columns(5)
    h_cols[0].metric("Epochs",       "12")
    h_cols[1].metric("Batch Size",   "32")
    h_cols[2].metric("Learning Rate","1e-3")
    h_cols[3].metric("Image Size",   "224 × 224")
    h_cols[4].metric("Train Split",  "80 %")

    # ── Training curves ───────────────────────────────────────────────────────
    st.markdown("### Training Curves")
    curve_files = sorted(PLOTS_DIR.glob("training_curves_*.png"))
    if curve_files:
        for f in curve_files:
            model_label = f.stem.replace("training_curves_", "").replace("_", " ").title()
            st.markdown(f"**{model_label}**")
            st.image(str(f), use_container_width=True)
    else:
        st.info("Training curve plots not found. Run `python train.py`.")

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.markdown("### Confusion Matrices")
    cm_files = sorted(CM_DIR.glob("confusion_matrix_*.png"))
    if cm_files:
        if len(cm_files) >= 2:
            c1, c2 = st.columns(2)
            cols_pair = [c1, c2]
            for col, f in zip(cols_pair, cm_files[:2]):
                label = f.stem.replace("confusion_matrix_", "").replace("_", " ").title()
                col.markdown(f"**{label}**")
                col.image(str(f), use_container_width=True)
            for f in cm_files[2:]:
                st.image(str(f), use_container_width=True)
        else:
            st.image(str(cm_files[0]), use_container_width=True)
    else:
        st.info("Confusion matrix plots not found. Run `python train.py`.")

    # ── Eval confusion matrices (from evaluate.py) ────────────────────────────
    eval_cm_files = sorted(CM_DIR.glob("eval_confusion_matrix_*.png"))
    if eval_cm_files:
        st.markdown("### Evaluation Confusion Matrices *(from evaluate.py)*")
        for f in eval_cm_files:
            label = f.stem.replace("eval_confusion_matrix_", "").replace("_", " ").title()
            st.markdown(f"**{label}**")
            st.image(str(f), use_container_width=True)

    # ── Confidence histograms ─────────────────────────────────────────────────
    conf_files = sorted(CM_DIR.glob("eval_confidence_*.png"))
    if conf_files:
        st.markdown("### Prediction Confidence Distribution")
        for f in conf_files:
            st.image(str(f), use_container_width=True)

    # ── Model comparison table ────────────────────────────────────────────────
    st.markdown("### Model Comparison")
    if COMPARE_FILE.exists():
        with st.expander("📄 Full comparison report", expanded=True):
            st.text(COMPARE_FILE.read_text(encoding="utf-8"))

    # ── Evaluation report ─────────────────────────────────────────────────────
    if EVAL_FILE.exists():
        with st.expander("📄 Evaluation report (from evaluate.py)"):
            st.text(EVAL_FILE.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Error Analysis":
    st.title("🔍 Error Analysis")
    st.markdown(
        "Images from the validation set that the best model **classified incorrectly**. "
        "Overlays show: `True label → Predicted label | Confidence`"
    )

    wrong_files = sorted(WRONG_DIR.glob("*.jpg")) if WRONG_DIR.exists() else []
    misc_files  = sorted(MISC_DIR.glob("*.jpg"))  if MISC_DIR.exists()  else []

    if not wrong_files and not misc_files:
        st.info(
            "No error analysis images found. Run `python train.py` to generate them "
            "(saved to `outputs/wrong_predictions/`)."
        )
    else:
        files = wrong_files if wrong_files else misc_files
        label_source = "wrong_predictions" if wrong_files else "misclassified"

        # Parse filenames for confusion pair stats
        from collections import defaultdict
        pair_counts: dict[str, int] = defaultdict(int)
        for f in files:
            parts = f.stem.split("_")
            # filename: true_{true_name}_pred_{pred_name}_{idx}
            try:
                true_idx = parts.index("true") + 1
                pred_idx = parts.index("pred") + 1
                true_cls = parts[true_idx]
                pred_cls = parts[pred_idx]
                pair_counts[f"{true_cls} → {pred_cls}"] += 1
            except (ValueError, IndexError):
                pass

        # Summary cards
        st.markdown(f"**Total misclassified samples shown:** {len(files)} (from `{label_source}/`)")
        if pair_counts:
            st.markdown("**Confusion pairs:**")
            p_cols = st.columns(min(len(pair_counts), 3))
            for col, (pair, cnt) in zip(p_cols, pair_counts.items()):
                col.metric(pair, cnt)

        st.divider()

        # Filter controls
        filter_opts = ["All"] + [f.stem.split("_")[1] if "_" in f.stem else "?" for f in files[:1]]
        all_trues   = set()
        for f in files:
            parts = f.stem.split("_")
            try:
                all_trues.add(parts[parts.index("true") + 1])
            except (ValueError, IndexError):
                pass

        filter_cls = st.selectbox(
            "Filter by true class",
            ["All"] + sorted(all_trues),
        )

        n_per_row = st.slider("Images per row", 2, 5, 3)
        max_show  = st.slider("Max images to display", 10, min(100, len(files)), 30)

        filtered = []
        for f in files:
            if filter_cls == "All":
                filtered.append(f)
            else:
                if f"true_{filter_cls}" in f.stem:
                    filtered.append(f)
            if len(filtered) >= max_show:
                break

        # Grid display
        for row_start in range(0, len(filtered), n_per_row):
            row_files = filtered[row_start: row_start + n_per_row]
            row_cols  = st.columns(n_per_row)
            for col, f in zip(row_cols, row_files):
                try:
                    img = Image.open(f)
                    # Parse caption from filename
                    parts = f.stem.split("_")
                    try:
                        t = parts[parts.index("true") + 1].capitalize()
                        p = parts[parts.index("pred") + 1].capitalize()
                        caption = f"True: {t} | Pred: {p}"
                    except (ValueError, IndexError):
                        caption = f.name
                    col.image(img, caption=caption, use_container_width=True)
                except Exception:
                    col.warning(f"Cannot open {f.name}")

        st.markdown("### Common Confusion Reasons")
        st.markdown(
            """
            | Reason | Explanation |
            |--------|-------------|
            | Side-angle shots | Snout geometry (broad U vs tapered V) is the primary differentiator — invisible in side profiles |
            | Mud/water coverage | Erases skin texture cues that distinguish the species |
            | Juvenile specimens | Similar proportions and colouring across species |
            | Background-dominated images | No clear body features visible — model relies on environmental context |
            """
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Conclusion":
    st.title("📋 Conclusion")

    # ── Metrics comparison chart ──────────────────────────────────────────────
    if is_trained():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        # Load individual model checkpoints for metrics
        model_metrics: dict[str, dict] = {}
        for ckpt_file in [MODEL_DIR / "best_mobilenetv2.pth",
                          MODEL_DIR / "best_resnet50.pth"]:
            if ckpt_file.exists():
                c = torch.load(ckpt_file, map_location="cpu")
                # model_name from filename fallback
                mname = c.get("model_name", ckpt_file.stem.replace("best_", "").replace("_", " ").title())
                # metrics may not be stored in individual checkpoints
                # best_model.pth has the best metrics; show what we have
                model_metrics[mname] = {"Best F1": c.get("best_f1", 0.0)}

        # Also load from best_model.pth for full metrics
        if BEST_CKPT.exists():
            c = torch.load(BEST_CKPT, map_location="cpu")
            bname = c.get("model_name", "Best Model")
            m     = c.get("metrics", {})
            if m:
                st.markdown("### Best Model — Final Metrics")
                mc = st.columns(4)
                for col, (metric, val) in zip(mc, m.items()):
                    col.metric(metric, f"{val:.4f}", f"{val*100:.2f}%")

        # Grouped bar chart
        if COMPARE_FILE.exists():
            st.markdown("### Model Comparison Chart")
            lines = COMPARE_FILE.read_text(encoding="utf-8").splitlines()
            # Parse metrics from the comparison file
            parsed: dict[str, dict] = {"MobileNetV2": {}, "ResNet50": {}}
            for line in lines:
                for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
                    if metric in line and "|" in line:
                        parts = [p.strip() for p in line.split("|")]
                        vals  = []
                        for p in parts[1:]:
                            try:
                                vals.append(float(p.replace("◄", "").strip()))
                            except ValueError:
                                pass
                        if len(vals) >= 2:
                            parsed["MobileNetV2"][metric] = vals[0]
                            parsed["ResNet50"][metric]    = vals[1]

            if any(parsed["MobileNetV2"]) and any(parsed["ResNet50"]):
                metrics_list = list(list(parsed.values())[0].keys())
                x  = np.arange(len(metrics_list))
                w  = 0.35
                fig, ax = plt.subplots(figsize=(9, 4.5))
                b1 = ax.bar(x - w/2,
                            [parsed["MobileNetV2"].get(m, 0) for m in metrics_list],
                            w, label="MobileNetV2", color="#3498db", alpha=0.85)
                b2 = ax.bar(x + w/2,
                            [parsed["ResNet50"].get(m, 0) for m in metrics_list],
                            w, label="ResNet50",    color="#e74c3c", alpha=0.85)

                for bar in list(b1) + list(b2):
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

                ax.set_xticks(x); ax.set_xticklabels(metrics_list, fontsize=11)
                ax.set_ylim(0, 1.12)
                ax.set_ylabel("Score", fontsize=11)
                ax.set_title("MobileNetV2 vs ResNet50 — All Metrics", fontsize=13, fontweight="bold")
                ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)

    # ── Conclusion text ───────────────────────────────────────────────────────
    st.markdown("### Project Conclusion Document")
    if CONC_FILE.exists():
        st.text(CONC_FILE.read_text(encoding="utf-8"))
    else:
        st.info("Run `python train.py` to auto-generate `conclusion/conclusion.txt`.")

    # ── Key observations ──────────────────────────────────────────────────────
    st.markdown("### Key Observations")
    st.markdown(
        """
        #### Why these two models?
        | Model | Strength | Weakness |
        |-------|----------|----------|
        | MobileNetV2 | Fast, lightweight, deployable on edge | Lower capacity — may miss subtle texture differences |
        | ResNet50 | Strong feature extraction, layer4 fine-tuned | Slower inference, more parameters |

        #### Primary visual differentiators
        - **Snout shape** — Alligator: broad rounded U-shape · Crocodile: narrow tapered V-shape
        - **Teeth visibility** — Crocodile's lower teeth remain visible when the jaw is closed
        - **Skin texture** — Subtle differences in scale pattern and colour

        #### Hardest cases
        - Pure side-profile shots where the snout shape cannot be determined
        - Submerged images (only eyes/nose visible above water)
        - Juvenile specimens with undeveloped distinguishing features
        - High-clutter backgrounds with no clear body in frame
        """
    )

    # ── Future improvements ───────────────────────────────────────────────────
    st.markdown("### Recommendations for Production")
    st.markdown(
        """
        1. **Full backbone fine-tuning** with LR = 1e-5 after head convergence
        2. **EfficientNetB4 or ConvNeXt** for better accuracy-efficiency tradeoff
        3. **GradCAM visualisation** to verify model attends to the snout region
        4. **Test-time augmentation (TTA)** — average predictions across flips/crops
        5. **Focal Loss** if class imbalance increases in production data
        6. **Attention modules** (CBAM / SE) to focus on discriminative regions
        """
    )
