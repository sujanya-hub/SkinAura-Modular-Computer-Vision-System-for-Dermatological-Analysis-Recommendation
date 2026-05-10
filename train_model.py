"""
train_model.py  —  SkinAura  |  Final optimized training pipeline
==================================================================
Architecture : EfficientNetB0 (ImageNet pretrained)
Preprocessing: efficientnet.preprocess_input  INSIDE model graph
Augmentation : dynamic every epoch, operates in [0,255], disk-cached raw images
Precision    : mixed_float16  (output layer pinned to float32)
Fine-tuning  : 3-phase staged unfreezing with cosine-decay LR

Canonical preprocessing flow:
  raw uint8 [0,255]
  → cast float32
  → disk cache (raw only)
  → dynamic augmentation  [0,255]
  → clip [0,255]
  → efficientnet.preprocess_input  ← inside model graph
  → EfficientNetB0 backbone
  → classifier head
  → softmax float32
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# ── Aliases ───────────────────────────────────────────────────────────────────
keras = tf.keras
layers = keras.layers
Adam = keras.optimizers.Adam
EarlyStopping = keras.callbacks.EarlyStopping
ModelCheckpoint = keras.callbacks.ModelCheckpoint
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
CSVLogger = keras.callbacks.CSVLogger

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Mixed precision ───────────────────────────────────────────────────────────
tf.keras.mixed_precision.set_global_policy("mixed_float16")
logger.info("Mixed precision: mixed_float16")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR      = "datasets/skin_conditions"
IMAGE_SIZE       = (192, 192)   # reduced from 224: ~27% fewer pixels per image;
                                # EfficientNetB0 handles 192 well with minimal
                                # accuracy loss and meaningful speed gain.
BATCH_SIZE       = 32
VALIDATION_SPLIT = 0.2
SEED             = 42

# Epoch schedule — tuned for fast iteration without sacrificing transfer quality.
EPOCHS_HEAD    = 10   # head converges quickly; EarlyStopping fires before 10 anyway
EPOCHS_FT_WARM =  6   # cautious warm-up; 6 is enough for top-20 layers to adapt
EPOCHS_FT_DEEP =  6   # deep phase; cosine decay + EarlyStopping(patience=3) governs

FINETUNE_LAYERS_WARM = 20
FINETUNE_LAYERS_DEEP = 50

CONFIDENCE_THRESHOLD = 0.60   # predictions below this are flagged "uncertain"

# Gaussian noise stddev expressed as a fraction of max pixel value.
# Keras GaussianNoise expects stddev in the SAME units as the input tensor.
# Our inputs are float32 in [0, 255], so stddev=5.0 should be valid — BUT
# some TF builds enforce stddev ∈ (0, 1] regardless.
# FIX: apply noise via a Lambda layer using tf.random.normal, which has
# no range restriction, giving us full control over the noise scale.
GAUSSIAN_NOISE_STD = 5.0   # ≈ 2% of 255 — realistic sensor noise

# Paths
MODEL_SAVE_DIR = "backend/models"
CACHE_DIR      = "cache"
MODEL_PATH     = f"{MODEL_SAVE_DIR}/skin_issue_model.keras"
BEST_MODEL_PATH = f"{MODEL_SAVE_DIR}/best_model.keras"
CLASS_NAMES_PATH = f"{MODEL_SAVE_DIR}/class_names.json"
METADATA_PATH  = f"{MODEL_SAVE_DIR}/model_metadata.json"
CSV_LOG_PATH   = f"{MODEL_SAVE_DIR}/training_log.csv"
TRAIN_CACHE    = f"{CACHE_DIR}/train_cache"
VAL_CACHE      = f"{CACHE_DIR}/val_cache"

AUTOTUNE = tf.data.AUTOTUNE

for _d in (MODEL_SAVE_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Focal loss with label smoothing
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(keras.losses.Loss):
    """
    Multi-class focal loss.
    - gamma        : down-weights easy samples; focuses on hard / minority classes.
    - label_smoothing: prevents overconfident outputs; improves calibration.
    All tensors are cast to float32 for numerical safety under mixed_float16.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        name: str = "focal_loss",
    ):
        super().__init__(name=name)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true      = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred      = tf.cast(y_pred, tf.float32)
        y_pred      = tf.clip_by_value(y_pred, 1e-7, 1.0)
        num_classes = tf.shape(y_pred)[-1]

        y_one_hot = tf.one_hot(y_true, num_classes)
        if self.label_smoothing > 0.0:
            smooth    = self.label_smoothing / tf.cast(num_classes, tf.float32)
            y_one_hot = y_one_hot * (1.0 - self.label_smoothing) + smooth

        ce           = -tf.reduce_sum(y_one_hot * tf.math.log(y_pred), axis=-1)
        p_t          = tf.reduce_sum(y_one_hot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        return tf.reduce_mean(focal_weight * ce)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "gamma": self.gamma,
            "label_smoothing": self.label_smoothing,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(dataset_dir: str, class_names: list[str]) -> None:
    """
    Pre-training sanity checks (non-blocking — logs warnings, never crashes):
    - Corrupted / unreadable images detected via tf.image.decode_image.
    - Exact duplicate files detected by MD5 hash.
    - Severe class imbalance (>5×) warned.
    """
    data_dir  = pathlib.Path(dataset_dir)
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    corrupt: list[str]     = []
    dupes: list[str]       = []
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    logger.info("── Dataset validation ───────────────────────────")
    for cls in class_names:
        cls_dir = data_dir / cls
        valid   = 0
        for fp in sorted(cls_dir.iterdir()):
            if fp.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                raw = fp.read_bytes()
                tf.image.decode_image(raw, channels=3, expand_animations=False)
                md5 = hashlib.md5(raw).hexdigest()
                if md5 in hashes:
                    dupes.append(f"{fp}  ↔  {hashes[md5]}")
                else:
                    hashes[md5] = str(fp)
                valid += 1
            except Exception:
                corrupt.append(str(fp))
        counts[cls] = valid

    if corrupt:
        logger.warning(f"  Corrupted images ({len(corrupt)}): {corrupt[:3]} …")
    if dupes:
        logger.warning(f"  Duplicates ({len(dupes)}): {dupes[:3]} …")

    vals = list(counts.values())
    ratio = max(vals) / max(min(vals), 1)
    if ratio > 5:
        logger.warning(f"  Imbalance {ratio:.1f}× — consider oversampling or extra augmentation.")
    else:
        logger.info(f"  Imbalance ratio {ratio:.1f}×  ✓")

    for cls, n in counts.items():
        logger.info(f"  {cls:<35} {n:>5} valid images")


# ─────────────────────────────────────────────────────────────────────────────
# Class weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(dataset_dir: str, class_names: list[str]) -> dict[int, float]:
    """Balanced class weights: total / (n_classes × class_count)."""
    data_dir = pathlib.Path(dataset_dir)
    counts = {
        idx: len(list((data_dir / cls).glob("*")))
        for idx, cls in enumerate(class_names)
    }
    total     = sum(counts.values())
    n_classes = len(counts)
    weights   = {
        idx: round(total / (n_classes * max(cnt, 1)), 4)
        for idx, cnt in counts.items()
    }
    logger.info("── Class weights ────────────────────────────────")
    for idx, cls in enumerate(class_names):
        logger.info(f"  [{idx}] {cls:<35} weight = {weights[idx]}")
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def _add_gaussian_noise(x: tf.Tensor, std: float = GAUSSIAN_NOISE_STD) -> tf.Tensor:
    """
    Add Gaussian noise in [0,255] space using tf.random.normal.

    WHY NOT layers.GaussianNoise(5.0)?
    Some TF/Keras builds enforce stddev ∈ (0,1] when the layer is used inside
    a Sequential model, raising:
        ValueError: Invalid value received for argument `stddev`.
        Expected a float value between 0 and 1. Received: stddev=5.0

    tf.random.normal has no such restriction, produces identical output,
    and is a pure TF op — fully graph-compatible, no py_function needed.
    Noise is only added during training (the Lambda is called with training=True
    inside build_augmentation's Sequential).
    """
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=std, dtype=x.dtype)
    return x + noise


def build_augmentation() -> keras.Sequential:
    """
    Mild, dermatology-safe augmentation pipeline.
    All ops run in float32 [0,255].  Clip to [0,255] is applied in the
    tf.data map step AFTER this sequential, so overflow from noise is handled.

    Policy rationale for skincare images:
    - Flip / Rotation / Zoom / Translation : lesions appear at arbitrary
      location and orientation on face / body.
    - Brightness / Contrast               : simulate varied lighting and
      different skin tones / camera exposures.
    - Gaussian noise (std=5 ≈ 2% of 255) : models camera sensor noise and
      mild JPEG compression artifacts without destroying lesion texture.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomBrightness(0.15),
            layers.RandomContrast(0.15),
            # Gaussian noise via Lambda — no stddev range restriction.
            layers.Lambda(
                _add_gaussian_noise,
                name="gaussian_noise",
            ),
        ],
        name="augmentation",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_datasets(
    dataset_dir: str,
    image_size: tuple[int, int],
    batch_size: int,
    validation_split: float,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Returns raw (uint8 [0,255]) train/val datasets and the Keras-assigned
    class list (alphabetical folder order = label-index ground truth).

    FIX:
    Automatically removes corrupted TensorFlow-invalid images BEFORE
    dataset loading to prevent:

        InvalidArgumentError:
        Number of channels requested does not match input
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    data_dir = pathlib.Path(dataset_dir)
    removed_files: list[str] = []

    logger.info("── TensorFlow corrupted-image cleanup ─────────")

    for class_dir in sorted(data_dir.iterdir()):

        if not class_dir.is_dir():
            continue

        for img_path in sorted(class_dir.iterdir()):

            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            try:
                raw = tf.io.read_file(str(img_path))

                # Force RGB decoding
                tf.image.decode_image(
                    raw,
                    channels=3,
                    expand_animations=False,
                )

            except Exception as e:

                removed_files.append(str(img_path))

                try:
                    os.remove(img_path)

                    logger.warning(
                        f"Removed corrupted image: {img_path} | Error: {e}"
                    )

                except Exception as delete_error:

                    logger.error(
                        f"Failed to delete corrupted image: {img_path} | "
                        f"Delete Error: {delete_error}"
                    )

    if removed_files:

        logger.warning(
            f"Removed {len(removed_files)} corrupted images before dataset loading."
        )

    else:

        logger.info("No corrupted TensorFlow-invalid images found.")

    # ─────────────────────────────────────────────────────────
    # Dataset loading
    # ─────────────────────────────────────────────────────────

    common = dict(
        directory        = dataset_dir,
        validation_split = validation_split,
        seed             = seed,
        image_size       = image_size,
        batch_size       = batch_size,
        label_mode       = "int",
    )

    train_ds = keras.utils.image_dataset_from_directory(
        subset="training",
        **common
    )

    val_ds = keras.utils.image_dataset_from_directory(
        subset="validation",
        **common
    )

    class_names: list[str] = [
        n.lower()
        for n in train_ds.class_names
    ]

    logger.info("── Keras class order (label ground truth) ───────")

    for idx, cls in enumerate(class_names):

        logger.info(f"  {idx} → {cls}")

    return train_ds, val_ds, class_names
    
# ─────────────────────────────────────────────────────────────────────────────
# tf.data pipelines
# ─────────────────────────────────────────────────────────────────────────────

def build_train_pipeline(
    raw_ds: tf.data.Dataset,
    augmentation: keras.Sequential,
) -> tf.data.Dataset:
    """
    cast → disk-cache (raw float32) → augment+clip (fresh every epoch) → prefetch

    cache() is BEFORE the augmentation map: decoded images are cached to disk;
    augmentation runs with fresh random seeds every epoch.
    """
    def cast_to_float(x, y):
        return tf.cast(x, tf.float32), y

    def augment_and_clip(x, y):
        x = augmentation(x, training=True)
        x = tf.clip_by_value(x, 0.0, 255.0)
        return x, y

    return (
        raw_ds
        .map(cast_to_float,    num_parallel_calls=AUTOTUNE)
        .cache(TRAIN_CACHE)
        .map(augment_and_clip, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )


def build_val_pipeline(raw_ds: tf.data.Dataset) -> tf.data.Dataset:
    """
    cast → disk-cache → prefetch.
    No augmentation. No manual normalisation (preprocess_input is in graph).
    """
    def cast_to_float(x, y):
        return tf.cast(x, tf.float32), y

    return (
        raw_ds
        .map(cast_to_float, num_parallel_calls=AUTOTUNE)
        .cache(VAL_CACHE)
        .prefetch(AUTOTUNE)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    num_classes: int,
    image_size: tuple[int, int],
) -> tuple[keras.Model, keras.Model]:
    """
    EfficientNetB0 transfer-learning model.

    Input contract : float32 [0,255]  (augmented + clipped externally)
    Preprocessing  : efficientnet.preprocess_input inside graph  →  [0,1]
    Grad-CAM       : Lambda preprocessing sits before conv stack;
                     gradient path to base's top_conv is fully intact.
    Mixed precision: output Dense pinned to float32 to prevent softmax overflow.
    """
    inputs = layers.Input(shape=image_size + (3,), name="input_image")

    # ── EfficientNet preprocessing — inside graph (Grad-CAM safe) ─────────────
    x = layers.Lambda(
        tf.keras.applications.efficientnet.preprocess_input,
        name="efficientnet_preprocess",
    )(inputs)

    # ── Backbone ──────────────────────────────────────────────────────────────
    base = keras.applications.EfficientNetB0(
        input_shape=image_size + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    x = base(x, training=False)

    # ── Classifier head ───────────────────────────────────────────────────────
    x       = layers.GlobalAveragePooling2D(name="gap")(x)
    x       = layers.BatchNormalization(name="bn")(x)
    x       = layers.Dense(256, activation="relu", name="dense_256")(x)
    x       = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",      # fp32 required for numerical stability
        name="predictions",
    )(x)

    model = keras.Model(inputs, outputs, name="SkinAura_EfficientNetB0")
    return model, base


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate schedule
# ─────────────────────────────────────────────────────────────────────────────

def cosine_decay(
    initial_lr: float,
    epochs: int,
    steps_per_epoch: int = 100,
) -> keras.optimizers.schedules.CosineDecay:
    """Cosine annealing: smoother convergence than step-wise ReduceLROnPlateau."""
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=epochs * steps_per_epoch,
        alpha=1e-7,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    model: keras.Model,
    val_ds: tf.data.Dataset,
    class_names: list[str],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Post-training evaluation suite:
    - Confusion matrix
    - Per-class precision / recall / F1
    - Macro F1
    - ROC-AUC (one-vs-rest)
    - Confidence histogram
    - Uncertain prediction count (conf < threshold)
    - Prediction distribution
    """
    y_true_list, y_pred_list, y_prob_list = [], [], []

    for images, labels in val_ds:
        probs = model.predict(images, verbose=0).astype(np.float32)
        y_true_list.extend(labels.numpy())
        y_pred_list.extend(np.argmax(probs, axis=1))
        y_prob_list.append(probs)

    y_true  = np.array(y_true_list)
    y_pred  = np.array(y_pred_list)
    y_probs = np.vstack(y_prob_list)
    confs   = y_probs.max(axis=1)

    uncertain_count = int((confs < confidence_threshold).sum())
    uncertain_pct   = uncertain_count / max(len(y_true), 1) * 100

    report   = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        y_bin   = label_binarize(y_true, classes=list(range(len(class_names))))
        roc_auc = roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr")
    except Exception:
        roc_auc = float("nan")

    cm       = confusion_matrix(y_true, y_pred)
    hist, edges = np.histogram(confs, bins=10, range=(0.0, 1.0))
    conf_hist   = {f"{edges[i]:.1f}-{edges[i+1]:.1f}": int(hist[i]) for i in range(len(hist))}
    pred_dist   = {class_names[k]: v for k, v in Counter(y_pred).items()}

    logger.info("── Evaluation report ────────────────────────────")
    logger.info(f"\n{report}")
    logger.info(f"  Macro F1  : {macro_f1:.4f}")
    logger.info(f"  ROC-AUC   : {roc_auc:.4f}")
    logger.info(f"  Uncertain : {uncertain_count} / {len(y_true)}  ({uncertain_pct:.1f}%)")

    logger.info("── Confusion matrix ─────────────────────────────")
    header = "".join(f"{n[:6]:>8}" for n in class_names)
    logger.info(f"  {'':>8}{header}")
    for i, row in enumerate(cm):
        logger.info(f"  {class_names[i][:8]:>8}" + "".join(f"{v:>8}" for v in row))

    logger.info("── Prediction distribution ──────────────────────")
    for cls, cnt in pred_dist.items():
        logger.info(f"  {cls:<35} {cnt:>6}")

    normal_idx = class_names.index("normal") if "normal" in class_names else -1
    if normal_idx >= 0 and Counter(y_pred).get(normal_idx, 0) > len(y_true) * 0.6:
        logger.warning("  'normal' >60%% of predictions — review class weights.")

    return {
        "macro_f1"         : round(float(macro_f1), 4),
        "roc_auc"          : round(float(roc_auc), 4),
        "uncertain_count"  : uncertain_count,
        "uncertain_pct"    : round(uncertain_pct, 2),
        "conf_histogram"   : conf_hist,
        "pred_distribution": pred_dist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Saving helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_class_names(class_names: list[str], path: str) -> None:
    """Single source of truth for inference — model_loader.py reads this file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(class_names, indent=2))
    logger.info(f"Class names → {path}")


def save_metadata(
    path: str,
    class_names: list[str],
    image_size: tuple[int, int],
    val_acc: float,
    val_loss: float,
    eval_metrics: dict,
) -> None:
    metadata = {
        "architecture"       : "EfficientNetB0",
        "tensorflow_version" : tf.__version__,
        "image_size"         : list(image_size),
        "class_names"        : class_names,
        "num_classes"        : len(class_names),
        "validation_accuracy": round(float(val_acc), 4),
        "validation_loss"    : round(float(val_loss), 4),
        "macro_f1"           : eval_metrics.get("macro_f1"),
        "roc_auc"            : eval_metrics.get("roc_auc"),
        "training_date"      : datetime.now().isoformat(),
        "preprocessing"      : (
            "efficientnet.preprocess_input inside model graph — "
            "input [0,255] float32 → backbone [0,1] normalised"
        ),
        "augmentation"       : [
            "RandomFlip(horizontal)",
            "RandomRotation(0.10)",
            "RandomZoom(0.10)",
            "RandomTranslation(0.08)",
            "RandomBrightness(0.15)",
            "RandomContrast(0.15)",
            f"GaussianNoise via tf.random.normal(std={GAUSSIAN_NOISE_STD})",
        ],
        "loss"               : "FocalLoss(gamma=2.0, label_smoothing=0.1) + class_weight",
        "fine_tuning"        : (
            f"Phase 1: frozen backbone ({EPOCHS_HEAD} epochs) | "
            f"Phase 2: last {FINETUNE_LAYERS_WARM} layers, cosine LR 5e-5 ({EPOCHS_FT_WARM} epochs) | "
            f"Phase 3: last {FINETUNE_LAYERS_DEEP} layers, cosine LR 1e-5 ({EPOCHS_FT_DEEP} epochs)"
        ),
        "mixed_precision"    : "mixed_float16  (output layer float32)",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "notes"              : (
            "class_names matches Keras alphabetical folder order. "
            "model_loader.py reads class_names.json. "
            "Input contract: float32 [0,255]. "
            "GradCAM targets top_conv inside EfficientNetB0 base."
        ),
    }
    Path(path).write_text(json.dumps(metadata, indent=2))
    logger.info(f"Metadata → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── 1. Load raw datasets ──────────────────────────────────────────────────
    train_ds_raw, val_ds_raw, class_names = load_datasets(
        DATASET_DIR, IMAGE_SIZE, BATCH_SIZE, VALIDATION_SPLIT, SEED
    )
    num_classes = len(class_names)

    # ── 2. Persist class names immediately (crash-safe) ───────────────────────
    save_class_names(class_names, CLASS_NAMES_PATH)

    # ── 3. Validate dataset ───────────────────────────────────────────────────
    validate_dataset(DATASET_DIR, class_names)

    # ── 4. Class weights ──────────────────────────────────────────────────────
    class_weights = compute_class_weights(DATASET_DIR, class_names)

    # ── 5. Build tf.data pipelines ────────────────────────────────────────────
    augmentation = build_augmentation()
    train_ds     = build_train_pipeline(train_ds_raw, augmentation)
    val_ds       = build_val_pipeline(val_ds_raw)

    # ── 6. Build model ────────────────────────────────────────────────────────
    model, base_model = build_model(num_classes, IMAGE_SIZE)
    model.summary()

    focal_loss = FocalLoss(gamma=2.0, label_smoothing=0.1)

    # Shared callback sets — EarlyStopping patience=3 for faster termination.
    def _base_callbacks(append_csv: bool) -> list:
        return [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ModelCheckpoint(BEST_MODEL_PATH, monitor="val_loss", save_best_only=True),
            CSVLogger(CSV_LOG_PATH, append=append_csv),
        ]

    # ── Phase 1: classifier head, backbone frozen ─────────────────────────────
    logger.info("── Phase 1: classifier head  (backbone frozen) ─────────────")
    model.compile(
        optimizer=Adam(learning_rate=3e-4, clipnorm=1.0),
        loss=focal_loss,
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=_base_callbacks(append_csv=False),
        verbose=1,
    )

    # ── Phase 2: unfreeze last 20 layers (warm-up) ────────────────────────────
    logger.info(f"── Phase 2: fine-tune last {FINETUNE_LAYERS_WARM} layers (warm-up) ────────")
    base_model.trainable = True
    for layer in base_model.layers[:-FINETUNE_LAYERS_WARM]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=cosine_decay(5e-5, EPOCHS_FT_WARM), clipnorm=1.0),
        loss=focal_loss,
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FT_WARM,
        class_weight=class_weights,
        callbacks=_base_callbacks(append_csv=True),
        verbose=1,
    )

    # ── Phase 3: unfreeze last 50 layers (deep) ───────────────────────────────
    logger.info(f"── Phase 3: fine-tune last {FINETUNE_LAYERS_DEEP} layers (deep) ──────────")
    for layer in base_model.layers[:-FINETUNE_LAYERS_DEEP]:
        layer.trainable = False
    for layer in base_model.layers[-FINETUNE_LAYERS_DEEP:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=cosine_decay(1e-5, EPOCHS_FT_DEEP), clipnorm=1.0),
        loss=focal_loss,
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FT_DEEP,
        class_weight=class_weights,
        callbacks=_base_callbacks(append_csv=True) + [
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-8),
        ],
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_loss, val_acc = model.evaluate(val_ds, verbose=1)
    logger.info(f"Final  →  acc: {val_acc:.4f}  |  loss: {val_loss:.4f}")
    eval_metrics = run_evaluation(model, val_ds, class_names, CONFIDENCE_THRESHOLD)

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    logger.info(f"Model saved → {MODEL_PATH}")
    save_metadata(METADATA_PATH, class_names, IMAGE_SIZE, val_acc, val_loss, eval_metrics)


if __name__ == "__main__":
    main()