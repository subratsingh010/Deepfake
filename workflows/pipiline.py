#!/usr/bin/env python3
"""
Real-only deepfake false-positive benchmark.

Single-file benchmark for:
  - normal testing at original, 1024, 720, 512, 256
  - stress testing of selected TN/FP originals at 1024, 720, 512, 256
  - one new central CSV per run, with checkpointing inside that run
  - summary, threshold sweep, and errors reports

Source images are never modified or deleted.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
import argparse
import re
import shutil
import tempfile
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

try:
    import cv2
except ImportError:
    cv2 = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import torch
except ImportError:
    torch = None
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
except ImportError:
    Image = None
    ImageEnhance = None
    ImageFilter = None
    ImageOps = None
    UnidentifiedImageError = OSError
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    from transformers import pipeline
except ImportError:
    pipeline = None


# ============================================================
# CONFIG
# ============================================================

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DATA_ROOT = Path("/Users/subrat/Desktop/Deepfake")
OUTPUT_DIR = Path.cwd() / "output"
GENERATED_VARIANT_DIR = OUTPUT_DIR / "generated_variants"
FAILED_EXPORT_DIR = OUTPUT_DIR / "failed_images"

DISCOVERED_SOURCES: list[str] = []

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

MODEL_NAME = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
THRESHOLD = 0.50

BATCH_SIZE = 8
CHECKPOINT_EVERY = 25
RANDOM_SEED = 42
KEEP_GENERATED_VARIANTS = True
STRESS_SAMPLE_COUNT = 100
METADATA_WORKERS = 4

NORMAL_TARGET_DIMENSIONS: list[str | int] = ["original", 1024, 720, 512, 256]
STRESS_TARGET_DIMENSIONS: list[str | int] = ["original", 1024, 720, 512, 256]
RESIZE_MODES = ["aspect", "square"]

JPEG_QUALITY_LEVELS = [95, 80, 60, 40]

MAIN_CSV = OUTPUT_DIR / "false_positive_complete_benchmark.csv"
SUMMARY_TXT = OUTPUT_DIR / "benchmark_summary.txt"
COLUMN_GUIDE_TXT = OUTPUT_DIR / "column_guide.txt"


CENTRAL_COLUMNS = [
    "variant_id",
    "parent_image_id",
    "original_image_id",
    "source",
    "source_subgroup",
    "test_type",
    "actual_label",
    "original_path",
    "original_file_name",
    "original_file_format",
    "original_width",
    "original_height",
    "original_megapixels",
    "source_fps",
    "target_dimension",
    "resize_mode",
    "variant_path",
    "variant_file_name",
    "variant_file_format",
    "variant_width",
    "variant_height",
    "variant_megapixels",
    "file_size_mb",
    "resolution_bucket",
    "brightness_mean",
    "brightness_bucket",
    "contrast_std",
    "blur_score",
    "blur_bucket",
    "sharpness_score",
    "model_name",
    "threshold",
    "normal_original_fake_probability",
    "normal_original_real_probability",
    "normal_same_resolution_fake_probability",
    "normal_same_resolution_real_probability",
    "normal_same_resolution_variant_id",
    "normal_prediction",
    "normal_result",
    "score_delta_resolution",
    "stress_type",
    "stress_level",
    "stress_parameter",
    "stress_fake_probability",
    "stress_real_probability",
    "stress_prediction",
    "stress_result",
    "prediction_transition",
    "score_delta_vs_clean",
    "score_delta_vs_original",
    "inference_ms",
    "random_seed",
    "stats_group_key",
    "stats_group_count",
    "stats_group_false_positives",
    "stats_group_false_positive_rate",
    "stats_group_score_mean",
    "stats_group_score_median",
    "stats_group_score_std",
    "stats_group_score_min",
    "stats_group_score_max",
    "stats_group_score_p95",
    "error",
]


COLUMN_DESCRIPTIONS = {
    "variant_id": "Unique id for this tested row/variant. Used to identify exactly one normal or stress test case.",
    "parent_image_id": "Stable id for the original source image, including the source category prefix.",
    "original_image_id": "Stable hash id for the original image path.",
    "source": "Source folder/category discovered under the input data root.",
    "source_subgroup": "Subfolder under the source folder, if any.",
    "test_type": "normal for clean resized/original tests, stress for one controlled edit factor.",
    "actual_label": "Ground truth label. This benchmark is real-only, so this is real.",
    "original_path": "Full path to the original source image. The script only reads this file.",
    "variant_path": "Path to the tested variant if it is persistent. Usually blank for temp variants when KEEP_GENERATED_VARIANTS is False.",
    "original_file_name": "Original source image filename.",
    "variant_file_name": "Generated logical filename for the tested variant.",
    "original_file_format": "Original file extension/format, for example jpg or png.",
    "variant_file_format": "Format used for the tested variant, for example jpg or png.",
    "original_width": "Original image width in pixels.",
    "original_height": "Original image height in pixels.",
    "original_megapixels": "Original image size in megapixels.",
    "target_dimension": "normal rows use original only. stress rows use original plus 1024, 720, 512, or 256.",
    "resize_mode": "For stress resized rows: aspect keeps original aspect ratio; square center-crops then resizes to target x target. Blank for original rows.",
    "variant_width": "Actual tested variant width after aspect-ratio-preserving resize.",
    "variant_height": "Actual tested variant height after aspect-ratio-preserving resize.",
    "variant_megapixels": "Tested variant size in megapixels.",
    "file_size_mb": "Actual disk size in MB of the exact image file tested by the model.",
    "resolution_bucket": "Bucket based on tested variant minimum dimension.",
    "brightness_mean": "Mean grayscale brightness measured from the tested variant.",
    "brightness_bucket": "Brightness category derived from brightness_mean.",
    "contrast_std": "Standard deviation of grayscale intensity; higher usually means more contrast.",
    "blur_score": "Laplacian variance blur/sharpness score; lower usually means blurrier.",
    "blur_bucket": "Fixed blur category derived from blur_score.",
    "sharpness_score": "Sharpness proxy. Currently same Laplacian-based value as blur_score.",
    "source_fps": "FPS parsed from filename if it contains a pattern like _30fps_; otherwise blank.",
    "stress_type": "Stress factor type: blur, brightness, sharpness, contrast, or jpeg_compression. Blank for normal rows.",
    "stress_level": "Stress factor level such as low, medium, high, darker, brighter, lower, higher, q80. Blank for normal rows.",
    "stress_parameter": "Exact stress parameter used, for example radius=2 or quality=80. Blank for normal rows.",
    "model_name": "Hugging Face model used for inference.",
    "threshold": "Decision threshold. Score >= threshold means fake prediction.",
    "normal_original_fake_probability": "Fake score of the original-resolution normal test for the same original image.",
    "normal_original_real_probability": "1 - normal_original_fake_probability.",
    "normal_same_resolution_fake_probability": "For normal rows, the clean score for this row. For stress rows, the matching clean normal score using the same parent_image_id, target_dimension, and resize_mode.",
    "normal_same_resolution_real_probability": "1 - normal_same_resolution_fake_probability.",
    "normal_same_resolution_variant_id": "Variant id of the clean normal baseline row matched by parent_image_id, target_dimension, and resize_mode.",
    "stress_fake_probability": "Fake score for this stress variant. Blank for normal rows.",
    "stress_real_probability": "1 - stress_fake_probability. Blank for normal rows.",
    "normal_prediction": "Normal row prediction: real or fake.",
    "stress_prediction": "Stress row prediction: real or fake.",
    "normal_result": "Real-only result for normal rows: TN if predicted real, FP if predicted fake.",
    "stress_result": "Real-only result for stress rows: TN if predicted real, FP if predicted fake.",
    "prediction_transition": "For stress rows, transition from clean same-resolution result to stress result: TN_to_TN, TN_to_FP, FP_to_TN, FP_to_FP.",
    "score_delta_resolution": "normal_same_resolution_fake_probability - normal_original_fake_probability.",
    "score_delta_vs_clean": "stress_fake_probability - normal_same_resolution_fake_probability.",
    "score_delta_vs_original": "stress_fake_probability - normal_original_fake_probability.",
    "inference_ms": "Approximate model inference time per image in milliseconds for the batch.",
    "random_seed": "Seed retained for reproducibility metadata. Passed stress images are selected from high/near-threshold, middle-score, and low-score TN groups.",
    "stats_group_key": "Group key used for row-level summary stats.",
    "stats_group_count": "Number of rows in this row's stats group.",
    "stats_group_false_positives": "False positives in this row's stats group.",
    "stats_group_false_positive_rate": "False positive rate in this row's stats group.",
    "stats_group_score_mean": "Mean fake score in this row's stats group.",
    "stats_group_score_median": "Median fake score in this row's stats group.",
    "stats_group_score_std": "Standard deviation of fake scores in this row's stats group.",
    "stats_group_score_min": "Minimum fake score in this row's stats group.",
    "stats_group_score_max": "Maximum fake score in this row's stats group.",
    "stats_group_score_p95": "95th percentile fake score in this row's stats group.",
    "error": "Error message if this row failed during preparation or inference; blank means success.",
}


DEVICE: str | int = -1


# ============================================================
# HELPERS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete real-only deepfake false-positive benchmark."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Input dataset root. Immediate subdirectories are treated as source categories. If a wrapper data/ folder is detected, it is used automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: ./output in the current working directory.",
    )
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--stress-sample-count", type=int, default=STRESS_SAMPLE_COUNT)
    parser.add_argument(
        "--keep-generated-variants",
        action="store_true",
        default=KEEP_GENERATED_VARIANTS,
        help="Persist generated normal/stress variants. By default they are temporary and discarded.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id for output filenames. Default: current timestamp.",
    )
    return parser.parse_args()


def ensure_package(import_name: str, package_name: str | None = None) -> Any:
    try:
        return import_module(import_name)
    except ImportError:
        package = package_name or import_name
        os.system(f'"{os.sys.executable}" -m pip install -q {package}')
        return import_module(import_name)


def ensure_runtime_dependencies() -> None:
    global cv2, np, pd, torch, Image, ImageEnhance, ImageFilter, ImageOps
    global UnidentifiedImageError, tqdm, pipeline, DEVICE

    np = ensure_package("numpy")
    pd = ensure_package("pandas")
    cv2 = ensure_package("cv2", "opencv-python")
    torch = ensure_package("torch")
    ensure_package("PIL", "pillow")
    Image = ensure_package("PIL.Image", "pillow")
    ImageEnhance = ensure_package("PIL.ImageEnhance", "pillow")
    ImageFilter = ensure_package("PIL.ImageFilter", "pillow")
    ImageOps = ensure_package("PIL.ImageOps", "pillow")
    pil_module = ensure_package("PIL", "pillow")
    UnidentifiedImageError = getattr(pil_module, "UnidentifiedImageError")
    tqdm_module = ensure_package("tqdm")
    tqdm = tqdm_module.tqdm
    transformers_pipeline = ensure_package("transformers").pipeline
    pipeline = transformers_pipeline

    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = 0
    else:
        DEVICE = -1


def configure_from_args(args: argparse.Namespace) -> None:
    global DATA_ROOT, OUTPUT_DIR, GENERATED_VARIANT_DIR, FAILED_EXPORT_DIR, MAIN_CSV, SUMMARY_TXT, COLUMN_GUIDE_TXT
    global DISCOVERED_SOURCES, THRESHOLD, BATCH_SIZE, CHECKPOINT_EVERY, RANDOM_SEED
    global STRESS_SAMPLE_COUNT, KEEP_GENERATED_VARIANTS

    DATA_ROOT = resolve_dataset_root(args.data_root.expanduser().resolve())
    OUTPUT_DIR = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path.cwd() / "output"
    )
    GENERATED_VARIANT_DIR = OUTPUT_DIR / "generated_variants"
    run_id = clean_name(args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    model_name = model_file_name()
    MAIN_CSV = OUTPUT_DIR / f"false_positive_complete_benchmark_{model_name}_{run_id}.csv"
    SUMMARY_TXT = OUTPUT_DIR / f"benchmark_summary_{model_name}_{run_id}.txt"
    COLUMN_GUIDE_TXT = OUTPUT_DIR / f"column_guide_{model_name}_{run_id}.txt"
    FAILED_EXPORT_DIR = OUTPUT_DIR / f"failed_images_{model_name}_{run_id}"

    THRESHOLD = args.threshold
    BATCH_SIZE = args.batch_size
    CHECKPOINT_EVERY = args.checkpoint_every
    RANDOM_SEED = args.random_seed
    STRESS_SAMPLE_COUNT = args.stress_sample_count
    KEEP_GENERATED_VARIANTS = args.keep_generated_variants

    DISCOVERED_SOURCES = discover_source_dirs(DATA_ROOT)


def has_direct_supported_images(directory: Path) -> bool:
    return any(is_supported_image_path(path) for path in directory.iterdir() if path.is_file())


def child_dirs_with_direct_images(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    return [
        child
        for child in sorted(data_root.iterdir())
        if child.is_dir() and has_direct_supported_images(child)
    ]


def resolve_dataset_root(data_root: Path) -> Path:
    if child_dirs_with_direct_images(data_root):
        return data_root

    nested_data = data_root / "data"
    if nested_data.is_dir() and child_dirs_with_direct_images(nested_data):
        return nested_data

    return data_root


def discover_source_dirs(data_root: Path) -> list[str]:
    ignored = {
        ".git",
        ".agents",
        ".codex",
        ".deepeval",
        ".pycache",
        "__pycache__",
        "benchmark_results",
        "output",
        "archive",
        "workflows",
    }

    sources: list[str] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir() or child.name in ignored or child.name.startswith("."):
            continue
        has_images = any(
            is_supported_image_path(path)
            for path in child.rglob("*")
        )
        if has_images:
            sources.append(child.name)

    return sources


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if KEEP_GENERATED_VARIANTS:
        GENERATED_VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    elif GENERATED_VARIANT_DIR.exists():
        shutil.rmtree(GENERATED_VARIANT_DIR)


def clean_name(value: Any) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    return safe[:180]


def model_file_name() -> str:
    return clean_name(MODEL_NAME.replace("/", "__"))


def stable_digest(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def image_id_for_path(path: Path) -> str:
    try:
        relative = str(path.relative_to(DATA_ROOT))
    except ValueError:
        relative = str(path)
    return stable_digest(relative, 16)


def variant_digest(parts: list[Any]) -> str:
    return stable_digest("|".join(str(part) for part in parts), 18)


def stable_numeric_index(text: str, width: int = 6) -> str:
    value = int(stable_digest(text, 10), 16) % (10 ** width)
    return f"{value:0{width}d}"


def parse_source_fps(filename: str) -> float:
    match = re.search(r"_(\d+(?:\.\d+)?)fps_", filename.lower())
    return float(match.group(1)) if match else float("nan")


def resolution_bucket(width: int, height: int) -> str:
    min_dim = min(width, height)
    if min_dim < 256:
        return "<256"
    if min_dim < 512:
        return "256-511"
    if min_dim < 720:
        return "512-719"
    if min_dim < 1080:
        return "720-1079"
    if min_dim < 1440:
        return "1080-1439"
    if min_dim < 2160:
        return "1440-2159"
    return ">=2160"


def brightness_bucket(value: float) -> str:
    if value < 60:
        return "very_dark"
    if value < 100:
        return "dark"
    if value < 170:
        return "normal"
    if value < 210:
        return "bright"
    return "very_bright"


def blur_bucket(value: float) -> str:
    if value < 50:
        return "very_blurry"
    if value < 150:
        return "blurry"
    if value < 500:
        return "moderate"
    if value < 1500:
        return "sharp"
    return "very_sharp"


def prediction_from_score(score: float) -> str:
    return "fake" if score >= THRESHOLD else "real"


def result_from_prediction(prediction: str) -> str:
    return "FP" if prediction == "fake" else "TN"


def transition(clean_result: str, stress_result: str) -> str:
    if not clean_result or not stress_result:
        return ""
    return f"{clean_result}_to_{stress_result}"


def target_label(target_dimension: str | int) -> str:
    return str(target_dimension)


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 6)


def source_save_format(path: Path) -> tuple[str, str, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "JPEG", ".jpg", {"quality": 95, "subsampling": 0}
    if suffix == ".png":
        return "PNG", ".png", {"compress_level": 6}
    return "PNG", ".png", {"compress_level": 6}


def is_supported_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def jpeg_save_format(quality: int) -> tuple[str, str, dict[str, Any]]:
    subsampling = 0 if quality >= 90 else 1 if quality >= 75 else 2
    return "JPEG", ".jpg", {"quality": quality, "subsampling": subsampling}


def open_source_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode not in {"RGB", "RGBA", "L"}:
        image = image.convert("RGB")
    return image


def resize_preserve_aspect(image: Image.Image, target_dimension: str | int) -> Image.Image:
    if target_dimension == "original":
        return image.copy()

    target = int(target_dimension)
    width, height = image.size
    max_dim = max(width, height)
    if max_dim == target:
        return image.copy()

    scale = target / max_dim
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def resize_square_center_crop(image: Image.Image, target_dimension: str | int) -> Image.Image:
    if target_dimension == "original":
        return image.copy()

    target = int(target_dimension)
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    cropped = image.crop((left, top, left + side, top + side))
    return cropped.resize((target, target), Image.Resampling.LANCZOS)


def resize_for_mode(image: Image.Image, target_dimension: str | int, resize_mode: str) -> Image.Image:
    if target_dimension == "original":
        return image.copy()
    if resize_mode == "square":
        return resize_square_center_crop(image, target_dimension)
    return resize_preserve_aspect(image, target_dimension)


def center_crop_preserve_scale(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, int(round(width * scale)))
    crop_height = max(1, int(round(height * scale)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def image_stats(image: Image.Image) -> dict[str, Any]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    stat_image = rgb.copy()
    stat_image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    array = np.asarray(stat_image)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = blur

    return {
        "variant_width": width,
        "variant_height": height,
        "variant_megapixels": round(width * height / 1_000_000, 6),
        "resolution_bucket": resolution_bucket(width, height),
        "brightness_mean": round(brightness, 6),
        "brightness_bucket": brightness_bucket(brightness),
        "contrast_std": round(contrast, 6),
        "blur_score": round(blur, 6),
        "blur_bucket": blur_bucket(blur),
        "sharpness_score": round(sharpness, 6),
    }


def original_metadata(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["original_path"])
    result = dict(record)
    result["error"] = ""

    try:
        with open_source_image(path) as image:
            width, height = image.size
        result.update(
            {
                "original_width": width,
                "original_height": height,
                "original_megapixels": round(width * height / 1_000_000, 6),
            }
        )
    except Exception as error:
        result["error"] = f"original_metadata_error: {error}"

    return result


# ============================================================
# DISCOVERY
# ============================================================

def discover_images() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for source in DISCOVERED_SOURCES:
        source_dir = DATA_ROOT / source
        if not source_dir.exists():
            print(f"[WARN] Missing source directory: {source_dir}")
            continue

        files = sorted(
            path
            for path in source_dir.rglob("*")
            if is_supported_image_path(path)
        )

        print(f"{source:<15}: {len(files):,} images")

        for path in files:
            relative_to_source = path.relative_to(source_dir)
            subgroup = str(relative_to_source.parent)
            original_image_id = image_id_for_path(path)
            rows.append(
                {
                    "parent_image_id": f"{source}_{original_image_id}",
                    "original_image_id": original_image_id,
                    "source": source,
                    "source_subgroup": "" if subgroup == "." else subgroup,
                    "actual_label": "real",
                    "original_path": str(path),
                    "original_file_name": path.name,
                    "original_file_format": path.suffix.lower().lstrip("."),
                    "source_fps": parse_source_fps(path.name),
                }
            )

    if not rows:
        return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=METADATA_WORKERS) as executor:
        enriched = list(
            tqdm(
                executor.map(original_metadata, rows),
                total=len(rows),
                desc="Original metadata",
                unit="img",
            )
        )

    return pd.DataFrame(enriched)


# ============================================================
# VARIANT PLANS
# ============================================================

def stress_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"stress_type": "blur", "stress_level": "low", "stress_parameter": "radius=1", "fn": lambda img: img.filter(ImageFilter.GaussianBlur(radius=1)), "jpeg_quality": None},
        {"stress_type": "blur", "stress_level": "medium", "stress_parameter": "radius=2", "fn": lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)), "jpeg_quality": None},
        {"stress_type": "blur", "stress_level": "high", "stress_parameter": "radius=4", "fn": lambda img: img.filter(ImageFilter.GaussianBlur(radius=4)), "jpeg_quality": None},
        {"stress_type": "brightness", "stress_level": "darker", "stress_parameter": "factor=0.70", "fn": lambda img: ImageEnhance.Brightness(img).enhance(0.70), "jpeg_quality": None},
        {"stress_type": "brightness", "stress_level": "brighter", "stress_parameter": "factor=1.30", "fn": lambda img: ImageEnhance.Brightness(img).enhance(1.30), "jpeg_quality": None},
        {"stress_type": "sharpness", "stress_level": "lower", "stress_parameter": "factor=0.50", "fn": lambda img: ImageEnhance.Sharpness(img).enhance(0.50), "jpeg_quality": None},
        {"stress_type": "sharpness", "stress_level": "higher", "stress_parameter": "factor=2.00", "fn": lambda img: ImageEnhance.Sharpness(img).enhance(2.00), "jpeg_quality": None},
        {"stress_type": "contrast", "stress_level": "lower", "stress_parameter": "factor=0.70", "fn": lambda img: ImageEnhance.Contrast(img).enhance(0.70), "jpeg_quality": None},
        {"stress_type": "contrast", "stress_level": "higher", "stress_parameter": "factor=1.30", "fn": lambda img: ImageEnhance.Contrast(img).enhance(1.30), "jpeg_quality": None},
    ]

    for quality in JPEG_QUALITY_LEVELS:
        specs.append(
            {
                "stress_type": "jpeg_compression",
                "stress_level": f"q{quality}",
                "stress_parameter": f"quality={quality}",
                "fn": lambda img: img.convert("RGB"),
                "jpeg_quality": quality,
            }
        )

    return specs


def normal_variant_id(parent_image_id: str, target_dimension: str | int, resize_mode: str) -> str:
    return "normal_" + variant_digest([parent_image_id, "normal", target_dimension, resize_mode])


def stress_variant_id(parent_image_id: str, target_dimension: int, resize_mode: str, stress_type: str, stress_level: str, stress_parameter: str) -> str:
    return "stress_" + variant_digest([parent_image_id, "stress", target_dimension, resize_mode, stress_type, stress_level, stress_parameter])


def make_empty_row() -> dict[str, Any]:
    return {column: "" for column in CENTRAL_COLUMNS}


def build_normal_plan(images: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, image in images.iterrows():
        if image.get("error", ""):
            row = make_empty_row()
            for key, value in image.to_dict().items():
                if key in row:
                    row[key] = value
            row["test_type"] = "normal"
            row["error"] = image["error"]
            rows.append(row)
            continue

        for target_dimension in NORMAL_TARGET_DIMENSIONS:
            resize_modes = [""] if target_dimension == "original" else RESIZE_MODES
            for resize_mode in resize_modes:
                row = make_empty_row()
                row.update(
                    {
                        "variant_id": normal_variant_id(image["parent_image_id"], target_dimension, resize_mode),
                        "parent_image_id": image["parent_image_id"],
                        "original_image_id": image["original_image_id"],
                        "source": image["source"],
                        "source_subgroup": image["source_subgroup"],
                        "test_type": "normal",
                        "actual_label": "real",
                        "original_path": image["original_path"],
                        "variant_path": image["original_path"] if target_dimension == "original" else "",
                        "original_file_name": image["original_file_name"],
                        "variant_file_name": "",
                        "original_file_format": image["original_file_format"],
                        "variant_file_format": image["original_file_format"] if target_dimension == "original" else "",
                        "original_width": image["original_width"],
                        "original_height": image["original_height"],
                        "original_megapixels": image["original_megapixels"],
                        "target_dimension": target_label(target_dimension),
                        "resize_mode": resize_mode,
                        "source_fps": image["source_fps"],
                        "stress_type": "",
                        "stress_level": "",
                        "stress_parameter": "",
                        "model_name": MODEL_NAME,
                        "threshold": THRESHOLD,
                        "random_seed": RANDOM_SEED,
                        "error": "",
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows, columns=CENTRAL_COLUMNS)


def select_stress_images(completed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    normal_original = completed[
        (completed["test_type"] == "normal")
        & (completed["target_dimension"].astype(str) == "original")
        & (completed["error"].fillna("") == "")
        & (completed["normal_original_fake_probability"].notna())
    ].copy()

    if normal_original.empty:
        raise RuntimeError("Cannot select stress images before normal original rows are completed.")

    normal_original["normal_result"] = normal_original["normal_prediction"].map(result_from_prediction)
    fp = normal_original[normal_original["normal_result"] == "FP"].copy()
    tn = normal_original[normal_original["normal_result"] == "TN"].copy()

    selected_fp = fp.sort_values(
        "normal_original_fake_probability",
        ascending=False,
    ).reset_index(drop=True)

    selected_tn = select_balanced_tn_for_stress(tn, STRESS_SAMPLE_COUNT)

    return selected_tn, selected_fp


def select_balanced_tn_for_stress(tn: pd.DataFrame, sample_count: int) -> pd.DataFrame:
    if tn.empty or sample_count <= 0:
        return tn.head(0).copy()

    tn_sorted_high = tn.sort_values("normal_original_fake_probability", ascending=False).copy()
    tn_sorted_low = tn.sort_values("normal_original_fake_probability", ascending=True).copy()
    tn_sorted_mid = tn.assign(
        _distance_to_median=(
            tn["normal_original_fake_probability"]
            - tn["normal_original_fake_probability"].median()
        ).abs()
    ).sort_values(["_distance_to_median", "normal_original_fake_probability"], ascending=[True, False])

    target = min(sample_count, len(tn))
    high_quota = target // 3
    mid_quota = target // 3
    low_quota = target - high_quota - mid_quota

    selected_parts: list[pd.DataFrame] = []
    selected_ids: set[str] = set()

    for frame, quota in [
        (tn_sorted_high, high_quota),
        (tn_sorted_mid, mid_quota),
        (tn_sorted_low, low_quota),
    ]:
        rows = []
        for _, row in frame.iterrows():
            parent_id = str(row["parent_image_id"])
            if parent_id in selected_ids:
                continue
            rows.append(row.drop(labels=["_distance_to_median"], errors="ignore"))
            selected_ids.add(parent_id)
            if len(rows) >= quota:
                break
        if rows:
            selected_parts.append(pd.DataFrame(rows))

    if len(selected_ids) < target:
        for _, row in tn_sorted_high.iterrows():
            parent_id = str(row["parent_image_id"])
            if parent_id in selected_ids:
                continue
            selected_parts.append(pd.DataFrame([row]))
            selected_ids.add(parent_id)
            if len(selected_ids) >= target:
                break

    return pd.concat(selected_parts, ignore_index=True).head(target)


def build_stress_plan(selected_tn: pd.DataFrame, selected_fp: pd.DataFrame) -> pd.DataFrame:
    selected = pd.concat([selected_tn, selected_fp], ignore_index=True)
    specs = stress_specs()
    rows: list[dict[str, Any]] = []

    for _, image in selected.iterrows():
        for target_dimension in STRESS_TARGET_DIMENSIONS:
            resize_modes = [""] if target_dimension == "original" else RESIZE_MODES
            for resize_mode in resize_modes:
                for spec in specs:
                    row = make_empty_row()
                    row.update(
                        {
                            "variant_id": stress_variant_id(
                                image["parent_image_id"],
                                target_dimension,
                                resize_mode,
                                spec["stress_type"],
                                spec["stress_level"],
                                spec["stress_parameter"],
                            ),
                            "parent_image_id": image["parent_image_id"],
                            "original_image_id": image["original_image_id"],
                            "source": image["source"],
                            "source_subgroup": image["source_subgroup"],
                            "test_type": "stress",
                            "actual_label": "real",
                            "original_path": image["original_path"],
                            "variant_path": "",
                            "original_file_name": image["original_file_name"],
                            "variant_file_name": "",
                            "original_file_format": image["original_file_format"],
                            "variant_file_format": "",
                            "original_width": image["original_width"],
                            "original_height": image["original_height"],
                            "original_megapixels": image["original_megapixels"],
                            "target_dimension": target_label(target_dimension),
                            "resize_mode": resize_mode,
                            "source_fps": image["source_fps"],
                            "stress_type": spec["stress_type"],
                            "stress_level": spec["stress_level"],
                            "stress_parameter": spec["stress_parameter"],
                            "model_name": MODEL_NAME,
                            "threshold": THRESHOLD,
                            "random_seed": RANDOM_SEED,
                            "error": "",
                        }
                    )
                    rows.append(row)

    return pd.DataFrame(rows, columns=CENTRAL_COLUMNS)


# ============================================================
# CHECKPOINT / RESUME
# ============================================================

def read_existing_results() -> pd.DataFrame:
    if not MAIN_CSV.exists():
        return pd.DataFrame(columns=CENTRAL_COLUMNS)
    df = pd.read_csv(MAIN_CSV)
    for column in CENTRAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[CENTRAL_COLUMNS].astype("object")


def merge_new_plan(existing: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return plan.copy()

    existing_ids = set(existing["variant_id"].astype(str))
    new_rows = plan[~plan["variant_id"].astype(str).isin(existing_ids)].copy()
    if new_rows.empty:
        return existing.copy()

    return pd.concat([existing, new_rows], ignore_index=True)[CENTRAL_COLUMNS]


NUMERIC_MUTABLE_COLUMNS = [
    "original_width",
    "original_height",
    "original_megapixels",
    "variant_width",
    "variant_height",
    "variant_megapixels",
    "file_size_mb",
    "brightness_mean",
    "contrast_std",
    "blur_score",
    "sharpness_score",
    "source_fps",
    "threshold",
    "normal_original_fake_probability",
    "normal_original_real_probability",
    "normal_same_resolution_fake_probability",
    "normal_same_resolution_real_probability",
    "stress_fake_probability",
    "stress_real_probability",
    "score_delta_resolution",
    "score_delta_vs_clean",
    "score_delta_vs_original",
    "inference_ms",
    "random_seed",
    "stats_group_count",
    "stats_group_false_positives",
    "stats_group_false_positive_rate",
    "stats_group_score_mean",
    "stats_group_score_median",
    "stats_group_score_std",
    "stats_group_score_min",
    "stats_group_score_max",
    "stats_group_score_p95",
]


TEXT_MUTABLE_COLUMNS = [
    "variant_path",
    "variant_file_name",
    "variant_file_format",
    "resize_mode",
    "resolution_bucket",
    "brightness_bucket",
    "blur_bucket",
    "normal_prediction",
    "normal_same_resolution_variant_id",
    "stress_prediction",
    "normal_result",
    "stress_result",
    "prediction_transition",
    "stats_group_key",
    "error",
]


def normalize_runtime_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in CENTRAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    for column in NUMERIC_MUTABLE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")

    for column in TEXT_MUTABLE_COLUMNS:
        df[column] = df[column].astype("object")

    return df[CENTRAL_COLUMNS]


def add_group_statistics(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_runtime_dtypes(df)

    stat_text_cols = [
        "stats_group_key",
    ]
    stat_numeric_cols = [
        "stats_group_count",
        "stats_group_false_positives",
        "stats_group_false_positive_rate",
        "stats_group_score_mean",
        "stats_group_score_median",
        "stats_group_score_std",
        "stats_group_score_min",
        "stats_group_score_max",
        "stats_group_score_p95",
    ]
    for column in stat_text_cols:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].astype("object")

    for column in stat_numeric_cols:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")

    if df.empty:
        return df[CENTRAL_COLUMNS]

    normal_mask = (
        (df["test_type"] == "normal")
        & (df["error"].fillna("") == "")
        & (df["normal_same_resolution_fake_probability"].notna())
    )
    stress_mask = (
        (df["test_type"] == "stress")
        & (df["error"].fillna("") == "")
        & (df["stress_fake_probability"].notna())
    )

    for mask, score_col, result_col in [
        (normal_mask, "normal_same_resolution_fake_probability", "normal_result"),
        (stress_mask, "stress_fake_probability", "stress_result"),
    ]:
        subset = df[mask].copy()
        if subset.empty:
            continue

        subset["_score_for_stats"] = pd.to_numeric(subset[score_col], errors="coerce")
        group_cols = ["test_type", "source", "target_dimension", "resize_mode"]
        if result_col == "stress_result":
            group_cols = ["test_type", "source", "stress_type", "stress_level", "target_dimension", "resize_mode"]

        grouped = subset.groupby(group_cols, dropna=False)
        for keys, indexes in grouped.groups.items():
            if not isinstance(keys, tuple):
                keys = (keys,)
            group = subset.loc[indexes]
            scores = group["_score_for_stats"].dropna()
            count = int(len(group))
            fp_count = int((group[result_col] == "FP").sum())
            key_text = "|".join(str(value) for value in keys)

            df.loc[indexes, "stats_group_key"] = key_text
            df.loc[indexes, "stats_group_count"] = count
            df.loc[indexes, "stats_group_false_positives"] = fp_count
            df.loc[indexes, "stats_group_false_positive_rate"] = fp_count / count if count else np.nan
            df.loc[indexes, "stats_group_score_mean"] = scores.mean()
            df.loc[indexes, "stats_group_score_median"] = scores.median()
            df.loc[indexes, "stats_group_score_std"] = scores.std(ddof=0)
            df.loc[indexes, "stats_group_score_min"] = scores.min()
            df.loc[indexes, "stats_group_score_max"] = scores.max()
            df.loc[indexes, "stats_group_score_p95"] = scores.quantile(0.95)

    return df[CENTRAL_COLUMNS]


def sort_central_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    test_order = {"normal": 0, "stress": 1}
    dimension_order = {"original": 0, "1024": 1, "720": 2, "512": 3, "256": 4}
    resize_order = {"": 0, "aspect": 1, "square": 2}

    df["_test_order"] = df["test_type"].map(test_order).fillna(99)
    df["_dimension_order"] = df["target_dimension"].astype(str).map(dimension_order).fillna(99)
    df["_resize_order"] = df["resize_mode"].fillna("").astype(str).map(resize_order).fillna(99)
    df["_score_sort"] = np.where(
        df["test_type"] == "normal",
        pd.to_numeric(df["normal_same_resolution_fake_probability"], errors="coerce"),
        pd.to_numeric(df["stress_fake_probability"], errors="coerce"),
    )

    sorted_df = df.sort_values(
        [
            "_test_order",
            "source",
            "_dimension_order",
            "_resize_order",
            "stress_type",
            "stress_level",
            "_score_sort",
            "original_file_name",
            "variant_id",
        ],
        ascending=[True, True, True, True, True, True, False, True, True],
        na_position="last",
    )

    return sorted_df.drop(columns=["_test_order", "_dimension_order", "_resize_order", "_score_sort"])[CENTRAL_COLUMNS]


def save_checkpoint(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    enriched = add_group_statistics(normalize_runtime_dtypes(df))
    sorted_df = sort_central_rows(enriched)
    sorted_df.to_csv(MAIN_CSV, index=False)


def write_column_guide() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "FALSE POSITIVE COMPLETE BENCHMARK CSV COLUMN GUIDE",
        "=" * 64,
        "",
        f"CSV file: {MAIN_CSV}",
        "",
        "Real-only result meaning:",
        "TN = real image predicted real",
        "FP = real image predicted fake",
        "",
        "Column definitions:",
        "",
    ]

    for column in CENTRAL_COLUMNS:
        lines.append(f"{column}: {COLUMN_DESCRIPTIONS.get(column, 'No description available.')}")

    COLUMN_GUIDE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def is_completed(row: pd.Series) -> bool:
    error = str(row.get("error", "") or "")
    if error:
        return True
    test_type = row.get("test_type", "")
    if test_type == "normal":
        return not pd.isna(safe_float(row.get("normal_same_resolution_fake_probability", np.nan)))
    if test_type == "stress":
        return not pd.isna(safe_float(row.get("stress_fake_probability", np.nan)))
    return False


# ============================================================
# VARIANT GENERATION
# ============================================================

def generated_path(row: pd.Series, actual_width: int, actual_height: int, extension: str) -> Path:
    source = clean_name(row["source"])
    target = clean_name(row["target_dimension"])
    resize_mode = clean_name(row.get("resize_mode", ""))
    file_name = logical_variant_file_name(row, actual_width, actual_height, extension)

    if row["test_type"] == "normal":
        return GENERATED_VARIANT_DIR / "normal" / source / target / resize_mode / file_name

    stress_type = clean_name(row["stress_type"])
    stress_level = clean_name(row["stress_level"])
    return GENERATED_VARIANT_DIR / "stress" / source / stress_type / stress_level / target / resize_mode / file_name


def logical_variant_file_name(row: pd.Series, actual_width: int, actual_height: int, extension: str) -> str:
    source = clean_name(row["source"])
    target = clean_name(row["target_dimension"])
    resize_mode = clean_name(row.get("resize_mode", ""))
    index = stable_numeric_index(str(row["variant_id"]))

    if row["test_type"] == "normal":
        mode_part = f"{resize_mode}_" if resize_mode else ""
        return f"normal_{source}_{target}_{mode_part}{actual_width}x{actual_height}_{index}{extension}"

    stress_type = clean_name(row["stress_type"])
    stress_level = clean_name(row["stress_level"])
    mode_part = f"{resize_mode}_" if resize_mode else ""
    file_name = (
        f"stress_{source}_{stress_type}_{stress_level}_{target}_{mode_part}{actual_width}x{actual_height}_"
        f"{index}{extension}"
    )
    return file_name


def apply_stress(image: Image.Image, stress_type_value: str, stress_level: str) -> tuple[Image.Image, int | None]:
    if stress_type_value == "blur":
        radius = {"low": 1, "medium": 2, "high": 4}[stress_level]
        return image.filter(ImageFilter.GaussianBlur(radius=radius)), None

    if stress_type_value == "brightness":
        factor = {"darker": 0.70, "brighter": 1.30}[stress_level]
        return ImageEnhance.Brightness(image).enhance(factor), None

    if stress_type_value == "sharpness":
        factor = {"lower": 0.50, "higher": 2.00}[stress_level]
        return ImageEnhance.Sharpness(image).enhance(factor), None

    if stress_type_value == "contrast":
        factor = {"lower": 0.70, "higher": 1.30}[stress_level]
        return ImageEnhance.Contrast(image).enhance(factor), None

    if stress_type_value == "jpeg_compression":
        quality = int(str(stress_level).replace("q", ""))
        return image.convert("RGB"), quality

    raise ValueError(f"Unknown stress_type: {stress_type_value}")


def prepare_variant(row: pd.Series, temp_dir: Path) -> tuple[Image.Image | str | Path | None, dict[str, Any], Path | None]:
    source_path = Path(row["original_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Source image missing: {source_path}")
    if not is_supported_image_path(source_path):
        raise ValueError(f"Unsupported image extension, only png/jpg/jpeg allowed: {source_path}")

    with open_source_image(source_path) as source_image:
        variant_image = resize_for_mode(
            source_image,
            row["target_dimension"],
            str(row.get("resize_mode", "")),
        )

        jpeg_quality: int | None = None
        if row["test_type"] == "stress":
            variant_image, jpeg_quality = apply_stress(
                variant_image,
                str(row["stress_type"]),
                str(row["stress_level"]),
            )

        stats = image_stats(variant_image)

        if row["test_type"] == "normal" and row["target_dimension"] == "original":
            stats["variant_path"] = ""
            stats["variant_file_name"] = ""
            stats["variant_file_format"] = ""
            stats["file_size_mb"] = file_size_mb(source_path)
            return str(source_path), stats, None

        if jpeg_quality is not None:
            save_format, extension, save_kwargs = jpeg_save_format(jpeg_quality)
        else:
            save_format, extension, save_kwargs = source_save_format(source_path)

        if KEEP_GENERATED_VARIANTS:
            out_path = generated_path(
                row,
                stats["variant_width"],
                stats["variant_height"],
                extension,
            )
        else:
            out_path = temp_dir / logical_variant_file_name(
                row,
                stats["variant_width"],
                stats["variant_height"],
                extension,
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)

        save_image = variant_image
        if save_format == "JPEG" and save_image.mode == "RGBA":
            save_image = save_image.convert("RGB")
        save_image.save(out_path, format=save_format, **save_kwargs)

        stats["variant_path"] = str(out_path) if KEEP_GENERATED_VARIANTS else ""
        stats["variant_file_name"] = out_path.name
        stats["variant_file_format"] = extension.lstrip(".")
        stats["file_size_mb"] = file_size_mb(out_path)
        return str(out_path), stats, None if KEEP_GENERATED_VARIANTS else out_path


# ============================================================
# INFERENCE
# ============================================================

def load_classifier():
    print("\nLoading Hugging Face image-classification pipeline...")
    classifier = pipeline(
        "image-classification",
        model=MODEL_NAME,
        device=DEVICE,
    )
    print(f"Model       : {MODEL_NAME}")
    print(f"Device      : {DEVICE}")
    print(f"num_labels  : {classifier.model.config.num_labels}")
    print(f"id2label    : {classifier.model.config.id2label}")
    print(f"torch_dtype : {getattr(classifier.model.config, 'torch_dtype', '')}")
    return classifier


def update_row_with_prediction(df: pd.DataFrame, index: int, score: float, inference_ms: float) -> None:
    real_score = 1.0 - score
    prediction = prediction_from_score(score)
    result = result_from_prediction(prediction)

    if df.at[index, "test_type"] == "normal":
        df.at[index, "normal_same_resolution_fake_probability"] = score
        df.at[index, "normal_same_resolution_real_probability"] = real_score
        df.at[index, "normal_same_resolution_variant_id"] = df.at[index, "variant_id"]
        df.at[index, "normal_prediction"] = prediction
        df.at[index, "normal_result"] = result
        if str(df.at[index, "target_dimension"]) == "original":
            df.at[index, "normal_original_fake_probability"] = score
            df.at[index, "normal_original_real_probability"] = real_score
    else:
        df.at[index, "stress_fake_probability"] = score
        df.at[index, "stress_real_probability"] = real_score
        df.at[index, "stress_prediction"] = prediction
        df.at[index, "stress_result"] = result

    df.at[index, "inference_ms"] = inference_ms
    df.at[index, "error"] = ""


def export_failed_variant(row: pd.Series, tested_path: Path | str | None) -> None:
    if tested_path is None:
        return

    test_type = str(row.get("test_type", ""))
    if test_type == "normal":
        result = str(row.get("normal_result", ""))
        score = safe_float(row.get("normal_same_resolution_fake_probability", np.nan))
    else:
        result = str(row.get("stress_result", ""))
        score = safe_float(row.get("stress_fake_probability", np.nan))

    if result != "FP":
        return

    source_path = Path(tested_path)
    if not source_path.exists():
        return

    source = clean_name(row.get("source", "unknown"))
    target = clean_name(row.get("target_dimension", "unknown"))
    resize_mode = clean_name(row.get("resize_mode", ""))

    if test_type == "normal":
        export_dir = FAILED_EXPORT_DIR / "normal" / source / target / resize_mode
    else:
        stress_type = clean_name(row.get("stress_type", "unknown"))
        stress_level = clean_name(row.get("stress_level", "unknown"))
        export_dir = FAILED_EXPORT_DIR / "stress" / source / stress_type / stress_level / target / resize_mode

    export_dir.mkdir(parents=True, exist_ok=True)
    variant_name = str(row.get("variant_file_name", "")) or source_path.name
    destination = export_dir / (
        f"score_{score:.4f}_{clean_name(row.get('variant_id', 'variant'))}_"
        f"{clean_name(variant_name)}"
    )
    shutil.copy2(source_path, destination)


def run_inference_for_pending(df: pd.DataFrame, classifier, test_type: str) -> pd.DataFrame:
    df = normalize_runtime_dtypes(df)
    pending_indexes = [
        index
        for index, row in df[df["test_type"] == test_type].iterrows()
        if not is_completed(row)
    ]

    if not pending_indexes:
        print(f"No pending {test_type} variants.")
        return df

    total_batches = math.ceil(len(pending_indexes) / BATCH_SIZE)
    completed_batches = 0

    for start in tqdm(
        range(0, len(pending_indexes), BATCH_SIZE),
        total=total_batches,
        desc=f"Inference {test_type}",
        unit="batch",
    ):
        with tempfile.TemporaryDirectory(prefix=f"{test_type}_batch_") as tmp:
            temp_dir = Path(tmp)
            batch_indexes = pending_indexes[start : start + BATCH_SIZE]
            model_inputs: list[Any] = []
            prepared: list[tuple[int, Path | None]] = []

            for index in batch_indexes:
                try:
                    model_input, stats, temp_path = prepare_variant(df.loc[index], temp_dir)
                    for key, value in stats.items():
                        if key in df.columns:
                            df.at[index, key] = value
                    model_inputs.append(model_input)
                    prepared.append((index, temp_path))
                except Exception as error:
                    df.at[index, "error"] = f"variant_prepare_error: {error}"

            if model_inputs:
                try:
                    start_time = time.perf_counter()
                    outputs = classifier(
                        model_inputs,
                        batch_size=BATCH_SIZE,
                        top_k=1,
                        function_to_apply="sigmoid",
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    per_image_ms = elapsed_ms / len(model_inputs)

                    for (index, temp_path), output in zip(prepared, outputs):
                        if isinstance(output, list):
                            output = output[0]
                        score = float(output["score"])
                        update_row_with_prediction(df, index, score, per_image_ms)
                        tested_path = temp_path or Path(df.at[index, "original_path"])
                        export_failed_variant(df.loc[index], tested_path)
                        if temp_path and temp_path.exists() and not KEEP_GENERATED_VARIANTS:
                            temp_path.unlink()
                except Exception as error:
                    for index, temp_path in prepared:
                        df.at[index, "error"] = f"inference_error: {error}"
                        if temp_path and temp_path.exists() and not KEEP_GENERATED_VARIANTS:
                            temp_path.unlink()

        completed_batches += 1
        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        if completed_batches % CHECKPOINT_EVERY == 0:
            save_checkpoint(df)

    save_checkpoint(df)
    return df


# ============================================================
# POSTPROCESS DELTAS
# ============================================================

def fill_normal_reference_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_runtime_dtypes(df)

    normal_rows = df[
        (df["test_type"] == "normal")
        & (df["error"].fillna("") == "")
        & (df["normal_same_resolution_fake_probability"].notna())
    ].copy()

    original_scores = (
        normal_rows[normal_rows["target_dimension"].astype(str) == "original"]
        .set_index("parent_image_id")
        [["variant_id", "normal_same_resolution_fake_probability", "normal_same_resolution_real_probability", "normal_prediction", "normal_result"]]
        .rename(
            columns={
                "variant_id": "orig_variant_id",
                "normal_same_resolution_fake_probability": "orig_fake",
                "normal_same_resolution_real_probability": "orig_real",
                "normal_prediction": "orig_prediction",
                "normal_result": "orig_result",
            }
        )
    )

    if not normal_rows.empty:
        normal_rows["target_dimension"] = normal_rows["target_dimension"].astype(str)
        normal_rows["resize_mode"] = normal_rows["resize_mode"].fillna("").astype(str)

    same_resolution_lookup = normal_rows.drop_duplicates(
        ["parent_image_id", "target_dimension", "resize_mode"],
        keep="last",
    )
    same_resolution_scores = (
        same_resolution_lookup.set_index(["parent_image_id", "target_dimension", "resize_mode"])
        [["variant_id", "normal_same_resolution_fake_probability", "normal_same_resolution_real_probability", "normal_prediction", "normal_result"]]
        .rename(
            columns={
                "variant_id": "same_variant_id",
                "normal_same_resolution_fake_probability": "same_fake",
                "normal_same_resolution_real_probability": "same_real",
                "normal_prediction": "same_prediction",
                "normal_result": "same_result",
            }
        )
    )

    for index, row in df.iterrows():
        parent_id = row["parent_image_id"]
        target_dimension = str(row["target_dimension"])
        resize_mode = str(row.get("resize_mode", "") or "")

        if parent_id in original_scores.index:
            orig = original_scores.loc[parent_id]
            df.at[index, "normal_original_fake_probability"] = safe_float(orig["orig_fake"])
            df.at[index, "normal_original_real_probability"] = safe_float(orig["orig_real"])

        same_key = (parent_id, target_dimension, resize_mode)
        if same_key in same_resolution_scores.index:
            same = same_resolution_scores.loc[same_key]
            df.at[index, "normal_same_resolution_fake_probability"] = safe_float(same["same_fake"])
            df.at[index, "normal_same_resolution_real_probability"] = safe_float(same["same_real"])
            df.at[index, "normal_same_resolution_variant_id"] = str(same["same_variant_id"])
            if row["test_type"] == "stress":
                clean_result = str(same["same_result"])
                stress_result = str(row.get("stress_result", ""))
                df.at[index, "prediction_transition"] = transition(clean_result, stress_result)

        normal_original = safe_float(df.at[index, "normal_original_fake_probability"])
        normal_same = safe_float(df.at[index, "normal_same_resolution_fake_probability"])
        stress_score = safe_float(df.at[index, "stress_fake_probability"])

        if not pd.isna(normal_original) and not pd.isna(normal_same):
            df.at[index, "score_delta_resolution"] = normal_same - normal_original

        if row["test_type"] == "stress":
            if not pd.isna(stress_score) and not pd.isna(normal_same):
                df.at[index, "score_delta_vs_clean"] = stress_score - normal_same
            if not pd.isna(stress_score) and not pd.isna(normal_original):
                df.at[index, "score_delta_vs_original"] = stress_score - normal_original

    return df[CENTRAL_COLUMNS]


# ============================================================
# REPORTS
# ============================================================

def summarize_group(df: pd.DataFrame, group_cols: list[str], score_col: str, result_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = df[
        (df["error"].fillna("") == "")
        & (df[score_col].notna())
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    for keys, group in valid.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        scores = group[score_col].dropna().astype(float)
        fp_count = int((group[result_col] == "FP").sum())
        count = int(len(group))

        row: dict[str, Any] = {
            "count": count,
            "false_positives": fp_count,
            "false_positive_rate": fp_count / count if count else np.nan,
            "true_negative_rate": 1 - (fp_count / count) if count else np.nan,
            "score_mean": scores.mean(),
            "score_median": scores.median(),
            "score_std": scores.std(ddof=0),
            "score_min": scores.min(),
            "score_max": scores.max(),
            "score_p75": scores.quantile(0.75),
            "score_p90": scores.quantile(0.90),
            "score_p95": scores.quantile(0.95),
            "score_p99": scores.quantile(0.99),
        }

        for col, value in zip(group_cols, keys):
            row[col] = value

        rows.append(row)

    leading_cols = group_cols + [
        "count",
        "false_positives",
        "false_positive_rate",
        "true_negative_rate",
        "score_mean",
        "score_median",
        "score_std",
        "score_min",
        "score_max",
        "score_p75",
        "score_p90",
        "score_p95",
        "score_p99",
    ]
    return pd.DataFrame(rows)[leading_cols].sort_values(
        ["false_positive_rate", "false_positives", "count"],
        ascending=[False, False, False],
    )


def create_normal_summary(df: pd.DataFrame) -> pd.DataFrame:
    normal = df[df["test_type"] == "normal"].copy()
    parts: list[pd.DataFrame] = []
    for group_cols in [
        ["target_dimension"],
        ["target_dimension", "resize_mode"],
        ["source"],
        ["source", "target_dimension", "resize_mode"],
        ["resolution_bucket"],
        ["brightness_bucket"],
        ["blur_bucket"],
        ["variant_file_format"],
        ["resize_mode"],
    ]:
        part = summarize_group(normal, group_cols, "normal_same_resolution_fake_probability", "normal_result")
        if not part.empty:
            part.insert(0, "summary_group", "+".join(group_cols))
            parts.append(part)

    summary = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return summary


def create_stress_summary(df: pd.DataFrame) -> pd.DataFrame:
    stress = df[df["test_type"] == "stress"].copy()
    parts: list[pd.DataFrame] = []
    for group_cols in [
        ["stress_type", "stress_level"],
        ["stress_type", "stress_level", "target_dimension", "resize_mode"],
        ["source", "stress_type", "stress_level"],
        ["prediction_transition"],
        ["target_dimension", "resize_mode"],
        ["resize_mode"],
    ]:
        part = summarize_group(stress, group_cols, "stress_fake_probability", "stress_result")
        if not part.empty:
            part.insert(0, "summary_group", "+".join(group_cols))
            if "score_delta_vs_clean" in stress.columns:
                delta = (
                    stress.groupby(group_cols, dropna=False)["score_delta_vs_clean"]
                    .mean()
                    .reset_index()
                    .rename(columns={"score_delta_vs_clean": "mean_score_delta_vs_clean"})
                )
                part = part.merge(delta, on=group_cols, how="left")
            parts.append(part)

    summary = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return summary


def create_threshold_sweep(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[
        (df["error"].fillna("") == "")
        & (
            ((df["test_type"] == "normal") & df["normal_same_resolution_fake_probability"].notna())
            | ((df["test_type"] == "stress") & df["stress_fake_probability"].notna())
        )
    ].copy()

    rows: list[dict[str, Any]] = []
    for test_type, group in valid.groupby("test_type", dropna=False):
        score_col = "normal_same_resolution_fake_probability" if test_type == "normal" else "stress_fake_probability"
        scores = group[score_col].astype(float)
        for threshold in np.arange(0.05, 1.00, 0.05):
            predicted_fake = scores >= threshold
            count = int(len(group))
            false_positives = int(predicted_fake.sum())
            rows.append(
                {
                    "test_type": test_type,
                    "threshold": round(float(threshold), 2),
                    "real_images": count,
                    "false_positives": false_positives,
                    "false_positive_rate": false_positives / count if count else np.nan,
                    "true_negative_rate": 1 - (false_positives / count) if count else np.nan,
                }
            )

    result = pd.DataFrame(rows)
    return result


def create_errors(df: pd.DataFrame) -> pd.DataFrame:
    errors = df[df["error"].fillna("") != ""].copy()
    return errors


def write_text_report(
    df: pd.DataFrame,
    normal_summary: pd.DataFrame,
    stress_summary: pd.DataFrame,
    threshold_sweep: pd.DataFrame,
    errors: pd.DataFrame,
) -> None:
    normal = df[
        (df["test_type"] == "normal")
        & (df["error"].fillna("") == "")
        & (df["normal_same_resolution_fake_probability"].notna())
    ]
    stress = df[
        (df["test_type"] == "stress")
        & (df["error"].fillna("") == "")
        & (df["stress_fake_probability"].notna())
    ]

    lines = [
        "REAL-ONLY COMPLETE FALSE-POSITIVE BENCHMARK",
        "=" * 80,
        f"Central CSV    : {MAIN_CSV}",
        f"Failed images  : {FAILED_EXPORT_DIR}",
        f"Normal rows    : {len(normal):,}",
        f"Stress rows    : {len(stress):,}",
        f"Error rows     : {len(errors):,}",
    ]

    if not normal.empty:
        normal_fp = int((normal["normal_result"] == "FP").sum())
        lines.append(f"Normal FP rate : {normal_fp:,}/{len(normal):,} = {normal_fp / len(normal):.2%}")

    if not stress.empty:
        stress_fp = int((stress["stress_result"] == "FP").sum())
        lines.append(f"Stress FP rate : {stress_fp:,}/{len(stress):,} = {stress_fp / len(stress):.2%}")

    lines.extend(["", "Top normal summary rows:"])
    if normal_summary.empty:
        lines.append("No normal summary rows.")
    else:
        lines.append(normal_summary.head(20).to_string(index=False))

    lines.extend(["", "Top stress summary rows:"])
    if stress_summary.empty:
        lines.append("No stress summary rows.")
    else:
        lines.append(stress_summary.head(30).to_string(index=False))

    lines.extend(["", "Threshold sweep preview:"])
    if threshold_sweep.empty:
        lines.append("No threshold sweep rows.")
    else:
        lines.append(threshold_sweep.head(40).to_string(index=False))

    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_report(df: pd.DataFrame) -> None:
    normal = df[
        (df["test_type"] == "normal")
        & (df["error"].fillna("") == "")
        & (df["normal_same_resolution_fake_probability"].notna())
    ]
    stress = df[
        (df["test_type"] == "stress")
        & (df["error"].fillna("") == "")
        & (df["stress_fake_probability"].notna())
    ]
    errors = df[df["error"].fillna("") != ""]

    print("\n" + "=" * 80)
    print("REAL-ONLY COMPLETE FALSE-POSITIVE BENCHMARK")
    print("=" * 80)
    print(f"Only CSV       : {MAIN_CSV}")
    print(f"Failed images  : {FAILED_EXPORT_DIR}")
    print(f"Text summary   : {SUMMARY_TXT}")
    print(f"Column guide   : {COLUMN_GUIDE_TXT}")
    print(f"Normal rows    : {len(normal):,}")
    print(f"Stress rows    : {len(stress):,}")
    print(f"Error rows     : {len(errors):,}")

    if not normal.empty:
        normal_fp = int((normal["normal_result"] == "FP").sum())
        print(f"Normal FP rate : {normal_fp:,}/{len(normal):,} = {normal_fp / len(normal):.2%}")

    if not stress.empty:
        stress_fp = int((stress["stress_result"] == "FP").sum())
        print(f"Stress FP rate : {stress_fp:,}/{len(stress):,} = {stress_fp / len(stress):.2%}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()
    ensure_runtime_dependencies()
    configure_from_args(args)
    ensure_dirs()
    write_column_guide()

    print("=" * 80)
    print("COMPLETE REAL-ONLY DEEPFAKE FALSE-POSITIVE BENCHMARK")
    print("=" * 80)
    print(f"Data root              : {DATA_ROOT}")
    print(f"Discovered sources     : {', '.join(DISCOVERED_SOURCES) if DISCOVERED_SOURCES else '(none found)'}")
    print(f"Output dir             : {OUTPUT_DIR}")
    print(f"Failed image dir       : {FAILED_EXPORT_DIR}")
    print(f"Column guide           : {COLUMN_GUIDE_TXT}")
    print(f"Model                  : {MODEL_NAME}")
    print(f"Device                 : {DEVICE}")
    print(f"Threshold              : {THRESHOLD}")
    print(f"Batch size             : {BATCH_SIZE}")
    print(f"Random seed            : {RANDOM_SEED}")
    print(f"Resize modes           : {', '.join(RESIZE_MODES)}")
    print(f"Keep generated variants: {KEEP_GENERATED_VARIANTS}")

    existing = pd.DataFrame(columns=CENTRAL_COLUMNS)

    print("\n[1/7] Discovering source images...")
    if not DISCOVERED_SOURCES:
        raise RuntimeError(f"No image-containing source directories found under {DATA_ROOT}")
    images = discover_images()
    if images.empty:
        raise RuntimeError("No images found.")

    print(f"Total source images: {len(images):,}")

    print("\n[2/7] Building normal benchmark plan...")
    normal_plan = build_normal_plan(images)
    df = merge_new_plan(existing, normal_plan)
    df = normalize_runtime_dtypes(df)
    save_checkpoint(df)
    print(f"Planned/loaded rows after normal plan: {len(df):,}")

    print("\n[3/7] Loading classifier...")
    classifier = load_classifier()

    print("\n[4/7] Running normal benchmark...")
    df = run_inference_for_pending(df, classifier, "normal")
    df = fill_normal_reference_columns(df)
    save_checkpoint(df)

    print("\n[5/7] Selecting stress images and building stress plan...")
    selected_tn, selected_fp = select_stress_images(df)
    print(f"Selected TN/passed originals: {len(selected_tn):,}")
    print(f"Selected FP/failed originals: {len(selected_fp):,}")
    stress_plan = build_stress_plan(selected_tn, selected_fp)
    df = merge_new_plan(df, stress_plan)
    df = normalize_runtime_dtypes(df)
    save_checkpoint(df)
    print(f"Planned/loaded rows after stress plan: {len(df):,}")

    print("\n[6/7] Running stress benchmark...")
    df = run_inference_for_pending(df, classifier, "stress")
    df = fill_normal_reference_columns(df)
    save_checkpoint(df)

    print("\n[7/7] Creating reports...")
    normal_summary = create_normal_summary(df)
    stress_summary = create_stress_summary(df)
    threshold_sweep = create_threshold_sweep(df)
    errors = create_errors(df)
    write_text_report(df, normal_summary, stress_summary, threshold_sweep, errors)
    save_checkpoint(df)
    print_report(df)


if __name__ == "__main__":
    main()
