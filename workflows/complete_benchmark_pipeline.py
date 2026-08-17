#!/usr/bin/env python3
"""
Complete real/fake deepfake benchmark pipeline.

What this script does:
  - discovers real and fake image datasets dynamically
  - runs one or more Hugging Face image-classification models
  - saves each model into its own output/<model_acronym>/ directory
  - evaluates original + resized variants without modifying source images
  - writes one central prediction CSV per model
  - writes standard binary-classification metrics and threshold sweeps

Expected input layouts supported:
  1. /root/data/real/<source>/*.jpg
     /root/data/fake/<source>/*.jpg

  2. /root/real/<source>/*.jpg
     /root/fake/<source>/*.jpg

  3. /root/data/<source>/*.jpg
     If no real/fake label folders are found, all images are treated as real
     for backward compatibility with the old false-positive benchmark.

Source images are never edited, moved, deleted, compressed, or resized.
Generated resized variants are temporary by default.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_DATA_ROOT = Path("/Users/subrat/Desktop/Deepfake")
DEFAULT_OUTPUT_ROOT = Path.cwd() / "output"

DEFAULT_MODELS = [
    "buildborderless/CommunityForensics-DeepfakeDet-ViT=cf_vit",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

TARGET_DIMENSIONS: list[str | int] = ["original", 1024, 720, 512, 256]
RESIZE_MODES = ["aspect", "square"]

BATCH_SIZE = 8
CHECKPOINT_EVERY = 25
RANDOM_SEED = 42
THRESHOLD = 0.50
KEEP_GENERATED_VARIANTS = False
METADATA_WORKERS = 4

REAL_FOLDER_NAMES = {"real", "reals", "authentic", "genuine", "original", "camera_real"}
FAKE_FOLDER_NAMES = {"fake", "fakes", "deepfake", "deepfakes", "synthetic", "generated", "ai_generated", "ai"}

FAKE_LABEL_PATTERNS = [
    "fake",
    "deepfake",
    "synthetic",
    "generated",
    "manipulated",
    "ai",
    "spoof",
    "label_0",
]

IGNORED_DATA_DIRS = {
    ".git",
    ".agents",
    ".codex",
    ".deepeval",
    ".pycache",
    "__pycache__",
    "output",
    "archive",
    "workflows",
    "benchmark_results",
}
REAL_LABEL_PATTERNS = [
    "real",
    "authentic",
    "genuine",
    "original",
    "camera",
    "label_1",
]


# Optional dependencies are loaded at runtime so the script can print a useful
# message or install missing basics when run locally.
cv2 = None
np = None
pd = None
torch = None
Image = None
ImageOps = None
tqdm = None
pipeline = None
DEVICE: str | int = -1


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    acronym: str


@dataclass
class RunPaths:
    output_dir: Path
    generated_dir: Path
    failed_dir: Path
    main_csv: Path
    metrics_overall_csv: Path
    metrics_by_group_csv: Path
    threshold_sweep_csv: Path
    confusion_matrix_csv: Path
    errors_csv: Path
    summary_txt: Path
    columns_txt: Path


CENTRAL_COLUMNS = [
    "variant_id",
    "parent_image_id",
    "original_image_id",
    "source",
    "source_subgroup",
    "test_type",
    "actual_label",
    "actual_binary",
    "original_path",
    "original_file_name",
    "original_file_format",
    "original_width",
    "original_height",
    "original_megapixels",
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
    "source_fps",
    "model_name",
    "model_acronym",
    "threshold",
    "fake_probability",
    "real_probability",
    "prediction",
    "prediction_binary",
    "result",
    "is_correct",
    "inference_ms",
    "random_seed",
    "error",
]

NUMERIC_COLUMNS = [
    "actual_binary",
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
    "fake_probability",
    "real_probability",
    "prediction_binary",
    "is_correct",
    "inference_ms",
    "random_seed",
]

TEXT_COLUMNS = [column for column in CENTRAL_COLUMNS if column not in NUMERIC_COLUMNS]


COLUMN_DESCRIPTIONS = {
    "variant_id": "Stable id for one evaluated image variant.",
    "parent_image_id": "Stable id for the original image before resizing.",
    "original_image_id": "Stable hash id for the source image path.",
    "source": "Dataset/source folder name, for example camera, ffhq, or a fake dataset name.",
    "source_subgroup": "Nested subfolder under the source folder, if present.",
    "test_type": "clean for original/resized non-stress evaluation.",
    "actual_label": "Ground-truth label: real or fake.",
    "actual_binary": "Ground-truth binary value: 1=fake, 0=real.",
    "original_path": "Absolute source image path. The script only reads this file.",
    "original_file_name": "Source image filename.",
    "original_file_format": "Source extension without dot.",
    "original_width": "Original source image width.",
    "original_height": "Original source image height.",
    "original_megapixels": "Original source image megapixels.",
    "target_dimension": "original, 1024, 720, 512, or 256.",
    "resize_mode": "blank for original, aspect for aspect-preserving resize, square for center-crop square resize.",
    "variant_path": "Saved variant path only when --keep-generated-variants is enabled or when original is tested.",
    "variant_file_name": "Logical evaluated filename.",
    "variant_file_format": "Evaluated image extension without dot.",
    "variant_width": "Evaluated variant width.",
    "variant_height": "Evaluated variant height.",
    "variant_megapixels": "Evaluated variant megapixels.",
    "file_size_mb": "Disk size in MB of the exact evaluated file.",
    "resolution_bucket": "Resolution bucket based on minimum evaluated dimension.",
    "brightness_mean": "Mean grayscale brightness.",
    "brightness_bucket": "Bucket derived from brightness_mean.",
    "contrast_std": "Grayscale contrast standard deviation.",
    "blur_score": "Laplacian variance blur/sharpness score.",
    "blur_bucket": "Bucket derived from blur_score.",
    "sharpness_score": "Same Laplacian-based sharpness proxy as blur_score.",
    "source_fps": "FPS parsed from filename if pattern like _30fps_ exists.",
    "model_name": "Hugging Face model id.",
    "model_acronym": "Short output folder name for the model.",
    "threshold": "Decision threshold. fake_probability >= threshold predicts fake.",
    "fake_probability": "Model fake probability/score used for binary decision.",
    "real_probability": "1 - fake_probability.",
    "prediction": "Predicted label: real or fake.",
    "prediction_binary": "Predicted binary value: 1=fake, 0=real.",
    "result": "Binary classification outcome: TP, TN, FP, or FN.",
    "is_correct": "1 if prediction equals actual label, else 0.",
    "inference_ms": "Approximate inference milliseconds per image in the batch.",
    "random_seed": "Seed stored for reproducibility.",
    "error": "Error text if variant prep or inference failed.",
}


# ============================================================
# RUNTIME / ARGS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete real/fake deepfake benchmark pipeline.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="One or more models. Use model_id=acronym, for example buildborderless/...=cf_vit",
    )
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--keep-generated-variants",
        action="store_true",
        default=KEEP_GENERATED_VARIANTS,
        help="Persist resized variants under each model output directory.",
    )
    parser.add_argument(
        "--no-clean-model-output",
        action="store_true",
        help="Do not remove an existing output/<model_acronym>/ directory before running that model. By default, reruns overwrite that model directory.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Discover images and print planned variant counts, but do not load models or run inference.",
    )
    parser.add_argument(
        "--no-balance-labels",
        action="store_true",
        help="Use all discovered images instead of randomly downsampling each label to the smallest label count.",
    )
    return parser.parse_args()


def ensure_package(import_name: str, package_name: str | None = None) -> Any:
    try:
        return import_module(import_name)
    except ImportError:
        package = package_name or import_name
        os.system(f'"{os.sys.executable}" -m pip install -q {package}')
        return import_module(import_name)


def ensure_runtime_dependencies(need_model_runtime: bool = True) -> None:
    global cv2, np, pd, torch, Image, ImageOps, tqdm, pipeline, DEVICE

    np = ensure_package("numpy")
    pd = ensure_package("pandas")
    Image = ensure_package("PIL.Image", "pillow")
    ImageOps = ensure_package("PIL.ImageOps", "pillow")
    tqdm = ensure_package("tqdm").tqdm

    if not need_model_runtime:
        DEVICE = -1
        return

    cv2 = ensure_package("cv2", "opencv-python")
    torch = ensure_package("torch")
    pipeline = ensure_package("transformers").pipeline

    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = 0
    else:
        DEVICE = -1


def parse_model_specs(values: list[str]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    seen: set[str] = set()

    for value in values:
        if "=" in value:
            model_name, acronym = value.split("=", 1)
        else:
            model_name = value
            acronym = model_acronym_from_name(model_name)
        model_name = model_name.strip()
        acronym = clean_name(acronym.strip() or model_acronym_from_name(model_name)).lower()
        if not model_name:
            raise ValueError(f"Invalid empty model spec: {value}")
        if acronym in seen:
            raise ValueError(f"Duplicate model acronym: {acronym}")
        seen.add(acronym)
        specs.append(ModelSpec(model_name=model_name, acronym=acronym))

    return specs


def model_acronym_from_name(model_name: str) -> str:
    tail = model_name.rstrip("/").split("/")[-1]
    tail = re.sub(r"[^A-Za-z0-9]+", "_", tail).strip("_").lower()
    return tail[:48] or "model"


def run_paths(output_root: Path, spec: ModelSpec) -> RunPaths:
    output_dir = output_root / spec.acronym
    return RunPaths(
        output_dir=output_dir,
        generated_dir=output_dir / "generated_variants",
        failed_dir=output_dir / "failed",
        main_csv=output_dir / f"benchmark_{spec.acronym}.csv",
        metrics_overall_csv=output_dir / "metrics_overall.csv",
        metrics_by_group_csv=output_dir / "metrics_by_group.csv",
        threshold_sweep_csv=output_dir / "threshold_sweep.csv",
        confusion_matrix_csv=output_dir / "confusion_matrix.csv",
        errors_csv=output_dir / "errors.csv",
        summary_txt=output_dir / "summary.txt",
        columns_txt=output_dir / "columns.txt",
    )


def prepare_output_dirs(paths: RunPaths, clean_output: bool, keep_generated: bool) -> None:
    if clean_output and paths.output_dir.exists():
        shutil.rmtree(paths.output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.failed_dir.mkdir(parents=True, exist_ok=True)
    if keep_generated:
        paths.generated_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_name(value: Any) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    return safe[:180]


def stable_digest(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def is_supported_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


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


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 6)


def image_id_for_path(path: Path, data_root: Path) -> str:
    try:
        relative = str(path.relative_to(data_root))
    except ValueError:
        relative = str(path)
    return stable_digest(relative, 16)


def variant_id_for(parent_id: str, target: str | int, resize_mode: str) -> str:
    return "clean_" + stable_digest(f"{parent_id}|{target}|{resize_mode}", 18)


def source_save_format(path: Path) -> tuple[str, str, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "JPEG", ".jpg", {"quality": 95, "subsampling": 0}
    return "PNG", ".png", {"compress_level": 6}


def prediction_from_score(fake_score: float, threshold: float) -> tuple[str, int]:
    if fake_score >= threshold:
        return "fake", 1
    return "real", 0


def result_from_binary(actual_binary: int, prediction_binary: int) -> str:
    if actual_binary == 1 and prediction_binary == 1:
        return "TP"
    if actual_binary == 0 and prediction_binary == 0:
        return "TN"
    if actual_binary == 0 and prediction_binary == 1:
        return "FP"
    return "FN"


# ============================================================
# DATA DISCOVERY
# ============================================================

def resolve_dataset_root(data_root: Path) -> Path:
    data_root = data_root.expanduser().resolve()
    nested = data_root / "data"
    if nested.is_dir() and any(child.is_dir() for child in nested.iterdir()):
        return nested
    return data_root


def folder_label(name: str) -> str | None:
    normalized = clean_name(name).lower()
    if normalized in REAL_FOLDER_NAMES:
        return "real"
    if normalized in FAKE_FOLDER_NAMES:
        return "fake"
    return None


def has_images(directory: Path) -> bool:
    return any(is_supported_image_path(path) for path in directory.rglob("*"))


def direct_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if is_supported_image_path(path))


def infer_direct_file_source(path: Path, actual_label: str, fallback: str) -> str:
    if actual_label != "fake":
        return fallback

    stem = clean_name(path.stem).lower()
    if stem.startswith(("gmni", "gemni", "gemini")):
        return "gemini"
    return "chatgpt"


def discover_images(data_root: Path, random_seed: int, balance_labels: bool) -> Any:
    rows: list[dict[str, Any]] = []
    data_root = resolve_dataset_root(data_root)

    label_dirs = [child for child in sorted(data_root.iterdir()) if child.is_dir() and folder_label(child.name)]

    if label_dirs:
        label_dir_names = {child.name for child in label_dirs}
        for label_dir in label_dirs:
            actual_label = folder_label(label_dir.name) or "real"
            source_dirs = [child for child in sorted(label_dir.iterdir()) if child.is_dir() and has_images(child)]
            direct_files = direct_images(label_dir)
            for source_dir in source_dirs:
                add_source_rows(rows, data_root, source_dir, source_dir.name, actual_label)
            if direct_files:
                grouped_files: dict[str, list[Path]] = {}
                for path in direct_files:
                    source = infer_direct_file_source(path, actual_label, actual_label)
                    grouped_files.setdefault(source, []).append(path)
                for source, files in sorted(grouped_files.items()):
                    add_file_rows(rows, data_root, files, source, actual_label, label_dir)

        extra_real_dirs = [
            child
            for child in sorted(data_root.iterdir())
            if child.is_dir()
            and child.name not in label_dir_names
            and not child.name.startswith(".")
            and child.name not in IGNORED_DATA_DIRS
            and has_images(child)
        ]
        for source_dir in extra_real_dirs:
            add_source_rows(rows, data_root, source_dir, source_dir.name, "real")
    else:
        source_dirs = [
            child
            for child in sorted(data_root.iterdir())
            if child.is_dir()
            and not child.name.startswith(".")
            and child.name not in IGNORED_DATA_DIRS
            and has_images(child)
        ]
        for source_dir in source_dirs:
            guessed_label = folder_label(source_dir.name) or "real"
            add_source_rows(rows, data_root, source_dir, source_dir.name, guessed_label)

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        return raw_df

    print(f"Discovered originals: {len(raw_df):,}")
    print(f"Discovered counts   : {label_count_dict(raw_df)}")
    selected_df = balance_images_by_label(raw_df, random_seed) if balance_labels else raw_df
    if not balance_labels:
        print("Balanced sampling   : disabled")

    with ThreadPoolExecutor(max_workers=METADATA_WORKERS) as executor:
        enriched = list(
            tqdm(
                executor.map(add_original_metadata, selected_df.to_dict("records")),
                total=len(selected_df),
                desc="Original metadata",
                unit="img",
            )
        )
    return pd.DataFrame(enriched)


def add_source_rows(rows: list[dict[str, Any]], data_root: Path, source_dir: Path, source: str, actual_label: str) -> None:
    files = sorted(path for path in source_dir.rglob("*") if is_supported_image_path(path))
    add_file_rows(rows, data_root, files, source, actual_label, source_dir)


def add_file_rows(rows: list[dict[str, Any]], data_root: Path, files: list[Path], source: str, actual_label: str, subgroup_root: Path) -> None:
    print(f"{actual_label:<5} {source:<24}: {len(files):,} images")
    for path in files:
        rel = path.relative_to(subgroup_root)
        subgroup = "" if str(rel.parent) == "." else str(rel.parent)
        image_id = image_id_for_path(path, data_root)
        actual_binary = 1 if actual_label == "fake" else 0
        rows.append(
            {
                "parent_image_id": f"{actual_label}_{source}_{image_id}",
                "original_image_id": image_id,
                "source": source,
                "source_subgroup": subgroup,
                "actual_label": actual_label,
                "actual_binary": actual_binary,
                "original_path": str(path),
                "original_file_name": path.name,
                "original_file_format": path.suffix.lower().lstrip("."),
                "source_fps": parse_source_fps(path.name),
                "error": "",
            }
        )


def label_count_dict(images: Any) -> dict[str, int]:
    return {str(label): int(count) for label, count in images.groupby("actual_label").size().items()}


def balance_images_by_label(images: Any, random_seed: int) -> Any:
    if images.empty or "actual_label" not in images.columns:
        return images

    label_counts = images.groupby("actual_label").size()
    if len(label_counts) < 2:
        print("Balanced sampling   : skipped, only one label found")
        return images

    min_count = int(label_counts.min())
    balanced_parts = []
    for label in sorted(label_counts.index):
        group = images[images["actual_label"] == label]
        balanced_parts.append(
            group.sample(n=min_count, random_state=random_seed, replace=False)
            if len(group) > min_count
            else group.copy()
        )

    balanced = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_seed)
        .reset_index(drop=True)
    )
    print(f"Balanced sampling   : {label_count_dict(images)} -> {label_count_dict(balanced)}")
    return balanced


def open_source_image(path: Path) -> Any:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode not in {"RGB", "RGBA", "L"}:
        image = image.convert("RGB")
    return image


def add_original_metadata(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    try:
        with open_source_image(Path(row["original_path"])) as image:
            width, height = image.size
        result["original_width"] = width
        result["original_height"] = height
        result["original_megapixels"] = round(width * height / 1_000_000, 6)
    except Exception as error:
        result["error"] = f"original_metadata_error: {error}"
    return result


# ============================================================
# VARIANTS
# ============================================================

def resize_preserve_aspect(image: Any, target_dimension: str | int) -> Any:
    if target_dimension == "original":
        return image.copy()
    target = int(target_dimension)
    width, height = image.size
    scale = target / max(width, height)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def resize_square_center_crop(image: Any, target_dimension: str | int) -> Any:
    if target_dimension == "original":
        return image.copy()
    target = int(target_dimension)
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((target, target), Image.Resampling.LANCZOS)


def resize_for_mode(image: Any, target_dimension: str | int, resize_mode: str) -> Any:
    if target_dimension == "original":
        return image.copy()
    if resize_mode == "square":
        return resize_square_center_crop(image, target_dimension)
    return resize_preserve_aspect(image, target_dimension)


def image_stats(image: Any) -> dict[str, Any]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    stat_image = rgb.copy()
    stat_image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    array = np.asarray(stat_image)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
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
        "sharpness_score": round(blur, 6),
    }


def logical_variant_file_name(row: Any, width: int, height: int, extension: str) -> str:
    source = clean_name(row["source"])
    label = clean_name(row["actual_label"])
    target = clean_name(row["target_dimension"])
    mode = clean_name(row.get("resize_mode", ""))
    idx = int(stable_digest(str(row["variant_id"]), 10), 16) % 1_000_000
    mode_part = f"{mode}_" if mode else ""
    return f"clean_{label}_{source}_{target}_{mode_part}{width}x{height}_{idx:06d}{extension}"


def generated_path(paths: RunPaths, row: Any, width: int, height: int, extension: str) -> Path:
    return (
        paths.generated_dir
        / clean_name(row["actual_label"])
        / clean_name(row["source"])
        / clean_name(row["target_dimension"])
        / clean_name(row.get("resize_mode", ""))
        / logical_variant_file_name(row, width, height, extension)
    )


def build_plan(images: Any, spec: ModelSpec, threshold: float, random_seed: int) -> Any:
    rows: list[dict[str, Any]] = []
    for _, image in images.iterrows():
        if str(image.get("error", "") or ""):
            row = empty_row()
            for key, value in image.to_dict().items():
                if key in row:
                    row[key] = value
            row["test_type"] = "clean"
            row["model_name"] = spec.model_name
            row["model_acronym"] = spec.acronym
            row["threshold"] = threshold
            row["random_seed"] = random_seed
            rows.append(row)
            continue

        for target_dimension in TARGET_DIMENSIONS:
            modes = [""] if target_dimension == "original" else RESIZE_MODES
            for resize_mode in modes:
                row = empty_row()
                row.update(image.to_dict())
                row.update(
                    {
                        "variant_id": variant_id_for(image["parent_image_id"], target_dimension, resize_mode),
                        "test_type": "clean",
                        "target_dimension": str(target_dimension),
                        "resize_mode": resize_mode,
                        "variant_path": image["original_path"] if target_dimension == "original" else "",
                        "variant_file_name": "",
                        "variant_file_format": image["original_file_format"] if target_dimension == "original" else "",
                        "model_name": spec.model_name,
                        "model_acronym": spec.acronym,
                        "threshold": threshold,
                        "random_seed": random_seed,
                        "error": "",
                    }
                )
                rows.append(row)
    return normalize_runtime_dtypes(pd.DataFrame(rows, columns=CENTRAL_COLUMNS))


def empty_row() -> dict[str, Any]:
    return {column: "" for column in CENTRAL_COLUMNS}


def normalize_runtime_dtypes(df: Any) -> Any:
    df = df.copy()
    for column in CENTRAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df = df[CENTRAL_COLUMNS].astype("object")
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    for column in TEXT_COLUMNS:
        df[column] = df[column].fillna("").astype("object")
    return df[CENTRAL_COLUMNS]


def prepare_variant(row: Any, paths: RunPaths, temp_dir: Path, keep_generated: bool) -> tuple[Any, dict[str, Any], Path | None]:
    source_path = Path(row["original_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source image: {source_path}")
    if not is_supported_image_path(source_path):
        raise ValueError(f"Unsupported image extension: {source_path}")

    with open_source_image(source_path) as source_image:
        variant_image = resize_for_mode(source_image, row["target_dimension"], str(row.get("resize_mode", "")))
        stats = image_stats(variant_image)

        if str(row["target_dimension"]) == "original":
            stats["variant_path"] = str(source_path)
            stats["variant_file_name"] = source_path.name
            stats["variant_file_format"] = source_path.suffix.lower().lstrip(".")
            stats["file_size_mb"] = file_size_mb(source_path)
            return str(source_path), stats, None

        save_format, extension, save_kwargs = source_save_format(source_path)
        out_path = (
            generated_path(paths, row, stats["variant_width"], stats["variant_height"], extension)
            if keep_generated
            else temp_dir / logical_variant_file_name(row, stats["variant_width"], stats["variant_height"], extension)
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_image = variant_image
        if save_format == "JPEG" and save_image.mode == "RGBA":
            save_image = save_image.convert("RGB")
        save_image.save(out_path, format=save_format, **save_kwargs)

        stats["variant_path"] = str(out_path) if keep_generated else ""
        stats["variant_file_name"] = out_path.name
        stats["variant_file_format"] = extension.lstrip(".")
        stats["file_size_mb"] = file_size_mb(out_path)
        return str(out_path), stats, None if keep_generated else out_path


# ============================================================
# MODEL OUTPUT HANDLING
# ============================================================

def load_classifier(spec: ModelSpec):
    print("\nLoading model")
    print(f"Model   : {spec.model_name}")
    print(f"Acronym : {spec.acronym}")
    print(f"Device  : {DEVICE}")
    classifier = pipeline("image-classification", model=spec.model_name, device=DEVICE)
    print(f"id2label: {getattr(classifier.model.config, 'id2label', {})}")
    return classifier


def normalize_pipeline_outputs(outputs: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(outputs, list):
        outputs = [outputs]
    normalized: list[list[dict[str, Any]]] = []
    for output in outputs:
        if isinstance(output, dict):
            normalized.append([output])
        else:
            normalized.append(list(output))
    return normalized


def label_matches(label: str, patterns: list[str]) -> bool:
    normalized = clean_name(label).lower()
    return any(pattern in normalized for pattern in patterns)


def extract_fake_probability(output: list[dict[str, Any]]) -> float:
    if not output:
        raise ValueError("Empty model output")

    by_label = {str(item.get("label", "")).lower(): float(item.get("score", 0.0)) for item in output}

    fake_scores = [
        float(item.get("score", 0.0))
        for item in output
        if label_matches(str(item.get("label", "")), FAKE_LABEL_PATTERNS)
    ]
    if fake_scores:
        return max(fake_scores)

    real_scores = [
        float(item.get("score", 0.0))
        for item in output
        if label_matches(str(item.get("label", "")), REAL_LABEL_PATTERNS)
    ]
    if real_scores and len(output) <= 2:
        return 1.0 - max(real_scores)

    if len(output) == 1:
        # CommunityForensics-DeepfakeDet-ViT exposes a single LABEL_0 score;
        # in the existing benchmark this score is treated as fake probability.
        return float(output[0].get("score", 0.0))

    if "label_0" in by_label:
        return by_label["label_0"]

    raise ValueError(f"Cannot identify fake label from model output labels: {[item.get('label') for item in output]}")


def update_prediction_row(df: Any, index: int, fake_score: float, inference_ms: float, threshold: float) -> None:
    fake_score = float(fake_score)
    real_score = 1.0 - fake_score
    prediction, prediction_binary = prediction_from_score(fake_score, threshold)
    actual_binary = int(df.at[index, "actual_binary"])
    result = result_from_binary(actual_binary, prediction_binary)

    df.at[index, "fake_probability"] = fake_score
    df.at[index, "real_probability"] = real_score
    df.at[index, "prediction"] = prediction
    df.at[index, "prediction_binary"] = prediction_binary
    df.at[index, "result"] = result
    df.at[index, "is_correct"] = int(actual_binary == prediction_binary)
    df.at[index, "inference_ms"] = inference_ms
    df.at[index, "error"] = ""


def is_completed(row: Any) -> bool:
    if str(row.get("error", "") or ""):
        return True
    try:
        return not pd.isna(float(row.get("fake_probability", np.nan)))
    except Exception:
        return False


def save_checkpoint(df: Any, paths: RunPaths) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    df = normalize_runtime_dtypes(df)
    df[CENTRAL_COLUMNS].to_csv(paths.main_csv, index=False)


def export_failed(row: Any, tested_path: Path | str | None, paths: RunPaths) -> None:
    result = str(row.get("result", ""))
    if result not in {"FP", "FN"} or tested_path is None:
        return
    source_path = Path(tested_path)
    if not source_path.exists():
        return
    score = float(row.get("fake_probability", 0.0))
    out_dir = paths.failed_dir / result / clean_name(row.get("actual_label", "")) / clean_name(row.get("source", ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"score_{score:.4f}_{clean_name(row.get('variant_id', 'variant'))}_{clean_name(source_path.name)}"
    shutil.copy2(source_path, out_dir / name)


def run_inference(df: Any, classifier: Any, paths: RunPaths, args: argparse.Namespace) -> Any:
    df = normalize_runtime_dtypes(df)
    pending_indexes = [index for index, row in df.iterrows() if not is_completed(row)]
    if not pending_indexes:
        print("No pending variants.")
        return df

    total_batches = math.ceil(len(pending_indexes) / args.batch_size)
    completed_batches = 0
    for start in tqdm(range(0, len(pending_indexes), args.batch_size), total=total_batches, desc="Inference", unit="batch"):
        with tempfile.TemporaryDirectory(prefix="complete_benchmark_") as tmp:
            temp_dir = Path(tmp)
            batch_indexes = pending_indexes[start : start + args.batch_size]
            model_inputs: list[Any] = []
            prepared: list[tuple[int, Path | None]] = []

            for index in batch_indexes:
                try:
                    model_input, stats, temp_path = prepare_variant(df.loc[index], paths, temp_dir, args.keep_generated_variants)
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
                    outputs = classifier(model_inputs, batch_size=args.batch_size, top_k=None, function_to_apply="sigmoid")
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    per_image_ms = elapsed_ms / len(model_inputs)
                    normalized_outputs = normalize_pipeline_outputs(outputs)

                    for (index, temp_path), output in zip(prepared, normalized_outputs):
                        fake_score = extract_fake_probability(output)
                        update_prediction_row(df, index, fake_score, per_image_ms, args.threshold)
                        tested_path = temp_path or Path(df.at[index, "original_path"])
                        export_failed(df.loc[index], tested_path, paths)
                        if temp_path and temp_path.exists() and not args.keep_generated_variants:
                            temp_path.unlink()
                except Exception as error:
                    for index, temp_path in prepared:
                        df.at[index, "error"] = f"inference_error: {error}"
                        if temp_path and temp_path.exists() and not args.keep_generated_variants:
                            temp_path.unlink()

        completed_batches += 1
        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        if completed_batches % args.checkpoint_every == 0:
            save_checkpoint(df, paths)

    save_checkpoint(df, paths)
    return df


# ============================================================
# METRICS
# ============================================================

def roc_auc_score_manual(y_true: Any, scores: Any) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = pd.Series(np.concatenate([pos, neg]))
    ranks = combined.rank(method="average").to_numpy()
    pos_ranks = ranks[: len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def metrics_for_group(group: Any, threshold: float) -> dict[str, Any]:
    valid = group[(group["error"].fillna("") == "") & group["fake_probability"].notna()].copy()
    count = int(len(valid))
    if count == 0:
        return empty_metrics()

    actual = valid["actual_binary"].astype(int)
    predicted = (valid["fake_probability"].astype(float) >= threshold).astype(int)
    scores = valid["fake_probability"].astype(float)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return {
        "count": count,
        "real_count": int((actual == 0).sum()),
        "fake_count": int((actual == 1).sum()),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": safe_div(tp + tn, count),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "TNR": specificity,
        "FPR": fpr,
        "FNR": fnr,
        "NPV": safe_div(tn, tn + fn),
        "FDR": safe_div(fp, tp + fp),
        "F1": f1,
        "balanced_accuracy": np.nanmean([recall, specificity]),
        "MCC": ((tp * tn) - (fp * fn)) / mcc_den if mcc_den else np.nan,
        "ROC_AUC": roc_auc_score_manual(actual, scores),
        "score_mean": scores.mean(),
        "score_median": scores.median(),
        "score_std": scores.std(ddof=0),
        "score_min": scores.min(),
        "score_max": scores.max(),
        "score_p95": scores.quantile(0.95),
        "score_p99": scores.quantile(0.99),
    }


def empty_metrics() -> dict[str, Any]:
    keys = [
        "count", "real_count", "fake_count", "TP", "TN", "FP", "FN", "accuracy", "precision", "recall",
        "sensitivity", "specificity", "TNR", "FPR", "FNR", "NPV", "FDR", "F1", "balanced_accuracy", "MCC",
        "ROC_AUC", "score_mean", "score_median", "score_std", "score_min", "score_max", "score_p95", "score_p99",
    ]
    return {key: 0 if key in {"count", "real_count", "fake_count", "TP", "TN", "FP", "FN"} else np.nan for key in keys}


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def create_metrics_reports(df: Any, paths: RunPaths, threshold: float) -> tuple[Any, Any, Any, Any]:
    overall = pd.DataFrame([{**{"group": "overall", "group_value": "all"}, **metrics_for_group(df, threshold)}])

    group_frames: list[Any] = []
    for group_name, group_cols in [
        ("source", ["source"]),
        ("actual_label", ["actual_label"]),
        ("target_dimension", ["target_dimension"]),
        ("resize_mode", ["resize_mode"]),
        ("source_label", ["source", "actual_label"]),
        ("source_resolution", ["source", "target_dimension", "resize_mode"]),
        ("resolution_bucket", ["resolution_bucket"]),
        ("brightness_bucket", ["brightness_bucket"]),
        ("blur_bucket", ["blur_bucket"]),
        ("file_format", ["variant_file_format"]),
    ]:
        rows = []
        for keys, group in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"group": group_name}
            for col, value in zip(group_cols, keys):
                row[col] = value
            row.update(metrics_for_group(group, threshold))
            rows.append(row)
        if rows:
            group_frames.append(pd.DataFrame(rows))

    by_group = pd.concat(group_frames, ignore_index=True, sort=False) if group_frames else pd.DataFrame()
    threshold_sweep = create_threshold_sweep(df)
    confusion = create_confusion_matrix(df)

    overall.to_csv(paths.metrics_overall_csv, index=False)
    by_group.to_csv(paths.metrics_by_group_csv, index=False)
    threshold_sweep.to_csv(paths.threshold_sweep_csv, index=False)
    confusion.to_csv(paths.confusion_matrix_csv, index=False)
    create_errors(df).to_csv(paths.errors_csv, index=False)
    return overall, by_group, threshold_sweep, confusion


def create_threshold_sweep(df: Any) -> Any:
    rows: list[dict[str, Any]] = []
    valid = df[(df["error"].fillna("") == "") & df["fake_probability"].notna()].copy()
    thresholds = [round(x / 100, 2) for x in range(5, 100, 5)]

    for threshold in thresholds:
        rows.append({**{"threshold": threshold, "group": "overall", "group_value": "all"}, **metrics_for_group(valid, threshold)})
        for source, group in valid.groupby("source", dropna=False):
            rows.append({**{"threshold": threshold, "group": "source", "group_value": source}, **metrics_for_group(group, threshold)})
        for target, group in valid.groupby("target_dimension", dropna=False):
            rows.append({**{"threshold": threshold, "group": "target_dimension", "group_value": target}, **metrics_for_group(group, threshold)})

    return pd.DataFrame(rows)


def create_confusion_matrix(df: Any) -> Any:
    valid = df[(df["error"].fillna("") == "") & df["result"].fillna("").ne("")]
    rows = []
    for actual in ["real", "fake"]:
        for predicted in ["real", "fake"]:
            rows.append(
                {
                    "actual_label": actual,
                    "prediction": predicted,
                    "count": int(((valid["actual_label"] == actual) & (valid["prediction"] == predicted)).sum()),
                }
            )
    return pd.DataFrame(rows)


def create_errors(df: Any) -> Any:
    return df[df["error"].fillna("") != ""].copy()


def write_column_guide(paths: RunPaths) -> None:
    lines = [
        "COMPLETE REAL/FAKE BENCHMARK CSV COLUMN GUIDE",
        "=" * 72,
        "",
        f"Main CSV: {paths.main_csv}",
        "",
        "Result meaning:",
        "TP = fake image predicted fake",
        "TN = real image predicted real",
        "FP = real image predicted fake",
        "FN = fake image predicted real",
        "",
        "Column definitions:",
        "",
    ]
    for column in CENTRAL_COLUMNS:
        lines.append(f"{column}: {COLUMN_DESCRIPTIONS.get(column, 'No description available.')}")
    paths.columns_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(paths: RunPaths, spec: ModelSpec, df: Any, overall: Any, by_group: Any) -> None:
    metrics = overall.iloc[0].to_dict() if not overall.empty else {}
    lines = [
        "COMPLETE REAL/FAKE DEEPFAKE BENCHMARK",
        "=" * 80,
        f"Model          : {spec.model_name}",
        f"Model acronym  : {spec.acronym}",
        f"Output dir     : {paths.output_dir}",
        f"Main CSV       : {paths.main_csv}",
        f"Rows           : {len(df):,}",
        f"Errors         : {int((df['error'].fillna('') != '').sum()):,}",
        "",
        "Overall metrics:",
    ]
    for key in ["count", "real_count", "fake_count", "TP", "TN", "FP", "FN", "accuracy", "precision", "recall", "specificity", "FPR", "FNR", "F1", "balanced_accuracy", "MCC", "ROC_AUC"]:
        value = metrics.get(key, "")
        if isinstance(value, float):
            lines.append(f"{key:<18}: {value:.6f}")
        else:
            lines.append(f"{key:<18}: {value}")

    if not by_group.empty:
        display_cols = [col for col in ["group", "source", "actual_label", "target_dimension", "resize_mode", "count", "TP", "TN", "FP", "FN", "accuracy", "F1", "FPR", "FNR"] if col in by_group.columns]
        lines.extend(["", "Top grouped rows:", by_group[display_cols].head(30).to_string(index=False)])

    paths.summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_report(paths: RunPaths, spec: ModelSpec, overall: Any) -> None:
    print("\n" + "=" * 80)
    print("COMPLETE REAL/FAKE BENCHMARK FINISHED")
    print("=" * 80)
    print(f"Model output dir : {paths.output_dir}")
    print(f"Main CSV         : {paths.main_csv}")
    print(f"Overall metrics  : {paths.metrics_overall_csv}")
    print(f"Grouped metrics  : {paths.metrics_by_group_csv}")
    print(f"Threshold sweep  : {paths.threshold_sweep_csv}")
    if not overall.empty:
        row = overall.iloc[0]
        print(f"Accuracy         : {row.get('accuracy', np.nan):.2%}")
        print(f"F1               : {row.get('F1', np.nan):.4f}")
        print(f"FPR              : {row.get('FPR', np.nan):.2%}")
        print(f"FNR              : {row.get('FNR', np.nan):.2%}")


# ============================================================
# MAIN
# ============================================================

def run_model(images: Any, spec: ModelSpec, args: argparse.Namespace) -> None:
    paths = run_paths(args.output_root.expanduser().resolve(), spec)
    prepare_output_dirs(paths, clean_output=not args.no_clean_model_output, keep_generated=args.keep_generated_variants)
    write_column_guide(paths)

    print("\n" + "=" * 80)
    print(f"MODEL RUN: {spec.acronym}")
    print("=" * 80)
    print(f"Model      : {spec.model_name}")
    print(f"Output dir : {paths.output_dir}")
    print(f"Overwrite  : {'no, keeping existing model output' if args.no_clean_model_output else 'yes, cleaning this model output before run'}")

    df = build_plan(images, spec, args.threshold, args.random_seed)
    save_checkpoint(df, paths)
    print(f"Planned variants: {len(df):,}")

    classifier = load_classifier(spec)
    df = run_inference(df, classifier, paths, args)
    save_checkpoint(df, paths)

    overall, by_group, _threshold_sweep, _confusion = create_metrics_reports(df, paths, args.threshold)
    write_summary(paths, spec, df, overall, by_group)
    print_report(paths, spec, overall)

    del classifier
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    ensure_runtime_dependencies(need_model_runtime=not args.plan_only)
    specs = parse_model_specs(args.models)

    data_root = resolve_dataset_root(args.data_root)
    print("=" * 80)
    print("COMPLETE REAL/FAKE DEEPFAKE BENCHMARK PIPELINE")
    print("=" * 80)
    print(f"Data root          : {data_root}")
    print(f"Output root        : {args.output_root.expanduser().resolve()}")
    print(f"Models             : {', '.join(f'{s.acronym}={s.model_name}' for s in specs)}")
    print(f"Threshold          : {args.threshold}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Device             : {DEVICE}")

    print("\n[1/3] Discovering images...")
    images = discover_images(data_root, args.random_seed, balance_labels=not args.no_balance_labels)
    if images.empty:
        raise RuntimeError(f"No png/jpg/jpeg images found under {data_root}")
    label_counts = label_count_dict(images)
    print(f"Total originals    : {len(images):,}")
    print(f"Label counts       : {label_counts}")

    if args.plan_only:
        variants_per_original = 1 + (len(TARGET_DIMENSIONS) - 1) * len(RESIZE_MODES)
        print("\nPlan only:")
        print(f"Variants/original  : {variants_per_original}")
        print(f"Variants/model     : {len(images) * variants_per_original:,}")
        print(f"Model output dirs  : {', '.join(str((args.output_root.expanduser().resolve() / s.acronym)) for s in specs)}")
        return

    for i, spec in enumerate(specs, start=1):
        print(f"\n[{i + 1}/3] Running model {spec.acronym}...")
        run_model(images, spec, args)

    print("\nAll model runs complete.")


if __name__ == "__main__":
    main()
