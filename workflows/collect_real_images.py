#!/usr/bin/env python3
"""
Collect reproducible 500-image real-image subsets from:
  - DIV2K
  - Unsplash Lite
  - FFHQ
  - CelebAMask-HQ

The script preserves archives, extracted datasets, TSV files, metadata, and raw
downloaded source files. Selected images are copied byte-for-byte into separate
output directories after PIL validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

SEED = 20260815
COUNT_PER_DATASET = 500
MAX_RETRIES = 5
TIMEOUT_SECONDS = 60
USER_AGENT = "Mozilla/5.0 (compatible; real-image-collector/1.0)"

DIV2K_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
UNSPLASH_LITE_URL = (
    "https://unsplash-datasets.s3.amazonaws.com/lite/latest/"
    "unsplash-research-dataset-lite-latest.zip"
)
FFHQ_DOWNLOADER_URL = (
    "https://raw.githubusercontent.com/NVlabs/ffhq-dataset/master/download_ffhq.py"
)
CELEBAMASKHQ_GDRIVE_ID = "1badu11NqxGf6qM3PTTooQDJvQbejgbTv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FORMAT_TO_EXTENSION = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "TIFF": ".tif",
}

Image = None
UnidentifiedImageError = None
tqdm = None


@dataclass(frozen=True)
class ValidImage:
    path: Path
    width: int
    height: int
    extension: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect 500 validated real images each from DIV2K, Unsplash, FFHQ, and CelebAMask-HQ."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("real_image_sources"),
        help="Where archives, extracted datasets, metadata, and raw downloads are kept.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("real_image_selection"),
        help="Where selected renamed images are copied.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=COUNT_PER_DATASET,
        help="Number of images to collect per dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Fixed random seed for reproducibility.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["div2k", "unsplash", "ffhq", "celebamaskhq"],
        default=["div2k", "unsplash", "ffhq", "celebamaskhq"],
        help="Datasets to collect.",
    )
    parser.add_argument(
        "--overwrite-selected",
        action="store_true",
        help="Remove previously selected images in output dataset folders before copying new selections.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_command(args: list[str], cwd: Path | None = None) -> None:
    print(f"Running: {' '.join(args)}")
    subprocess.run(args, cwd=cwd, check=True)


def ensure_python_package(import_name: str, package_name: str | None = None) -> object:
    try:
        return import_module(import_name)
    except ImportError:
        package = package_name or import_name
        run_command([sys.executable, "-m", "pip", "install", "-q", package])
        return import_module(import_name)


def ensure_runtime_dependencies() -> None:
    global Image, UnidentifiedImageError, tqdm
    pil_image_module = ensure_python_package("PIL.Image", "Pillow")
    pil_module = ensure_python_package("PIL", "Pillow")
    tqdm_module = ensure_python_package("tqdm", "tqdm")
    Image = pil_image_module
    UnidentifiedImageError = getattr(pil_module, "UnidentifiedImageError")
    tqdm = tqdm_module.tqdm


def download_url(url: str, output_path: Path, desc: str | None = None) -> Path:
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    ensure_dir(output_path.parent)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                total = int(response.headers.get("Content-Length") or 0)
                with tmp_path.open("wb") as f, tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    desc=desc or output_path.name,
                    leave=False,
                ) as bar:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))
            tmp_path.replace(output_path)
            return output_path
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < MAX_RETRIES:
                sleep_seconds = min(2 ** attempt, 30)
                print(f"Download failed for {output_path.name}; retrying in {sleep_seconds}s ({attempt}/{MAX_RETRIES})")
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to download {url} to {output_path}: {last_error}")


def download_with_wget(url: str, output_path: Path) -> Path:
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    ensure_dir(output_path.parent)
    if shutil.which("wget"):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                run_command(["wget", url, "-O", str(output_path)])
                return output_path
            except subprocess.CalledProcessError:
                if output_path.exists():
                    output_path.unlink()
                if attempt < MAX_RETRIES:
                    sleep_seconds = min(2 ** attempt, 30)
                    print(f"wget failed for {output_path.name}; retrying in {sleep_seconds}s ({attempt}/{MAX_RETRIES})")
                    time.sleep(sleep_seconds)
        raise RuntimeError(f"wget failed after {MAX_RETRIES} attempts: {url}")

    print("wget was not found; falling back to Python download.")
    return download_url(url, output_path, desc=output_path.name)


def unzip_if_needed(zip_path: Path, extract_dir: Path, marker_name: str | None = None) -> None:
    ensure_dir(extract_dir)
    marker = extract_dir / (marker_name or ".extracted")
    if marker.exists():
        return

    if shutil.which("unzip"):
        run_command(["unzip", "-n", str(zip_path), "-d", str(extract_dir)])
    else:
        print("unzip was not found; falling back to Python zip extraction.")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    marker.touch()


def pil_validate(path: Path) -> ValidImage | None:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
            image_format = img.format
        extension = FORMAT_TO_EXTENSION.get(image_format or "", path.suffix.lower())
        if width <= 0 or height <= 0 or extension.lower() not in IMAGE_EXTENSIONS:
            return None
        return ValidImage(path=path, width=width, height=height, extension=extension.lower())
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def clear_selected_dir(path: Path, overwrite: bool) -> None:
    ensure_dir(path)
    if not overwrite:
        return
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            item.unlink()


def copy_selected(images: list[ValidImage], dataset_name: str, output_dir: Path, overwrite: bool) -> None:
    clear_selected_dir(output_dir, overwrite)
    ensure_dir(output_dir)
    for index, image in enumerate(tqdm(images, desc=f"Copying {dataset_name}", unit="img"), start=1):
        destination = output_dir / f"{dataset_name}_{image.width}x{image.height}_{index:04d}{image.extension}"
        if destination.exists() and not overwrite:
            continue
        shutil.copy2(image.path, destination)


def random_valid_subset(paths: Iterable[Path], count: int, rng: random.Random, desc: str) -> list[ValidImage]:
    candidates = list(paths)
    rng.shuffle(candidates)
    selected: list[ValidImage] = []
    for path in tqdm(candidates, desc=desc, unit="img"):
        valid = pil_validate(path)
        if valid is not None:
            selected.append(valid)
        if len(selected) >= count:
            return selected
    raise RuntimeError(f"Only found {len(selected)} valid images for {desc}; need {count}.")


def collect_div2k(data_root: Path, output_root: Path, count: int, rng: random.Random, overwrite: bool) -> None:
    archive = data_root / "DIV2K_train_HR.zip"
    extract_dir = data_root / "DIV2K"
    download_with_wget(DIV2K_URL, archive)
    unzip_if_needed(archive, extract_dir)

    hr_dirs = [p for p in extract_dir.rglob("DIV2K_train_HR") if p.is_dir()]
    search_dir = hr_dirs[0] if hr_dirs else extract_dir
    pngs = sorted(search_dir.rglob("*.png"))
    selected = random_valid_subset(pngs, count, rng, "Validating DIV2K")
    copy_selected(selected, "div2k", output_root / "div2k", overwrite)


def find_unsplash_photos_tsv(extract_dir: Path) -> Path:
    matches = sorted(extract_dir.rglob("photos.tsv"))
    if not matches:
        raise FileNotFoundError(f"Could not find photos.tsv under {extract_dir}")
    return matches[0]


def unsplash_photo_url(row: dict[str, str]) -> str | None:
    for key in ("photo_image_url", "image_url", "url"):
        value = row.get(key)
        if value:
            return value.strip()
    for key, value in row.items():
        if "image" in key.lower() and "url" in key.lower() and value:
            return value.strip()
    return None


def url_file_stem(url: str, fallback: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if name:
        return name.replace(".", "_")[:120]
    query_id = parse_qs(parsed.query).get("id", [fallback])[0]
    return query_id[:120] or fallback


def collect_unsplash(data_root: Path, output_root: Path, count: int, rng: random.Random, overwrite: bool) -> None:
    archive = data_root / "unsplash_lite.zip"
    extract_dir = data_root / "unsplash_lite"
    raw_dir = ensure_dir(data_root / "unsplash_downloaded_images")
    download_with_wget(UNSPLASH_LITE_URL, archive)
    unzip_if_needed(archive, extract_dir)

    photos_tsv = find_unsplash_photos_tsv(extract_dir)
    with photos_tsv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rng.shuffle(rows)

    selected: list[ValidImage] = []
    for row in tqdm(rows, desc="Downloading Unsplash", unit="img"):
        if len(selected) >= count:
            break
        url = unsplash_photo_url(row)
        if not url:
            continue
        photo_id = row.get("photo_id") or f"unsplash_{len(selected):04d}"
        raw_path = raw_dir / f"{photo_id}_{url_file_stem(url, photo_id)}.download"
        try:
            download_url(url, raw_path, desc=f"unsplash {len(selected) + 1}/{count}")
        except RuntimeError as exc:
            print(exc)
            continue
        valid = pil_validate(raw_path)
        if valid is not None:
            selected.append(valid)

    if len(selected) < count:
        raise RuntimeError(f"Only collected {len(selected)} valid Unsplash images; need {count}.")
    copy_selected(selected, "unsplash", output_root / "unsplash", overwrite)


def ensure_gdown() -> None:
    ensure_python_package("gdown", "gdown")


def gdown_file(file_id: str, output_path: Path) -> Path:
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    ensure_gdown()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            run_command(["gdown", file_id, "-O", str(output_path)])
            return output_path
        except subprocess.CalledProcessError:
            if output_path.exists():
                output_path.unlink()
            if attempt < MAX_RETRIES:
                sleep_seconds = min(2 ** attempt, 30)
                print(f"gdown failed for {output_path.name}; retrying in {sleep_seconds}s ({attempt}/{MAX_RETRIES})")
                time.sleep(sleep_seconds)
    raise RuntimeError(f"gdown failed after {MAX_RETRIES} attempts for file id {file_id}")


def collect_celebamaskhq(data_root: Path, output_root: Path, count: int, rng: random.Random, overwrite: bool) -> None:
    archive = data_root / "CelebAMask-HQ.zip"
    extract_dir = data_root / "CelebAMask-HQ"
    gdown_file(CELEBAMASKHQ_GDRIVE_ID, archive)
    unzip_if_needed(archive, extract_dir)

    preferred_dirs = [p for p in extract_dir.rglob("CelebA-HQ-img") if p.is_dir()]
    search_dir = preferred_dirs[0] if preferred_dirs else extract_dir
    image_paths = sorted(
        p for p in search_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    selected = random_valid_subset(image_paths, count, rng, "Validating CelebAMask-HQ")
    copy_selected(selected, "celebamaskhq", output_root / "celebamaskhq", overwrite)


def download_ffhq_downloader(data_root: Path) -> Path:
    downloader = data_root / "download_ffhq.py"
    download_with_wget(FFHQ_DOWNLOADER_URL, downloader)
    return downloader


def ensure_ffhq_metadata(data_root: Path, downloader: Path) -> Path:
    metadata = data_root / "ffhq-dataset-v2.json"
    if metadata.exists() and metadata.stat().st_size > 0:
        return metadata

    before = {p.resolve() for p in data_root.rglob("*.json")}
    run_command([sys.executable, str(downloader.resolve()), "--json"], cwd=data_root)
    after = [p for p in data_root.rglob("*.json") if p.resolve() not in before]
    candidates = [p for p in after if "ffhq" in p.name.lower()] or after
    if not candidates:
        candidates = sorted(data_root.rglob("ffhq-dataset-v2.json"))
    if not candidates:
        raise FileNotFoundError("The FFHQ downloader did not create an FFHQ metadata JSON file.")
    if candidates[0].resolve() != metadata.resolve():
        candidates[0].replace(metadata)
    return metadata


def ffhq_entries(metadata_path: Path) -> list[dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if isinstance(metadata, dict):
        values = list(metadata.values())
    elif isinstance(metadata, list):
        values = metadata
    else:
        raise ValueError(f"Unexpected FFHQ metadata shape in {metadata_path}")

    entries: list[dict] = []
    for item in values:
        image = item.get("image", {}) if isinstance(item, dict) else {}
        file_url = image.get("file_url") or item.get("file_url") if isinstance(item, dict) else None
        file_path = image.get("file_path") or item.get("file_path") if isinstance(item, dict) else None
        pixel_size = image.get("pixel_size") or item.get("pixel_size") if isinstance(item, dict) else None
        if not file_url:
            continue
        if pixel_size and tuple(pixel_size) != (1024, 1024):
            continue
        entries.append({"url": file_url, "file_path": file_path})
    if not entries:
        raise RuntimeError("No downloadable 1024x1024 FFHQ PNG entries were found in metadata.")
    return entries


def ffhq_output_name(entry: dict, ordinal: int) -> str:
    file_path = entry.get("file_path")
    if file_path:
        name = Path(file_path).name
        if name:
            return name
    return f"ffhq_source_{ordinal:05d}.png"


def collect_ffhq(data_root: Path, output_root: Path, count: int, rng: random.Random, overwrite: bool) -> None:
    downloader = download_ffhq_downloader(data_root)
    metadata_path = ensure_ffhq_metadata(data_root, downloader)
    entries = ffhq_entries(metadata_path)
    rng.shuffle(entries)

    raw_dir = ensure_dir(data_root / "ffhq_downloaded_images")
    selected: list[ValidImage] = []
    for ordinal, entry in enumerate(tqdm(entries, desc="Downloading FFHQ", unit="img"), start=1):
        if len(selected) >= count:
            break
        raw_path = raw_dir / ffhq_output_name(entry, ordinal)
        try:
            download_url(entry["url"], raw_path, desc=f"ffhq {len(selected) + 1}/{count}")
        except RuntimeError as exc:
            print(exc)
            continue
        valid = pil_validate(raw_path)
        if valid and valid.width == 1024 and valid.height == 1024 and valid.extension == ".png":
            selected.append(valid)

    if len(selected) < count:
        raise RuntimeError(f"Only collected {len(selected)} valid FFHQ images; need {count}.")
    copy_selected(selected, "ffhq", output_root / "ffhq", overwrite)


def main() -> None:
    args = parse_args()
    ensure_runtime_dependencies()
    data_root = ensure_dir(args.data_root)
    output_root = ensure_dir(args.output_root)
    rng = random.Random(args.seed)

    collectors = {
        "div2k": collect_div2k,
        "unsplash": collect_unsplash,
        "ffhq": collect_ffhq,
        "celebamaskhq": collect_celebamaskhq,
    }

    print(f"Using fixed random seed: {args.seed}")
    print(f"Keeping source data in: {data_root.resolve()}")
    print(f"Writing selected images to: {output_root.resolve()}")

    for dataset in args.only:
        print(f"\n=== {dataset} ===")
        collectors[dataset](data_root, output_root, args.count, rng, args.overwrite_selected)

    print("\nDone.")


if __name__ == "__main__":
    main()
