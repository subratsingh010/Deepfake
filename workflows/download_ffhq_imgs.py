import json
import random
import subprocess
import sys
import importlib.util
from copy import deepcopy
from pathlib import Path

from PIL import Image


# =========================
# CONFIG
# =========================

COUNT = 100
SEED = 42

BASE_DIR = Path("/Users/subrat/Desktop/Deepfake")

METADATA_DIR = BASE_DIR / "workflows" / "ffhq_metadata"
JSON_PATH = METADATA_DIR / "ffhq-dataset-v2.json"

OFFICIAL_DOWNLOADER = f"{BASE_DIR}/workflows/download_ffhq.py"

PNG_DIR = BASE_DIR / "ffhq" / "png"
JPG_DIR = BASE_DIR / "ffhq" / "jpg"

FFHQ_METADATA_ID = "16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA"


# =========================
# LOAD NVIDIA DOWNLOADER
# =========================

def load_ffhq_downloader():
    if not Path(OFFICIAL_DOWNLOADER).exists():
        raise FileNotFoundError(
            f"Official FFHQ downloader not found: {OFFICIAL_DOWNLOADER}"
        )

    spec = importlib.util.spec_from_file_location(
        "ffhq_official",
        OFFICIAL_DOWNLOADER
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# =========================
# DOWNLOAD METADATA
# =========================

def ensure_metadata():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    if JSON_PATH.exists():
        print(f"Metadata already exists:\n{JSON_PATH}")
        return

    print("FFHQ metadata not found.")
    print("Downloading metadata...")

    cmd = [
        sys.executable,
        "-m",
        "gdown",
        FFHQ_METADATA_ID,
        "-O",
        str(JSON_PATH),
    ]

    subprocess.run(cmd, check=True)

    print(f"Metadata saved to:\n{JSON_PATH}")


# =========================
# MAIN
# =========================

def main():

    # 1. Download metadata if missing
    ensure_metadata()

    # 2. Load official NVIDIA downloader
    download_ffhq = load_ffhq_downloader()

    # 3. Create output folders
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    JPG_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Read metadata
    print("Loading FFHQ metadata...")

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    print(f"Available FFHQ images: {len(data)}")

    # 5. Random selection
    items = list(data.items())

    random.seed(SEED)
    selected = random.sample(items, COUNT)

    specs = []

    for index, (ffhq_id, item) in enumerate(selected, start=1):

        spec = deepcopy(item["image"])

        png_path = (
            PNG_DIR
            / f"ffhq_1024x1024_{index:04d}.png"
        )

        spec["file_path"] = str(png_path)

        specs.append(spec)

    # 6. Download PNG images
    print(f"\nDownloading {COUNT} random FFHQ PNG images...")

    download_ffhq.download_files(
        specs,
        num_threads=8,
        num_attempts=10,
    )

    # 7. Create JPG copies
    print("\nCreating JPG copies...")

    for index in range(1, COUNT + 1):

        png_path = (
            PNG_DIR
            / f"ffhq_1024x1024_{index:04d}.png"
        )

        jpg_path = (
            JPG_DIR
            / f"ffhq_1024x1024_{index:04d}.jpg"
        )

        if not png_path.exists():
            print(f"Missing: {png_path.name}")
            continue

        with Image.open(png_path) as img:
            img.convert("RGB").save(
                jpg_path,
                "JPEG",
                quality=95,
            )

    print("\nDone.")
    print(f"Metadata : {JSON_PATH}")
    print(f"PNG      : {PNG_DIR}")
    print(f"JPG      : {JPG_DIR}")


if __name__ == "__main__":
    main()