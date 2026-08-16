from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

INPUT_DIR = Path("/Users/subrat/Downloads/Deepfake")
OUTPUT_DIR = Path("/Users/subrat/Desktop/Deepfake/camera")
print(f"Input directory: {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
SAVE_PNG = True
SAVE_JPG = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

heic_files = list(INPUT_DIR.glob("*.heic")) + list(INPUT_DIR.glob("*.HEIC"))
print(f"Found {len(heic_files)} HEIC files in {INPUT_DIR}")

for index, file_path in enumerate(heic_files, start=1):
    if file_path.suffix.lower() not in {".heic", ".heif"}:
        print(f"Skipping non-HEIC/HEIF file: {file_path.name}")
        continue
    try:
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            width, height = img.size

            base_name = f"camera_{width}x{height}_{index:03d}"

            if SAVE_PNG:
                img.save(
                    OUTPUT_DIR / f"{base_name}.png",
                    "PNG"
                )

            if SAVE_JPG:
                img.save(
                    OUTPUT_DIR / f"{base_name}.jpg",
                    "JPEG",
                    quality=95
                )

        print(f"[{index}/{len(heic_files)}] Converted: {file_path.name}")

    except Exception as e:
        print(f"Failed: {file_path.name} -> {e}")

print("Done.")