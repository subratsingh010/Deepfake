import argparse
import random
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download images from Unsplash photos.tsv"
    )

    parser.add_argument(
        "--tsv",
        required=True,
        help="Path to photos.tsv"
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=300,
        help="Number of source images to download"
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["jpg", "png"],
        default=["jpg"],
        help="Output formats: jpg, png, or both"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality"
    )

    return parser.parse_args()


def download_image(url, timeout=30):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=timeout
    )

    response.raise_for_status()

    return Image.open(BytesIO(response.content))


def main():
    args = parse_args()

    tsv_path = Path(args.tsv)
    output_dir = Path(args.out)

    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    # Create format directories
    for fmt in args.formats:
        (output_dir / fmt).mkdir(
            parents=True,
            exist_ok=True
        )

    print(f"Reading: {tsv_path}")

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        low_memory=False
    )

    if "photo_image_url" not in df.columns:
        raise ValueError(
            "photos.tsv does not contain 'photo_image_url'"
        )

    # Remove invalid URLs
    df = df[
        df["photo_image_url"].notna()
    ].copy()

    print(f"Available images: {len(df)}")

    random.seed(args.seed)

    # Shuffle dataset reproducibly.
    indexes = list(df.index)
    random.shuffle(indexes)

    success_count = 0
    failed_count = 0

    progress = tqdm(
        total=args.count,
        desc="Downloading"
    )

    for row_index in indexes:

        if success_count >= args.count:
            break

        row = df.loc[row_index]

        photo_id = str(row["photo_id"])
        url = str(row["photo_image_url"])

        try:
            image = download_image(url)

            # Load fully before response buffer disappears
            image.load()

            width, height = image.size

            success_count += 1

            filename_base = (
                f"unsplash_{width}x{height}_{success_count:04d}"
            )

            # JPG
            if "jpg" in args.formats:
                jpg_image = image.convert("RGB")

                jpg_path = (
                    output_dir
                    / "jpg"
                    / f"{filename_base}.jpg"
                )

                jpg_image.save(
                    jpg_path,
                    "JPEG",
                    quality=args.jpg_quality
                )

            # PNG
            if "png" in args.formats:
                if image.mode not in ("RGB", "RGBA"):
                    png_image = image.convert("RGB")
                else:
                    png_image = image

                png_path = (
                    output_dir
                    / "png"
                    / f"{filename_base}.png"
                )

                png_image.save(
                    png_path,
                    "PNG"
                )

            progress.update(1)

        except Exception as e:
            failed_count += 1
            tqdm.write(
                f"Failed [{photo_id}]: {e}"
            )

    progress.close()

    print("\nCompleted")
    print(f"Downloaded source images : {success_count}")
    print(f"Failed attempts          : {failed_count}")
    print(f"Formats                  : {', '.join(args.formats)}")
    print(f"Output directory         : {output_dir}")


if __name__ == "__main__":
    main()