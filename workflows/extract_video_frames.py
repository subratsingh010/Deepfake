import json
import subprocess
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

INPUT_DIR = Path("/Users/subrat/Downloads/Deepfake")
OUTPUT_DIR = Path("/Users/subrat/Desktop/Deepfake/video_frames")

# Use ONE mode

# Example:
# 1   = 1 frame per second
# 0.5 = 1 frame every 2 seconds
EXTRACT_FPS = 1

# Example:
# 3  = every 3rd frame
# 5  = every 5th frame
# 10 = every 10th frame
EVERY_NTH_FRAME = None

SAVE_PNG = True
SAVE_JPG = True

# FFmpeg JPG quality:
# 2 = high quality
# 31 = low quality
JPG_QUALITY = 2

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
    ".webm",
}


# =========================================================
# VIDEO INFO
# =========================================================

def get_video_info(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate:format=duration",
        "-of", "json",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    info = json.loads(result.stdout)

    stream = info["streams"][0]

    width = int(stream["width"])
    height = int(stream["height"])

    fps_raw = stream["r_frame_rate"]
    numerator, denominator = fps_raw.split("/")

    fps = float(numerator) / float(denominator)

    duration = float(info["format"]["duration"])

    return width, height, fps, duration


# =========================================================
# FILTER
# =========================================================

def get_filter():

    if EVERY_NTH_FRAME is not None:

        if EVERY_NTH_FRAME <= 0:
            raise ValueError(
                "EVERY_NTH_FRAME must be greater than 0"
            )

        return f"select=not(mod(n\\,{EVERY_NTH_FRAME}))"

    if EXTRACT_FPS is not None:

        if EXTRACT_FPS <= 0:
            raise ValueError(
                "EXTRACT_FPS must be greater than 0"
            )

        return f"fps={EXTRACT_FPS}"

    raise ValueError(
        "Set either EXTRACT_FPS or EVERY_NTH_FRAME"
    )


# =========================================================
# EXTRACT
# =========================================================

def extract_frames(video_path):

    width, height, source_fps, duration = get_video_info(
        video_path
    )

    source_fps_int = round(source_fps)

    minutes = int(duration // 60)
    seconds = duration % 60

    print("\n==========================================")
    print(f"Video      : {video_path.name}")
    print(f"Resolution : {width}x{height}")
    print(f"Source FPS : {source_fps:.2f}")
    print(f"Duration   : {minutes:02d}:{seconds:05.2f}")

    if EVERY_NTH_FRAME is not None:
        print(
            f"Extract    : Every {EVERY_NTH_FRAME}th frame"
        )
    else:
        print(
            f"Extract    : {EXTRACT_FPS} frame(s)/second"
        )

    print("==========================================")

    video_name = video_path.stem.replace(" ", "_")

    video_output_dir = OUTPUT_DIR / video_name

    png_dir = video_output_dir / "png"
    jpg_dir = video_output_dir / "jpg"

    if SAVE_PNG:
        png_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if SAVE_JPG:
        jpg_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    video_filter = get_filter()

    # =====================================
    # PNG
    # =====================================

    if SAVE_PNG:

        png_pattern = (
            png_dir
            / f"video_{width}x{height}_{source_fps_int}fps_%04d.png"
        )

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-i", str(video_path),
            "-vf", video_filter,
            "-fps_mode", "vfr",
            str(png_pattern),
        ]

        print("Extracting PNG frames...")

        subprocess.run(
            cmd,
            check=True,
        )

    # =====================================
    # JPG
    # =====================================

    if SAVE_JPG:

        jpg_pattern = (
            jpg_dir
            / f"video_{width}x{height}_{source_fps_int}fps_%04d.jpg"
        )

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-i", str(video_path),
            "-vf", video_filter,
            "-fps_mode", "vfr",
            "-q:v", str(JPG_QUALITY),
            str(jpg_pattern),
        ]

        print("Extracting JPG frames...")

        subprocess.run(
            cmd,
            check=True,
        )

    print(f"Completed  : {video_path.name}")


# =========================================================
# MAIN
# =========================================================

def main():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    videos = sorted(
        [
            path
            for path in INPUT_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
    )

    if not videos:
        print(
            f"No video files found in: {INPUT_DIR}"
        )
        return

    print(f"Found {len(videos)} video(s).")

    # Print summary before extraction
    print("\nVIDEO SUMMARY")
    print("=" * 80)

    for video in videos:

        try:
            width, height, fps, duration = get_video_info(
                video
            )

            minutes = int(duration // 60)
            seconds = duration % 60

            print(
                f"{video.name} | "
                f"{width}x{height} | "
                f"{fps:.2f} FPS | "
                f"{minutes:02d}:{seconds:05.2f}"
            )

        except Exception as e:
            print(
                f"{video.name} | Unable to read info: {e}"
            )

    print("=" * 80)

    # Extract frames
    for video in videos:

        try:
            extract_frames(video)

        except Exception as e:
            print(
                f"Failed: {video.name} -> {e}"
            )

    print("\nAll videos processed.")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()