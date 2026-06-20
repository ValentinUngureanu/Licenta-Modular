import shutil
from pathlib import Path

import cv2

from crop import crop_ultrasound
from preprocessing import preprocess_crop


INPUT_DIR = Path("ORIGINAL_IMAGES")
OUTPUT_DIR = Path("RESULTS/PREPROCESSING_CLEAN_TEST1")


def image_index(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 10**9


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    (OUTPUT_DIR / "00_CROP").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "01_GRAY").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "02_PALETTE_7").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "03_BINARY_TOP1").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "04_BINARY_TOP2").mkdir(parents=True, exist_ok=True)

    image_paths = sorted(INPUT_DIR.glob("*.jpg"), key=image_index)
    total = len(image_paths)

    for current, image_path in enumerate(image_paths, start=1):
        print(f"[{current}/{total}] Imagine {image_path.stem}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            continue

        crop, _ = crop_ultrasound(image)
        preprocessing_result = preprocess_crop(crop)

        cv2.imwrite(
            str(OUTPUT_DIR / "00_CROP" / f"{image_path.stem}_crop.png"),
            crop,
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "01_GRAY" / f"{image_path.stem}_gray.png"),
            preprocessing_result["gray"],
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "02_PALETTE_7" / f"{image_path.stem}_palette_7.png"),
            preprocessing_result["palette_7"],
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "03_BINARY_TOP1" / f"{image_path.stem}_binary_top1.png"),
            preprocessing_result["binary_top1"],
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "04_BINARY_TOP2" / f"{image_path.stem}_binary_top2.png"),
            preprocessing_result["binary_top2"],
        )


if __name__ == "__main__":
    main()
