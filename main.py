import shutil
from pathlib import Path

import cv2

from crop import crop_ultrasound, draw_crop_box_on_original


INPUT_DIR = Path("ORIGINAL_IMAGES")
OUTPUT_DIR = Path("RESULTS/CROP_CLEAN_TEST1")


def image_index(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 10**9


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(INPUT_DIR.glob("*.jpg"), key=image_index)

    total = len(image_paths)

    for current, image_path in enumerate(image_paths, start=1):
        print(f"[{current}/{total}] Imagine {image_path.stem}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            continue

        crop, crop_box = crop_ultrasound(image)
        crop_debug = draw_crop_box_on_original(image, crop_box)

        cv2.imwrite(str(OUTPUT_DIR / f"{image_path.stem}_crop.png"), crop)
        cv2.imwrite(str(OUTPUT_DIR / f"{image_path.stem}_crop_box.png"), crop_debug)


if __name__ == "__main__":
    main()
