import shutil
from pathlib import Path

import cv2

from crop import crop_ultrasound
from preprocessing import preprocess_crop
from traveler import (
    build_traveler,
    draw_clean_selected_component,
    draw_extended_component,
    draw_raw_traveler_points,
    draw_selected_component,
    draw_traveler_components_colored,
)


INPUT_DIR = Path("ORIGINAL_IMAGES")
OUTPUT_DIR = Path("RESULTS/TRAVELER_CLEAN_TEST1")


def image_index(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 10**9


def save(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    image_paths = sorted(INPUT_DIR.glob("*.jpg"), key=image_index)
    total = len(image_paths)

    for current, image_path in enumerate(image_paths, start=1):
        print(f"[{current}/{total}] Imagine {image_path.stem}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            continue

        crop, _ = crop_ultrasound(image)
        preprocessing_result = preprocess_crop(crop)

        binary_top1 = preprocessing_result["binary_top1"]
        traveler_result = build_traveler(binary_top1)

        raw_points = traveler_result["raw_points"]
        components = traveler_result["components"]
        selected_points = traveler_result["selected_points"]
        cleaned_points = traveler_result["cleaned_points"]
        removed_points = traveler_result["removed_points"]
        extended_points = traveler_result["extended_points"]
        added_component_ids = traveler_result["added_component_ids"]

        stem = image_path.stem

        save(
            OUTPUT_DIR / "00_CROP" / f"{stem}_crop.png",
            crop,
        )

        save(
            OUTPUT_DIR / "01_BINARY_TOP1" / f"{stem}_binary_top1.png",
            binary_top1,
        )

        save(
            OUTPUT_DIR / "02_RAW_TRAVELER" / f"{stem}_raw_traveler.png",
            draw_raw_traveler_points(crop, raw_points),
        )

        save(
            OUTPUT_DIR / "03_COMPONENTS" / f"{stem}_components.png",
            draw_traveler_components_colored(crop, components),
        )

        save(
            OUTPUT_DIR / "04_SELECTED" / f"{stem}_selected.png",
            draw_selected_component(crop, raw_points, selected_points),
        )

        save(
            OUTPUT_DIR / "05_CLEANED" / f"{stem}_cleaned.png",
            draw_clean_selected_component(
                crop,
                selected_points,
                cleaned_points,
                removed_points,
            ),
        )

        save(
            OUTPUT_DIR / "06_EXTENDED" / f"{stem}_extended.png",
            draw_extended_component(
                crop,
                raw_points,
                extended_points,
                components,
                added_component_ids,
            ),
        )


if __name__ == "__main__":
    main()
