import shutil
from pathlib import Path

import cv2

from crop import crop_ultrasound
from preprocessing import preprocess_crop
from traveler import build_traveler
from principal_component import build_principal_component
from principal_selector import select_principal_by_lower_candidate
from horizontal_rescue import (
    draw_horizontal_merged_debug,
    draw_horizontal_rescue_debug,
    horizontal_rescue_before_secondary,
)


INPUT_DIR = Path("ORIGINAL_IMAGES")
OUTPUT_DIR = Path("RESULTS/HORIZONTAL_RESCUE_CLEAN_TEST1")


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
        binary_top2 = preprocessing_result["binary_top2"]

        traveler_result = build_traveler(binary_top1)
        traveler_points = traveler_result["extended_points"]

        principal_result = build_principal_component(
            binary_top1,
            traveler_points,
        )

        selector_result = select_principal_by_lower_candidate(
            binary_top2,
            principal_result["principal_mask"],
        )

        principal_mask = selector_result["principal_mask"]

        horizontal_result = horizontal_rescue_before_secondary(
            binary_top2,
            principal_mask,
            traveler_points=traveler_points,
        )

        stem = image_path.stem

        save(
            OUTPUT_DIR / "00_CROP" / f"{stem}_crop.png",
            crop,
        )

        save(
            OUTPUT_DIR / "01_PRINCIPAL_MASK" / f"{stem}_principal_mask.png",
            principal_mask,
        )

        save(
            OUTPUT_DIR / "02_HORIZONTAL_ROI" / f"{stem}_horizontal_roi.png",
            horizontal_result["roi_mask"],
        )

        save(
            OUTPUT_DIR / "03_HORIZONTAL_CANDIDATE" / f"{stem}_horizontal_candidate.png",
            horizontal_result["candidate_mask"],
        )

        save(
            OUTPUT_DIR / "04_HORIZONTAL_ACCEPTED" / f"{stem}_horizontal_accepted.png",
            horizontal_result["accepted_mask"],
        )

        save(
            OUTPUT_DIR / "05_HORIZONTAL_REJECTED" / f"{stem}_horizontal_rejected.png",
            horizontal_result["rejected_mask"],
        )

        save(
            OUTPUT_DIR / "06_HORIZONTAL_RESCUE" / f"{stem}_horizontal_rescue.png",
            horizontal_result["rescue_mask"],
        )

        save(
            OUTPUT_DIR / "07_MERGED_MASK" / f"{stem}_merged_mask.png",
            horizontal_result["merged_mask"],
        )

        save(
            OUTPUT_DIR / "08_HORIZONTAL_DEBUG" / f"{stem}_horizontal_debug.png",
            draw_horizontal_rescue_debug(
                crop,
                principal_mask,
                horizontal_result["rescue_mask"],
                horizontal_result["roi_mask"],
                horizontal_result["candidate_mask"],
                horizontal_result["accepted_mask"],
                horizontal_result["rejected_mask"],
                traveler_points=traveler_points,
            ),
        )

        save(
            OUTPUT_DIR / "09_MERGED_DEBUG" / f"{stem}_merged_debug.png",
            draw_horizontal_merged_debug(
                crop,
                horizontal_result["merged_mask"],
                traveler_points=traveler_points,
            ),
        )


if __name__ == "__main__":
    main()
