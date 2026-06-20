import shutil
from pathlib import Path

import cv2

from crop import crop_ultrasound
from preprocessing import preprocess_crop
from traveler import build_traveler
from principal_component import build_principal_component
from principal_selector import (
    draw_principal_selector_debug,
    select_principal_by_lower_candidate,
)


INPUT_DIR = Path("ORIGINAL_IMAGES")
OUTPUT_DIR = Path("RESULTS/PRINCIPAL_SELECTOR_CLEAN_TEST1")


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

        original_principal_mask = principal_result["principal_mask"]

        selector_result = select_principal_by_lower_candidate(
            binary_top2,
            original_principal_mask,
        )

        selected_principal_mask = selector_result["principal_mask"]
        stem = image_path.stem

        save(
            OUTPUT_DIR / "00_CROP" / f"{stem}_crop.png",
            crop,
        )

        save(
            OUTPUT_DIR / "01_ORIGINAL_PRINCIPAL" / f"{stem}_original_principal.png",
            original_principal_mask,
        )

        save(
            OUTPUT_DIR / "02_SELECTOR_ROI" / f"{stem}_selector_roi.png",
            selector_result["roi_mask"],
        )

        save(
            OUTPUT_DIR / "03_SELECTOR_SEARCH" / f"{stem}_selector_search.png",
            selector_result["search_mask"],
        )

        save(
            OUTPUT_DIR / "04_SELECTOR_CANDIDATE" / f"{stem}_selector_candidate.png",
            selector_result["candidate_mask"],
        )

        save(
            OUTPUT_DIR / "05_SELECTOR_SELECTED" / f"{stem}_selector_selected.png",
            selector_result["selected_mask"],
        )

        save(
            OUTPUT_DIR / "06_FINAL_PRINCIPAL" / f"{stem}_final_principal.png",
            selected_principal_mask,
        )

        save(
            OUTPUT_DIR / "07_SELECTOR_DEBUG" / f"{stem}_selector_debug.png",
            draw_principal_selector_debug(
                crop,
                original_principal_mask,
                selector_result,
                traveler_points,
            ),
        )


if __name__ == "__main__":
    main()
