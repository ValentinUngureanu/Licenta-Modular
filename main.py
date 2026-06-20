import shutil
from pathlib import Path

import cv2

from crop import crop_ultrasound
from preprocessing import preprocess_crop
from traveler import build_traveler
from principal_component import (
    build_principal_component,
    draw_candidate_mask,
    draw_principal_component,
    draw_principal_roi,
)


INPUT_DIR = Path("ORIGINAL_IMAGES")
OUTPUT_DIR = Path("RESULTS/PRINCIPAL_COMPONENT_CLEAN_TEST1")


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
        traveler_points = traveler_result["extended_points"]

        principal_result = build_principal_component(
            binary_top1,
            traveler_points,
        )

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
            OUTPUT_DIR / "02_ROI_MASK" / f"{stem}_roi_mask.png",
            principal_result["roi_mask"],
        )

        save(
            OUTPUT_DIR / "03_CANDIDATE_MASK" / f"{stem}_candidate_mask.png",
            principal_result["candidate_mask"],
        )

        save(
            OUTPUT_DIR / "04_PRINCIPAL_MASK" / f"{stem}_principal_mask.png",
            principal_result["principal_mask"],
        )

        save(
            OUTPUT_DIR / "05_REJECTED_MASK" / f"{stem}_rejected_mask.png",
            principal_result["rejected_mask"],
        )

        save(
            OUTPUT_DIR / "06_ROI_OVERLAY" / f"{stem}_roi_overlay.png",
            draw_principal_roi(
                crop,
                principal_result["roi_mask"],
                traveler_points,
            ),
        )

        save(
            OUTPUT_DIR / "07_CANDIDATE_OVERLAY" / f"{stem}_candidate_overlay.png",
            draw_candidate_mask(
                crop,
                principal_result["candidate_mask"],
                traveler_points,
            ),
        )

        save(
            OUTPUT_DIR / "08_PRINCIPAL_OVERLAY" / f"{stem}_principal_overlay.png",
            draw_principal_component(
                crop,
                principal_result["principal_mask"],
                principal_result["rejected_mask"],
                traveler_points,
            ),
        )


if __name__ == "__main__":
    main()
