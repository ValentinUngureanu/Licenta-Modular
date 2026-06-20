import shutil
from pathlib import Path

import cv2

from crop import crop_ultrasound
from preprocessing import preprocess_crop
from traveler import build_traveler
from principal_component import build_principal_component
from principal_sanity import repair_principal_if_upper_artifact
from principal_selector import select_principal_by_lower_candidate
from horizontal_rescue import horizontal_rescue_before_secondary
from secondary_component import build_secondary_components
from gap_rescue import (
    draw_gap_merged_debug,
    draw_gap_rescue_debug,
    gap_rescue_after_secondary,
)


INPUT_DIR = Path("ORIGINAL_IMAGES")
OUTPUT_DIR = Path("RESULTS/GAP_RESCUE_CLEAN_TEST1")


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

        sanity_result = repair_principal_if_upper_artifact(
            binary_top2,
            principal_result["principal_mask"],
        )

        selector_result = select_principal_by_lower_candidate(
            binary_top2,
            sanity_result["principal_mask"],
        )

        principal_mask = selector_result["principal_mask"]

        horizontal_result = horizontal_rescue_before_secondary(
            binary_top2,
            principal_mask,
            traveler_points=traveler_points,
        )

        principal_after_horizontal_mask = horizontal_result["merged_mask"]
        binary_top2_guarded = horizontal_result["binary_top2_guarded"]

        secondary_result = build_secondary_components(
            binary_top1,
            binary_top2_guarded,
            principal_after_horizontal_mask,
            traveler_points,
        )

        secondary_mask = secondary_result["secondary_mask"]
        merged_before_gap = secondary_result["merged_mask"]

        gap_result = gap_rescue_after_secondary(
            binary_top2=binary_top2_guarded,
            principal_mask=principal_after_horizontal_mask,
            secondary_mask=secondary_mask,
            merged_mask=merged_before_gap,
            traveler_points=traveler_points,
        )

        stem = image_path.stem

        save(
            OUTPUT_DIR / "00_CROP" / f"{stem}_crop.png",
            crop,
        )

        save(
            OUTPUT_DIR / "01_PRINCIPAL_AFTER_HORIZONTAL" / f"{stem}_principal_after_horizontal.png",
            principal_after_horizontal_mask,
        )

        save(
            OUTPUT_DIR / "02_SECONDARY_MASK" / f"{stem}_secondary_mask.png",
            secondary_mask,
        )

        save(
            OUTPUT_DIR / "03_MERGED_BEFORE_GAP" / f"{stem}_merged_before_gap.png",
            merged_before_gap,
        )

        save(
            OUTPUT_DIR / "04_GAP_ROI" / f"{stem}_gap_roi.png",
            gap_result["roi_mask"],
        )

        save(
            OUTPUT_DIR / "05_GAP_CANDIDATE" / f"{stem}_gap_candidate.png",
            gap_result["candidate_mask"],
        )

        save(
            OUTPUT_DIR / "06_GAP_ACCEPTED" / f"{stem}_gap_accepted.png",
            gap_result["accepted_mask"],
        )

        save(
            OUTPUT_DIR / "07_GAP_REJECTED" / f"{stem}_gap_rejected.png",
            gap_result["rejected_mask"],
        )

        save(
            OUTPUT_DIR / "08_GAP_RESCUE" / f"{stem}_gap_rescue.png",
            gap_result["rescue_mask"],
        )

        save(
            OUTPUT_DIR / "09_MERGED_AFTER_GAP" / f"{stem}_merged_after_gap.png",
            gap_result["merged_mask"],
        )

        save(
            OUTPUT_DIR / "10_GAP_DEBUG" / f"{stem}_gap_debug.png",
            draw_gap_rescue_debug(
                crop,
                principal_after_horizontal_mask,
                secondary_mask,
                gap_result["rescue_mask"],
                gap_result["roi_mask"],
                gap_result["candidate_mask"],
                gap_result["accepted_mask"],
                gap_result["rejected_mask"],
                traveler_points=traveler_points,
            ),
        )

        save(
            OUTPUT_DIR / "11_GAP_MERGED_DEBUG" / f"{stem}_gap_merged_debug.png",
            draw_gap_merged_debug(
                crop,
                gap_result["merged_mask"],
                traveler_points=traveler_points,
            ),
        )


if __name__ == "__main__":
    main()
