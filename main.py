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
from gap_rescue import gap_rescue_after_secondary
from secondary_rescue import (
    draw_merged_after_rescue,
    draw_rescue_debug,
    rescue_after_secondary,
)


INPUT_DIR = Path("ORIGINAL_IMAGES")
OUTPUT_DIR = Path("RESULTS/SECONDARY_RESCUE_CLEAN_TEST1")


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

        gap_result = gap_rescue_after_secondary(
            binary_top2=binary_top2_guarded,
            principal_mask=principal_after_horizontal_mask,
            secondary_mask=secondary_result["secondary_mask"],
            merged_mask=secondary_result["merged_mask"],
            traveler_points=traveler_points,
        )

        rescue_result = rescue_after_secondary(
            binary_top2_guarded,
            principal_after_horizontal_mask,
            gap_result["secondary_mask"],
            gap_result["merged_mask"],
            traveler_points,
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
            OUTPUT_DIR / "02_SECONDARY_AFTER_GAP" / f"{stem}_secondary_after_gap.png",
            gap_result["secondary_mask"],
        )

        save(
            OUTPUT_DIR / "03_MERGED_BEFORE_RESCUE" / f"{stem}_merged_before_rescue.png",
            gap_result["merged_mask"],
        )

        save(
            OUTPUT_DIR / "04_RESCUE_ROI" / f"{stem}_rescue_roi.png",
            rescue_result["roi_mask"],
        )

        save(
            OUTPUT_DIR / "05_RESCUE_CANDIDATE" / f"{stem}_rescue_candidate.png",
            rescue_result["candidate_mask"],
        )

        save(
            OUTPUT_DIR / "06_RESCUE_ACCEPTED" / f"{stem}_rescue_accepted.png",
            rescue_result["accepted_mask"],
        )

        save(
            OUTPUT_DIR / "07_RESCUE_REJECTED" / f"{stem}_rescue_rejected.png",
            rescue_result["rejected_mask"],
        )

        save(
            OUTPUT_DIR / "08_REMOVED_SECONDARY" / f"{stem}_removed_secondary.png",
            rescue_result["removed_secondary_mask"],
        )

        save(
            OUTPUT_DIR / "09_RESCUE_MASK" / f"{stem}_rescue_mask.png",
            rescue_result["rescue_mask"],
        )

        save(
            OUTPUT_DIR / "10_MERGED_AFTER_RESCUE" / f"{stem}_merged_after_rescue.png",
            rescue_result["merged_mask"],
        )

        save(
            OUTPUT_DIR / "11_RESCUE_DEBUG" / f"{stem}_rescue_debug.png",
            draw_rescue_debug(
                crop,
                principal_after_horizontal_mask,
                rescue_result["secondary_mask"],
                rescue_result["rescue_mask"],
                removed_secondary_mask=rescue_result["removed_secondary_mask"],
                roi_mask=rescue_result["roi_mask"],
                candidate_mask=rescue_result["candidate_mask"],
                rejected_mask=rescue_result["rejected_mask"],
                traveler_points=traveler_points,
            ),
        )

        save(
            OUTPUT_DIR / "12_MERGED_DEBUG" / f"{stem}_merged_debug.png",
            draw_merged_after_rescue(
                crop,
                rescue_result["merged_mask"],
                traveler_points=traveler_points,
            ),
        )


if __name__ == "__main__":
    main()
