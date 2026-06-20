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
from secondary_component import (
    build_secondary_components,
    draw_merged_components,
    draw_secondary_candidates,
    draw_secondary_components,
    draw_secondary_roi,
)


INPUT_DIR = Path("ORIGINAL_IMAGES")
OUTPUT_DIR = Path("RESULTS/SECONDARY_COMPONENT_CLEAN_TEST1")


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
            OUTPUT_DIR / "02_SECONDARY_ROI" / f"{stem}_secondary_roi.png",
            secondary_result["roi_mask"],
        )

        save(
            OUTPUT_DIR / "03_SECONDARY_CANDIDATE" / f"{stem}_secondary_candidate.png",
            secondary_result["candidate_mask"],
        )

        save(
            OUTPUT_DIR / "04_SECONDARY_ACCEPTED" / f"{stem}_secondary_accepted.png",
            secondary_result["accepted_mask"],
        )

        save(
            OUTPUT_DIR / "05_SECONDARY_REJECTED" / f"{stem}_secondary_rejected.png",
            secondary_result["rejected_mask"],
        )

        save(
            OUTPUT_DIR / "06_SECONDARY_MASK" / f"{stem}_secondary_mask.png",
            secondary_result["secondary_mask"],
        )

        save(
            OUTPUT_DIR / "07_MERGED_MASK" / f"{stem}_merged_mask.png",
            secondary_result["merged_mask"],
        )

        save(
            OUTPUT_DIR / "08_SECONDARY_ROI_DEBUG" / f"{stem}_secondary_roi_debug.png",
            draw_secondary_roi(
                crop,
                secondary_result["roi_mask"],
                principal_after_horizontal_mask,
                traveler_points,
            ),
        )

        save(
            OUTPUT_DIR / "09_SECONDARY_CANDIDATES_DEBUG" / f"{stem}_secondary_candidates_debug.png",
            draw_secondary_candidates(
                crop,
                secondary_result["candidate_mask"],
                secondary_result["rejected_mask"],
                traveler_points,
            ),
        )

        save(
            OUTPUT_DIR / "10_SECONDARY_COMPONENTS_DEBUG" / f"{stem}_secondary_components_debug.png",
            draw_secondary_components(
                crop,
                principal_after_horizontal_mask,
                secondary_result["secondary_mask"],
                traveler_points,
            ),
        )

        save(
            OUTPUT_DIR / "11_MERGED_DEBUG" / f"{stem}_merged_debug.png",
            draw_merged_components(
                crop,
                secondary_result["merged_mask"],
                traveler_points,
            ),
        )


if __name__ == "__main__":
    main()
