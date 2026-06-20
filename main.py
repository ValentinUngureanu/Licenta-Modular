import shutil
from pathlib import Path

import cv2

from crop import crop_ultrasound
from preprocessing import preprocess_crop
from traveler import build_traveler
from principal_component import build_principal_component
from principal_sanity import (
    draw_principal_sanity_debug,
    repair_principal_if_upper_artifact,
)


INPUT_DIR = Path("ORIGINAL_IMAGES")
OUTPUT_DIR = Path("RESULTS/PRINCIPAL_SANITY_CLEAN_TEST1")


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

        sanity_result = repair_principal_if_upper_artifact(
            binary_top2,
            original_principal_mask,
        )

        repaired_principal_mask = sanity_result["principal_mask"]
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
            OUTPUT_DIR / "02_SANITY_ROI" / f"{stem}_sanity_roi.png",
            sanity_result["roi_mask"],
        )

        save(
            OUTPUT_DIR / "03_SANITY_CANDIDATE" / f"{stem}_sanity_candidate.png",
            sanity_result["candidate_mask"],
        )

        save(
            OUTPUT_DIR / "04_SANITY_REPLACEMENT" / f"{stem}_sanity_replacement.png",
            sanity_result["replacement_mask"],
        )

        save(
            OUTPUT_DIR / "05_REPAIRED_PRINCIPAL" / f"{stem}_repaired_principal.png",
            repaired_principal_mask,
        )

        save(
            OUTPUT_DIR / "06_SANITY_DEBUG" / f"{stem}_sanity_debug.png",
            draw_principal_sanity_debug(
                crop,
                original_principal_mask,
                repaired_principal_mask,
                sanity_result["roi_mask"],
                sanity_result["candidate_mask"],
                sanity_result["rejected_mask"],
            ),
        )


if __name__ == "__main__":
    main()
