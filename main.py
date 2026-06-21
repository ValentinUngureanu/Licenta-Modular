import re
import shutil

import cv2
import numpy as np

import config
from crop import crop_ultrasound
from final_contour import (
    build_final_contour,
    draw_final_contour_on_crop,
    draw_final_contour_on_original,
    draw_final_mask_on_crop,
    draw_final_mask_on_original,
    project_mask_to_original,
    remove_very_small_components,
)
from gap_rescue import (
    draw_gap_rescue_debug,
    filter_floating_right_gap_rescue,
    filter_upper_right_gap_rescue,
    gap_rescue_after_secondary,
)
from horizontal_rescue import (
    draw_horizontal_rescue_debug,
    filter_floating_upper_horizontal_strip,
    filter_layered_horizontal_tail,
    filter_right_isolated_horizontal_component,
    run_guarded_horizontal_rescue,
)
from image_io import (
    ensure_dir,
    find_image_path,
    get_indices_to_process,
    make_output_name,
    read_image_bgr,
    save_image,
)
from preprocessing import preprocess_crop
from principal_component import (
    build_principal_component,
    draw_principal_component,
)
from principal_selector import (
    draw_principal_selector_debug,
    select_principal_by_lower_candidate,
)
from secondary_component import (
    build_secondary_components,
    draw_merged_components,
    draw_secondary_candidates,
    draw_secondary_components,
    draw_secondary_roi,
    filter_secondary_floating_strip_after_horizontal_reject,
    filter_secondary_tail_after_horizontal,
)
from secondary_rescue import (
    draw_merged_after_rescue as draw_secondary_rescue_merged_debug,
)
from secondary_rescue import (
    draw_rescue_debug as draw_secondary_rescue_debug,
)
from secondary_rescue import (
    filter_right_isolated_secondary_rescue,
    run_guarded_secondary_rescue,
)
from top2_final_contour import build_top2_final_contour
from top2_pleura import build_top2_guided_pleura
from traveler import build_traveler, draw_extended_component

FINAL_TEST_DIR = config.RESULTS_DIR / "06_FINAL_CONTOUR_TEST"

CROP_DIR = FINAL_TEST_DIR / "00_CROP"
PALETTE_7_DIR = FINAL_TEST_DIR / "01_PALETTE_7"
BINARY_TOP1_DIR = FINAL_TEST_DIR / "02_BINARY_TOP1"
BINARY_TOP2_DIR = FINAL_TEST_DIR / "03_BINARY_TOP2_SECONDARY_SEARCH"
TRAVELER_DIR = FINAL_TEST_DIR / "04_EXTENDED_TRAVELER"
PRINCIPAL_DIR = FINAL_TEST_DIR / "05_PRINCIPAL_COMPONENT"
PRINCIPAL_SELECTOR_DIR = FINAL_TEST_DIR / "05B_PRINCIPAL_SELECTOR"
HORIZONTAL_RESCUE_DIR = FINAL_TEST_DIR / "06_HORIZONTAL_RESCUE_BEFORE_SECONDARY"
BINARY_TOP2_GUARDED_DIR = FINAL_TEST_DIR / "07_BINARY_TOP2_GUARDED_FOR_SECONDARY"
SECONDARY_ROI_DIR = FINAL_TEST_DIR / "08_SECONDARY_SEARCH_ROI_TOP2"
SECONDARY_CANDIDATES_DIR = FINAL_TEST_DIR / "09_SECONDARY_TOP2_CANDIDATES_IN_ROI"
SECONDARY_DIR = FINAL_TEST_DIR / "10_PRINCIPAL_HORIZONTAL_SECONDARY_COMPONENTS"
GAP_RESCUE_DIR = FINAL_TEST_DIR / "11_GAP_RESCUE_AFTER_SECONDARY"
MERGED_BEFORE_SECONDARY_RESCUE_DIR = (
    FINAL_TEST_DIR / "12_MERGED_BEFORE_SECONDARY_RESCUE"
)
SECONDARY_RESCUE_DIR = FINAL_TEST_DIR / "13_SECONDARY_RESCUE_AFTER_SECONDARY"
MERGED_FINAL_DIR = FINAL_TEST_DIR / "14_MERGED_FINAL_AFTER_ALL_RESCUES"
SMALL_COMPONENT_CLEAN_DIR = FINAL_TEST_DIR / "14B_SMALL_COMPONENT_CLEAN"
SOURCE_ORIGIN_DEBUG_DIR = FINAL_TEST_DIR / "14C_SOURCE_ORIGIN_DEBUG"
FINAL_MASK_CROP_DIR = FINAL_TEST_DIR / "15_FINAL_MASK_ON_CROP"
FINAL_CONTOUR_CROP_DIR = FINAL_TEST_DIR / "16_FINAL_CONTOUR_ON_CROP"
FINAL_MASK_ORIGINAL_DIR = FINAL_TEST_DIR / "17_FINAL_MASK_ON_ORIGINAL"
FINAL_CONTOUR_ORIGINAL_DIR = FINAL_TEST_DIR / "18_FINAL_CONTOUR_ON_ORIGINAL"
FINAL_BINARY_ORIGINAL_DIR = FINAL_TEST_DIR / "19_FINAL_BINARY_MASK_ORIGINAL"
TOP2_CONTOUR_ONLY_DIR = FINAL_TEST_DIR / "21_TOP2_FINAL_CONTOUR_ONLY_ON_CROP"
REST_CONTACT_SHEET_PATH = config.RESULTS_DIR / "00_TOATE_POZELE_FINAL_CONTOUR.jpg"

from postprocessing import (
    draw_artifact_source_overlay,
    filter_left_low_far_artifact_components,
    filter_principal_tail_under_horizontal,
    mask_area,
    merge_masks,
)


def process_image(index: int, current: int, total: int) -> None:
    print(f"[{current}/{total}] Imagine {index}")

    image_path = find_image_path(index)

    if image_path is None:
        return

    image = read_image_bgr(image_path)
    crop, crop_box = crop_ultrasound(image)

    preprocessing_result = preprocess_crop(crop)
    palette_7 = preprocessing_result["palette_7"]
    binary_top1 = preprocessing_result["binary_top1"]
    binary_top2 = preprocessing_result["binary_top2"]

    traveler_result = build_traveler(binary_top1)
    raw_points = traveler_result["raw_points"]
    components = traveler_result["components"]
    extended_points = traveler_result["extended_points"]
    added_component_ids = traveler_result["added_component_ids"]

    principal_result = build_principal_component(
        binary_top1,
        extended_points,
    )

    principal_mask_initial = principal_result["principal_mask"]
    rejected_principal_mask = principal_result["rejected_mask"]

    principal_selector_result = select_principal_by_lower_candidate(
        binary_top2,
        principal_mask_initial,
    )
    principal_mask = principal_selector_result["principal_mask"]

    principal_mask, left_low_principal_removed_mask = (
        filter_left_low_far_artifact_components(
            principal_mask,
            principal_mask,
        )
    )

    if mask_area(left_low_principal_removed_mask) > 0:
        rejected_principal_mask = merge_masks(
            rejected_principal_mask,
            left_low_principal_removed_mask,
        )

    horizontal_result = run_guarded_horizontal_rescue(
        binary_top2,
        principal_mask,
        traveler_points=extended_points,
    )

    horizontal_rescue_mask = horizontal_result["rescue_mask"]
    binary_top2_guarded = horizontal_result["binary_top2_guarded"]
    horizontal_roi_mask = horizontal_result["roi_mask"]
    horizontal_candidate_mask = horizontal_result["candidate_mask"]
    horizontal_rejected_mask = horizontal_result["rejected_mask"]

    horizontal_rescue_mask, horizontal_layered_tail_removed_mask = (
        filter_layered_horizontal_tail(
            horizontal_rescue_mask,
            principal_mask,
        )
    )

    horizontal_rescue_mask, horizontal_floating_strip_removed_mask = (
        filter_floating_upper_horizontal_strip(
            horizontal_rescue_mask,
            principal_mask,
        )
    )

    horizontal_rescue_mask, horizontal_right_isolated_removed_mask = (
        filter_right_isolated_horizontal_component(
            horizontal_rescue_mask,
            principal_mask,
        )
    )

    horizontal_removed_mask = merge_masks(
        horizontal_layered_tail_removed_mask,
        horizontal_floating_strip_removed_mask,
    )
    horizontal_removed_mask = merge_masks(
        horizontal_removed_mask,
        horizontal_right_isolated_removed_mask,
    )

    horizontal_accepted_mask = horizontal_rescue_mask
    horizontal_rejected_mask = merge_masks(
        horizontal_rejected_mask,
        horizontal_removed_mask,
    )

    if mask_area(horizontal_removed_mask) > 0:
        binary_top2_guarded = binary_top2_guarded.copy()
        binary_top2_guarded[horizontal_removed_mask > 0] = 0

    principal_mask, principal_under_horizontal_removed_mask = (
        filter_principal_tail_under_horizontal(
            principal_mask,
            horizontal_rescue_mask,
        )
    )

    if mask_area(principal_under_horizontal_removed_mask) > 0:
        rejected_principal_mask = merge_masks(
            rejected_principal_mask,
            principal_under_horizontal_removed_mask,
        )

        binary_top2_guarded = binary_top2_guarded.copy()
        binary_top2_guarded[principal_under_horizontal_removed_mask > 0] = 0

    principal_after_horizontal_mask = merge_masks(
        principal_mask,
        horizontal_rescue_mask,
    )

    secondary_result = build_secondary_components(
        binary_top1,
        binary_top2_guarded,
        principal_after_horizontal_mask,
        extended_points,
    )

    secondary_mask_normal = secondary_result["secondary_mask"]
    secondary_roi_mask = secondary_result["roi_mask"]
    secondary_candidate_mask = secondary_result["candidate_mask"]
    secondary_rejected_mask = secondary_result["rejected_mask"]

    secondary_mask_normal, secondary_tail_after_horizontal_removed_mask = (
        filter_secondary_tail_after_horizontal(
            secondary_mask_normal,
            horizontal_rescue_mask,
        )
    )

    secondary_mask_normal, secondary_floating_readd_removed_mask = (
        filter_secondary_floating_strip_after_horizontal_reject(
            secondary_mask_normal,
            principal_mask,
        )
    )

    secondary_mask_normal, left_low_secondary_removed_mask = (
        filter_left_low_far_artifact_components(
            secondary_mask_normal,
            principal_mask,
        )
    )

    secondary_total_removed_mask = merge_masks(
        secondary_tail_after_horizontal_removed_mask,
        secondary_floating_readd_removed_mask,
    )
    secondary_total_removed_mask = merge_masks(
        secondary_total_removed_mask,
        left_low_secondary_removed_mask,
    )

    secondary_rejected_mask = merge_masks(
        secondary_rejected_mask,
        secondary_total_removed_mask,
    )

    if mask_area(secondary_total_removed_mask) > 0:
        binary_top2_guarded = binary_top2_guarded.copy()
        binary_top2_guarded[secondary_total_removed_mask > 0] = 0

    merged_before_gap_rescue = merge_masks(
        principal_after_horizontal_mask,
        secondary_mask_normal,
    )

    gap_rescue_result = gap_rescue_after_secondary(
        binary_top2=binary_top2_guarded,
        principal_mask=principal_after_horizontal_mask,
        secondary_mask=secondary_mask_normal,
        merged_mask=merged_before_gap_rescue,
        traveler_points=extended_points,
    )

    gap_rescue_mask_raw = gap_rescue_result["rescue_mask"]

    gap_rescue_mask, floating_gap_removed_mask = filter_floating_right_gap_rescue(
        gap_rescue_mask_raw,
        principal_after_horizontal_mask,
        secondary_mask_normal,
    )

    gap_rescue_mask, upper_right_gap_removed_mask = filter_upper_right_gap_rescue(
        gap_rescue_mask,
        principal_after_horizontal_mask,
        secondary_mask_normal,
    )

    floating_gap_removed_mask = merge_masks(
        floating_gap_removed_mask,
        upper_right_gap_removed_mask,
    )

    secondary_mask_before_rescue = np.zeros_like(secondary_mask_normal, dtype=np.uint8)
    secondary_mask_before_rescue[secondary_mask_normal > 0] = 255
    secondary_mask_before_rescue[gap_rescue_mask > 0] = 255

    secondary_mask_before_rescue, combined_secondary_gap_removed_mask = (
        filter_secondary_floating_strip_after_horizontal_reject(
            secondary_mask_before_rescue,
            principal_mask,
        )
    )

    secondary_mask_before_rescue, left_low_combined_removed_mask = (
        filter_left_low_far_artifact_components(
            secondary_mask_before_rescue,
            principal_mask,
        )
    )

    combined_secondary_gap_removed_mask = merge_masks(
        combined_secondary_gap_removed_mask,
        left_low_combined_removed_mask,
    )

    if mask_area(combined_secondary_gap_removed_mask) > 0:
        binary_top2_guarded = binary_top2_guarded.copy()
        binary_top2_guarded[combined_secondary_gap_removed_mask > 0] = 0

    merged_before_secondary_rescue = np.zeros_like(
        principal_after_horizontal_mask, dtype=np.uint8
    )
    merged_before_secondary_rescue[principal_after_horizontal_mask > 0] = 255
    merged_before_secondary_rescue[secondary_mask_before_rescue > 0] = 255

    gap_rescue_roi_mask = gap_rescue_result["roi_mask"]
    gap_rescue_candidate_mask = gap_rescue_result["candidate_mask"]
    gap_rescue_rejected_mask = gap_rescue_result["rejected_mask"]

    gap_rescue_rejected_mask = merge_masks(
        gap_rescue_rejected_mask,
        floating_gap_removed_mask,
    )
    gap_rescue_accepted_mask = gap_rescue_mask

    secondary_rescue_result = run_guarded_secondary_rescue(
        binary_top2_guarded=binary_top2_guarded,
        principal_after_horizontal_mask=principal_after_horizontal_mask,
        secondary_mask_before_rescue=secondary_mask_before_rescue,
        merged_before_secondary_rescue=merged_before_secondary_rescue,
        traveler_points=extended_points,
    )

    secondary_mask_after_rescue = secondary_rescue_result["secondary_mask"]
    secondary_rescue_mask_raw = secondary_rescue_result["rescue_mask"]

    secondary_rescue_mask, right_isolated_rescue_removed_mask = (
        filter_right_isolated_secondary_rescue(
            secondary_rescue_mask_raw,
            merged_before_secondary_rescue,
        )
    )

    secondary_rescue_mask, left_low_secondary_rescue_removed_mask = (
        filter_left_low_far_artifact_components(
            secondary_rescue_mask,
            principal_mask,
        )
    )

    merged_final_mask_before_small_clean = np.zeros_like(
        merged_before_secondary_rescue, dtype=np.uint8
    )
    merged_final_mask_before_small_clean[merged_before_secondary_rescue > 0] = 255
    merged_final_mask_before_small_clean[secondary_rescue_mask > 0] = 255

    merged_final_mask, small_component_removed_mask = remove_very_small_components(
        merged_final_mask_before_small_clean,
    )
    secondary_rescue_removed_mask = secondary_rescue_result["removed_secondary_mask"]
    secondary_rescue_removed_mask = merge_masks(
        secondary_rescue_removed_mask,
        right_isolated_rescue_removed_mask,
    )
    secondary_rescue_removed_mask = merge_masks(
        secondary_rescue_removed_mask,
        left_low_secondary_rescue_removed_mask,
    )
    secondary_rescue_roi_mask = secondary_rescue_result["roi_mask"]
    secondary_rescue_candidate_mask = secondary_rescue_result["candidate_mask"]
    secondary_rescue_rejected_mask = secondary_rescue_result["rejected_mask"]

    final_result = build_final_contour(merged_final_mask)
    final_mask = final_result["final_mask"]
    contour_points = final_result["contour_points"]

    traveler_debug = draw_extended_component(
        crop,
        raw_points,
        extended_points,
        components,
        added_component_ids,
    )

    principal_debug = draw_principal_component(
        crop,
        principal_mask,
        rejected_mask=rejected_principal_mask,
        traveler_points=extended_points,
    )

    principal_selector_debug = draw_principal_selector_debug(
        crop,
        principal_mask_initial,
        principal_selector_result,
        traveler_points=extended_points,
    )

    horizontal_rescue_debug = draw_horizontal_rescue_debug(
        crop,
        principal_mask,
        horizontal_rescue_mask,
        horizontal_roi_mask,
        horizontal_candidate_mask,
        horizontal_accepted_mask,
        horizontal_rejected_mask,
        traveler_points=extended_points,
    )

    secondary_roi_debug = draw_secondary_roi(
        crop,
        secondary_roi_mask,
        principal_mask=principal_after_horizontal_mask,
        traveler_points=extended_points,
    )

    secondary_candidates_debug = draw_secondary_candidates(
        crop,
        secondary_candidate_mask,
        rejected_mask=secondary_rejected_mask,
        traveler_points=extended_points,
    )

    secondary_debug = draw_secondary_components(
        crop,
        principal_after_horizontal_mask,
        secondary_mask_normal,
        traveler_points=extended_points,
    )

    gap_rescue_debug = draw_gap_rescue_debug(
        crop,
        principal_after_horizontal_mask,
        secondary_mask_normal,
        gap_rescue_mask,
        gap_rescue_roi_mask,
        gap_rescue_candidate_mask,
        gap_rescue_accepted_mask,
        gap_rescue_rejected_mask,
        traveler_points=extended_points,
    )

    merged_before_secondary_rescue_debug = draw_merged_components(
        crop,
        merged_before_secondary_rescue,
        traveler_points=extended_points,
    )

    secondary_rescue_debug = draw_secondary_rescue_debug(
        crop,
        principal_after_horizontal_mask,
        secondary_mask_after_rescue,
        secondary_rescue_mask,
        removed_secondary_mask=secondary_rescue_removed_mask,
        roi_mask=secondary_rescue_roi_mask,
        candidate_mask=secondary_rescue_candidate_mask,
        rejected_mask=secondary_rescue_rejected_mask,
        traveler_points=extended_points,
    )

    merged_final_debug = draw_secondary_rescue_merged_debug(
        crop,
        merged_final_mask_before_small_clean,
        traveler_points=extended_points,
    )

    small_component_clean_debug = draw_merged_components(
        crop,
        merged_final_mask,
        traveler_points=extended_points,
    )

    source_origin_debug = draw_artifact_source_overlay(
        crop,
        principal_after_horizontal_mask,
        horizontal_rescue_mask,
        secondary_mask_normal,
        gap_rescue_mask,
        secondary_rescue_mask,
        final_mask=merged_final_mask,
        traveler_points=extended_points,
    )

    final_mask_crop_debug = draw_final_mask_on_crop(
        crop,
        final_mask,
    )

    final_contour_crop_debug = draw_final_contour_on_crop(
        crop,
        final_mask,
        contour_points,
    )

    final_mask_original_debug = draw_final_mask_on_original(
        image,
        final_mask,
        crop_box,
    )

    final_contour_original_debug = draw_final_contour_on_original(
        image,
        final_mask,
        contour_points,
        crop_box,
    )

    final_binary_original = project_mask_to_original(
        final_mask,
        image.shape,
        crop_box,
    )

    top2_guided_result = build_top2_guided_pleura(
        crop_bgr=crop,
        binary_top1=binary_top1,
        binary_top2=binary_top2,
        current_pleura_mask=merged_final_mask,
    )
    top2_final_contour_result = build_top2_final_contour(
        crop_bgr=crop,
        current_pleura_mask=merged_final_mask,
        top2_guided_mask=top2_guided_result["top2_guided_mask"],
        top2_added_mask=top2_guided_result["top2_added_to_current"],
    )
    save_image(CROP_DIR / make_output_name(index, "crop"), crop)
    save_image(PALETTE_7_DIR / make_output_name(index, "palette_7"), palette_7)
    save_image(BINARY_TOP1_DIR / make_output_name(index, "binary_top1"), binary_top1)
    save_image(
        BINARY_TOP2_DIR / make_output_name(index, "binary_top2_secondary_search"),
        binary_top2,
    )
    save_image(
        TRAVELER_DIR / make_output_name(index, "extended_traveler"), traveler_debug
    )
    save_image(
        PRINCIPAL_DIR / make_output_name(index, "principal_component"), principal_debug
    )
    save_image(
        PRINCIPAL_SELECTOR_DIR / make_output_name(index, "principal_selector"),
        principal_selector_debug,
    )
    save_image(
        HORIZONTAL_RESCUE_DIR
        / make_output_name(index, "horizontal_rescue_before_secondary"),
        horizontal_rescue_debug,
    )
    save_image(
        BINARY_TOP2_GUARDED_DIR
        / make_output_name(index, "binary_top2_guarded_for_secondary"),
        binary_top2_guarded,
    )
    save_image(
        SECONDARY_ROI_DIR / make_output_name(index, "secondary_search_roi_top2"),
        secondary_roi_debug,
    )
    save_image(
        SECONDARY_CANDIDATES_DIR
        / make_output_name(index, "secondary_top2_candidates_in_roi"),
        secondary_candidates_debug,
    )
    save_image(
        SECONDARY_DIR
        / make_output_name(index, "principal_horizontal_secondary_components"),
        secondary_debug,
    )
    save_image(
        GAP_RESCUE_DIR / make_output_name(index, "gap_rescue_after_secondary"),
        gap_rescue_debug,
    )
    save_image(
        MERGED_BEFORE_SECONDARY_RESCUE_DIR
        / make_output_name(index, "merged_before_secondary_rescue"),
        merged_before_secondary_rescue_debug,
    )
    save_image(
        SECONDARY_RESCUE_DIR
        / make_output_name(index, "secondary_rescue_after_secondary"),
        secondary_rescue_debug,
    )
    save_image(
        MERGED_FINAL_DIR / make_output_name(index, "merged_final_after_all_rescues"),
        merged_final_debug,
    )
    save_image(
        SMALL_COMPONENT_CLEAN_DIR / make_output_name(index, "small_component_clean"),
        small_component_clean_debug,
    )
    save_image(
        SOURCE_ORIGIN_DEBUG_DIR / make_output_name(index, "source_origin_debug"),
        source_origin_debug,
    )
    save_image(
        FINAL_MASK_CROP_DIR / make_output_name(index, "final_mask_on_crop"),
        final_mask_crop_debug,
    )
    save_image(
        FINAL_CONTOUR_CROP_DIR / make_output_name(index, "final_contour_on_crop"),
        final_contour_crop_debug,
    )
    save_image(
        FINAL_MASK_ORIGINAL_DIR / make_output_name(index, "final_mask_on_original"),
        final_mask_original_debug,
    )
    save_image(
        FINAL_CONTOUR_ORIGINAL_DIR
        / make_output_name(index, "final_contour_on_original"),
        final_contour_original_debug,
    )
    save_image(
        FINAL_BINARY_ORIGINAL_DIR
        / make_output_name(index, "final_binary_mask_original"),
        final_binary_original,
    )

    top2_contour_only = top2_final_contour_result["contour_on_crop"]
    save_image(
        TOP2_CONTOUR_ONLY_DIR
        / make_output_name(index, "top2_final_contour_only_on_crop"),
        top2_contour_only,
    )


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)
    return path


def make_final_contours_contact_sheet(exclude_indices=None):
    if exclude_indices is None:
        exclude_indices = set()

    image_paths = []

    for path in sorted(
        FINAL_CONTOUR_ORIGINAL_DIR.glob("*_final_contour_on_original.png")
    ):
        match = re.match(r"(\d+)_", path.name)

        if match is None:
            continue

        index = int(match.group(1))

        if index in exclude_indices:
            continue

        image_paths.append((index, path))

    if len(image_paths) == 0:
        return

    thumbs = []
    tile_width = 360
    tile_height = 230
    label_height = 28

    for index, path in image_paths:
        image = cv2.imread(str(path))

        if image is None:
            continue

        height, width = image.shape[:2]
        scale = min(tile_width / max(width, 1), tile_height / max(height, 1))

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        tile = np.zeros((tile_height + label_height, tile_width, 3), dtype=np.uint8)

        y0 = label_height + (tile_height - new_height) // 2
        x0 = (tile_width - new_width) // 2
        tile[y0 : y0 + new_height, x0 : x0 + new_width] = resized

        cv2.putText(
            tile,
            f"Imagine {index}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        thumbs.append(tile)

    if len(thumbs) == 0:
        return

    cols = 5
    rows = int(np.ceil(len(thumbs) / cols))

    sheet_height = rows * (tile_height + label_height)
    sheet_width = cols * tile_width
    sheet = np.zeros((sheet_height, sheet_width, 3), dtype=np.uint8)

    for pos, tile in enumerate(thumbs):
        row = pos // cols
        col = pos % cols

        y0 = row * (tile_height + label_height)
        x0 = col * tile_width

        sheet[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile

    save_image(REST_CONTACT_SHEET_PATH, sheet)


def main() -> None:
    reset_dir(config.RESULTS_DIR)

    ensure_dir(CROP_DIR)
    ensure_dir(PALETTE_7_DIR)
    ensure_dir(BINARY_TOP1_DIR)
    ensure_dir(BINARY_TOP2_DIR)
    ensure_dir(TRAVELER_DIR)
    ensure_dir(PRINCIPAL_DIR)
    ensure_dir(PRINCIPAL_SELECTOR_DIR)
    ensure_dir(HORIZONTAL_RESCUE_DIR)
    ensure_dir(BINARY_TOP2_GUARDED_DIR)
    ensure_dir(SECONDARY_ROI_DIR)
    ensure_dir(SECONDARY_CANDIDATES_DIR)
    ensure_dir(SECONDARY_DIR)
    ensure_dir(GAP_RESCUE_DIR)
    ensure_dir(MERGED_BEFORE_SECONDARY_RESCUE_DIR)
    ensure_dir(SECONDARY_RESCUE_DIR)
    ensure_dir(MERGED_FINAL_DIR)
    ensure_dir(SMALL_COMPONENT_CLEAN_DIR)
    ensure_dir(SOURCE_ORIGIN_DEBUG_DIR)
    ensure_dir(FINAL_MASK_CROP_DIR)
    ensure_dir(FINAL_CONTOUR_CROP_DIR)
    ensure_dir(FINAL_MASK_ORIGINAL_DIR)
    ensure_dir(FINAL_CONTOUR_ORIGINAL_DIR)
    ensure_dir(FINAL_BINARY_ORIGINAL_DIR)
    ensure_dir(TOP2_CONTOUR_ONLY_DIR)

    indices = get_indices_to_process()

    if len(indices) == 0:
        print("Nu s-au gasit imagini de procesat.")
        return

    total = len(indices)

    for current, index in enumerate(indices, start=1):
        process_image(index, current, total)

    make_final_contours_contact_sheet(exclude_indices=set())


if __name__ == "__main__":
    main()
