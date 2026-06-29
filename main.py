import shutil

import numpy as np

import config
from crop import crop_ultrasound
from final_contour import remove_very_small_components
from gap_rescue import (
    filter_floating_right_gap_rescue,
    filter_upper_right_gap_rescue,
    gap_rescue_after_secondary,
)
from horizontal_rescue import (
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
from postprocessing import (
    filter_left_low_far_artifact_components,
    filter_principal_tail_under_horizontal,
    mask_area,
    merge_masks,
)
from preprocessing import preprocess_crop
from principal_component import build_principal_component
from principal_selector import select_principal_by_lower_candidate
from secondary_component import (
    build_secondary_components,
    filter_secondary_floating_strip_after_horizontal_reject,
    filter_secondary_tail_after_horizontal,
)
from secondary_rescue import (
    filter_right_isolated_secondary_rescue,
    run_guarded_secondary_rescue,
)
from top2_final_contour import build_top2_final_contour
from top2_pleura import build_top2_guided_pleura
from traveler import build_traveler
from unification import build_top2_unification_debug

# ============================================================
# MAIN - UNIFICATION SPECIAL OUTPUT
# ============================================================
# Scop:
#   - pastram pipeline-ul de identificare existent pana la top2_final_mask;
#   - nu salvam debug-ul vechi din pipeline;
#   - testam doar PASUL 9:
#       polyline netezita -> curatare coborari redundante;
#   - in acest pas curatam coborarile locale redundante;
#   - nu facem inca validare pe support_mask.
#
# Output:
#   RESULTS/56_UNIFICATION_NATURAL_BRIDGE_RANDOM_JAGGED_LIMITED
#       <index>_01_natural_bridge_random_jagged_limited.png
# ============================================================


UNIFICATION_SPECIAL_DIR = (
    config.RESULTS_DIR / "56_UNIFICATION_NATURAL_BRIDGE_RANDOM_JAGGED_LIMITED"
)


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)
    return path


def build_unification_input(index: int):
    image_path = find_image_path(index)

    if image_path is None:
        return None

    image = read_image_bgr(image_path)
    crop, _crop_box = crop_ultrasound(image)

    preprocessing_result = preprocess_crop(crop)
    binary_top1 = preprocessing_result["binary_top1"]
    binary_top2 = preprocessing_result["binary_top2"]

    traveler_result = build_traveler(binary_top1)
    extended_points = traveler_result["extended_points"]

    principal_result = build_principal_component(
        binary_top1,
        extended_points,
    )

    principal_mask_initial = principal_result["principal_mask"]

    principal_selector_result = select_principal_by_lower_candidate(
        binary_top2,
        principal_mask_initial,
    )
    principal_mask = principal_selector_result["principal_mask"]

    principal_mask, _left_low_principal_removed_mask = (
        filter_left_low_far_artifact_components(
            principal_mask,
            principal_mask,
        )
    )

    horizontal_result = run_guarded_horizontal_rescue(
        binary_top2,
        principal_mask,
        traveler_points=extended_points,
    )

    horizontal_rescue_mask = horizontal_result["rescue_mask"]
    binary_top2_guarded = horizontal_result["binary_top2_guarded"]

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
    combined_secondary_gap_removed_mask = merge_masks(
        combined_secondary_gap_removed_mask,
        floating_gap_removed_mask,
    )

    if mask_area(combined_secondary_gap_removed_mask) > 0:
        binary_top2_guarded = binary_top2_guarded.copy()
        binary_top2_guarded[combined_secondary_gap_removed_mask > 0] = 0

    merged_before_secondary_rescue = np.zeros_like(
        principal_after_horizontal_mask,
        dtype=np.uint8,
    )
    merged_before_secondary_rescue[principal_after_horizontal_mask > 0] = 255
    merged_before_secondary_rescue[secondary_mask_before_rescue > 0] = 255

    secondary_rescue_result = run_guarded_secondary_rescue(
        binary_top2_guarded=binary_top2_guarded,
        principal_after_horizontal_mask=principal_after_horizontal_mask,
        secondary_mask_before_rescue=secondary_mask_before_rescue,
        merged_before_secondary_rescue=merged_before_secondary_rescue,
        traveler_points=extended_points,
    )

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

    secondary_rescue_removed_mask = merge_masks(
        right_isolated_rescue_removed_mask,
        left_low_secondary_rescue_removed_mask,
    )

    if mask_area(secondary_rescue_removed_mask) > 0:
        binary_top2_guarded = binary_top2_guarded.copy()
        binary_top2_guarded[secondary_rescue_removed_mask > 0] = 0

    merged_final_mask_before_small_clean = np.zeros_like(
        merged_before_secondary_rescue,
        dtype=np.uint8,
    )
    merged_final_mask_before_small_clean[merged_before_secondary_rescue > 0] = 255
    merged_final_mask_before_small_clean[secondary_rescue_mask > 0] = 255

    merged_final_mask, _small_component_removed_mask = remove_very_small_components(
        merged_final_mask_before_small_clean,
    )

    top2_guided_result = build_top2_guided_pleura(
        binary_top1=binary_top1,
        binary_top2=binary_top2,
        current_pleura_mask=merged_final_mask,
    )

    top2_final_contour_result = build_top2_final_contour(
        crop_bgr=crop,
        current_pleura_mask=merged_final_mask,
        top2_guided_mask=top2_guided_result["top2_guided_mask"],
    )

    return {
        "crop": crop,
        "binary_top2": binary_top2,
        "top2_final_mask": top2_final_contour_result["final_top2_mask"],
    }


def process_image(index: int, current: int, total: int) -> None:
    print(f"[{current}/{total}] Imagine {index} - natural bridge random jagged limited")

    input_result = build_unification_input(index)

    if input_result is None:
        print(f"Imaginea {index} nu a fost gasita.")
        return

    unification_result = build_top2_unification_debug(
        crop_bgr=input_result["crop"],
        top2_final_mask=input_result["top2_final_mask"],
        support_mask=input_result["binary_top2"],
    )

    contour_with_polyline = unification_result["images"][
        "special_contour_with_middle_polyline_on_crop"
    ]

    save_path = UNIFICATION_SPECIAL_DIR / make_output_name(
        index, "01_natural_bridge_random_jagged_limited"
    )

    save_image(
        save_path,
        contour_with_polyline,
    )


def main() -> None:
    reset_dir(config.RESULTS_DIR)
    ensure_dir(UNIFICATION_SPECIAL_DIR)

    indices = get_indices_to_process()

    if len(indices) == 0:
        print("Nu s-au gasit imagini de procesat.")
        return

    total = len(indices)

    for current, index in enumerate(indices, start=1):
        process_image(index, current, total)

    print("")
    print("Unification special output terminat.")
    print(f"Rezultate salvate in: {UNIFICATION_SPECIAL_DIR}")


if __name__ == "__main__":
    main()
