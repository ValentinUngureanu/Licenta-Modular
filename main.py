# MAIN TEST ONLY IMAGE 41 - SECONDARY RESCUE UPPER GUARD 1
# Ruleaza exclusiv poza 41.
# Testeaza secondary_rescue1.py.
# Pastreaza fixurile anterioare: secondary_component pentru 18 si gap_rescue left_guard_only pentru 37/39/40.
import shutil
import re
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
)
from horizontal_rescue import (
    draw_horizontal_merged_debug,
    draw_horizontal_rescue_debug,
    horizontal_rescue_before_secondary,
)
from gap_rescue import (
    draw_gap_merged_debug,
    draw_gap_rescue_debug,
    gap_rescue_after_secondary,
)
from image_io import (
    ensure_dir,
    find_image_path,
    get_indices_to_process,
    make_output_name,
    prepare_results_dir,
    read_image_bgr,
    save_image,
)
from preprocessing import preprocess_crop
from principal_component import (
    build_principal_component,
    draw_candidate_mask,
    draw_principal_component,
    draw_principal_roi,
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
)
from secondary_rescue import (
    draw_merged_after_rescue as draw_secondary_rescue_merged_debug,
    draw_rescue_debug as draw_secondary_rescue_debug,
    rescue_after_secondary,
)
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
MERGED_BEFORE_SECONDARY_RESCUE_DIR = FINAL_TEST_DIR / "12_MERGED_BEFORE_SECONDARY_RESCUE"
SECONDARY_RESCUE_DIR = FINAL_TEST_DIR / "13_SECONDARY_RESCUE_AFTER_SECONDARY"
MERGED_FINAL_DIR = FINAL_TEST_DIR / "14_MERGED_FINAL_AFTER_ALL_RESCUES"
SMALL_COMPONENT_CLEAN_DIR = FINAL_TEST_DIR / "14B_SMALL_COMPONENT_CLEAN"
SOURCE_ORIGIN_DEBUG_DIR = FINAL_TEST_DIR / "14C_SOURCE_ORIGIN_DEBUG"
FINAL_MASK_CROP_DIR = FINAL_TEST_DIR / "15_FINAL_MASK_ON_CROP"
FINAL_CONTOUR_CROP_DIR = FINAL_TEST_DIR / "16_FINAL_CONTOUR_ON_CROP"
FINAL_MASK_ORIGINAL_DIR = FINAL_TEST_DIR / "17_FINAL_MASK_ON_ORIGINAL"
FINAL_CONTOUR_ORIGINAL_DIR = FINAL_TEST_DIR / "18_FINAL_CONTOUR_ON_ORIGINAL"
FINAL_BINARY_ORIGINAL_DIR = FINAL_TEST_DIR / "19_FINAL_BINARY_MASK_ORIGINAL"
REST_CONTACT_SHEET_PATH = config.RESULTS_DIR / "00_TOATE_POZELE_FINAL_CONTOUR_LEFT_GUARD_ONLY.jpg"

FOCUS_DEBUG_INDICES = {41}
FOCUS_DEBUG_DIR = config.RESULTS_DIR / "99_DEBUG_POZA_41_SECONDARY_RESCUE1"


SMALL_COMPONENT_CLEAN_ENABLE = True
SMALL_COMPONENT_MIN_AREA = 28
SMALL_COMPONENT_MIN_WIDTH = 7
SMALL_COMPONENT_MIN_HEIGHT = 4
SMALL_COMPONENT_REMOVE_COMPACT_AREA = 35
SMALL_COMPONENT_REMOVE_COMPACT_MAX_WIDTH = 14
SMALL_COMPONENT_REMOVE_COMPACT_MAX_HEIGHT = 14


# ============================================================
# AUTOMATIC RESCUE GUARDS
# ============================================================
# Ideea:
#   - horizontal_rescue se ruleaza, dar se accepta doar daca extensia
#     este laterala, subtire si rezonabila fata de principal.
#   - secondary_rescue se ruleaza, dar nu este lasat sa stearga agresiv
#     secondary-ul bun, decat daca secondary-ul este clar overgrown.
# Nu se decide dupa indexul imaginii.
# ============================================================

HORIZONTAL_RESCUE_MAX_AREA_FACTOR = 0.85
HORIZONTAL_RESCUE_MAX_WIDTH_FACTOR = 1.75
HORIZONTAL_RESCUE_MAX_WIDTH_PX = 240
HORIZONTAL_RESCUE_MAX_HEIGHT_FRAC = 0.18
HORIZONTAL_RESCUE_MIN_EXTENSION_PX = 8
HORIZONTAL_RESCUE_MAX_BOTH_SIDE_EXTENSION_PX = 50

SECONDARY_RESCUE_OVERGROWTH_AREA_RATIO = 3.0
SECONDARY_RESCUE_OVERGROWTH_MIN_AREA = 9000
SECONDARY_RESCUE_OVERGROWTH_MIN_DENSITY = 0.18
SECONDARY_RESCUE_OVERGROWTH_MIN_HEIGHT = 125
SECONDARY_RESCUE_MIN_KEEP_FRAC_WHEN_NOT_OVERGROWN = 0.65
SECONDARY_RESCUE_MIN_WIDTH_KEEP_FRAC_WHEN_NOT_OVERGROWN = 0.88

PRINCIPAL_UPPER_ARTIFACT_GUARD_ENABLE = True
PRINCIPAL_UPPER_ARTIFACT_MAX_MEDIAN_Y_FRAC = 0.26
PRINCIPAL_UPPER_ARTIFACT_MIN_HEIGHT_PX = 70
PRINCIPAL_UPPER_ARTIFACT_MIN_AREA = 3000
PRINCIPAL_UPPER_ARTIFACT_KEEP_ABOVE_BOTTOM_PX = 13
PRINCIPAL_UPPER_ARTIFACT_KEEP_BELOW_BOTTOM_PX = 2


def mask_area(mask) -> int:
    if mask is None:
        return 0

    return int(np.count_nonzero(mask > 0))


def get_mask_bounds(mask):
    if mask is None:
        return None

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return {
        "min_x": int(np.min(xs)),
        "max_x": int(np.max(xs)),
        "min_y": int(np.min(ys)),
        "max_y": int(np.max(ys)),
        "width": int(np.max(xs) - np.min(xs) + 1),
        "height": int(np.max(ys) - np.min(ys) + 1),
        "area": int(len(xs)),
    }


def mask_density(mask):
    bounds = get_mask_bounds(mask)

    if bounds is None:
        return 0.0

    box_area = max(bounds["width"] * bounds["height"], 1)
    return float(bounds["area"] / box_area)



def mask_median_y(mask):
    if mask is None:
        return None

    ys, _ = np.where(mask > 0)

    if len(ys) == 0:
        return None

    return float(np.median(ys))


def keep_bottom_band_by_column(mask, keep_above_px, keep_below_px):
    result = np.zeros_like(mask, dtype=np.uint8)
    height, width = mask.shape[:2]

    for x in range(width):
        ys = np.where(mask[:, x] > 0)[0]

        if len(ys) == 0:
            continue

        bottom_y = int(np.max(ys))
        top = max(0, bottom_y - keep_above_px)
        bottom = min(height, bottom_y + keep_below_px + 1)

        column = mask[top:bottom, x]
        result[top:bottom, x][column > 0] = 255

    return result


def apply_principal_upper_artifact_guard(principal_mask):
    if not PRINCIPAL_UPPER_ARTIFACT_GUARD_ENABLE:
        return principal_mask

    bounds = get_mask_bounds(principal_mask)
    median_y = mask_median_y(principal_mask)

    if bounds is None or median_y is None:
        return principal_mask

    image_height = principal_mask.shape[0]

    is_upper_thick_component = (
        median_y < PRINCIPAL_UPPER_ARTIFACT_MAX_MEDIAN_Y_FRAC * image_height
        and bounds["height"] >= PRINCIPAL_UPPER_ARTIFACT_MIN_HEIGHT_PX
        and bounds["area"] >= PRINCIPAL_UPPER_ARTIFACT_MIN_AREA
    )

    if not is_upper_thick_component:
        return principal_mask

    slim_mask = keep_bottom_band_by_column(
        principal_mask,
        keep_above_px=PRINCIPAL_UPPER_ARTIFACT_KEEP_ABOVE_BOTTOM_PX,
        keep_below_px=PRINCIPAL_UPPER_ARTIFACT_KEEP_BELOW_BOTTOM_PX,
    )

    # Protectie: daca subtierea elimina aproape tot, revenim la masca initiala.
    if mask_area(slim_mask) < 0.18 * mask_area(principal_mask):
        return principal_mask

    return slim_mask


def empty_mask_like(mask):
    return np.zeros_like(mask, dtype=np.uint8)


def merge_masks(mask_a, mask_b):
    result = np.zeros_like(mask_a, dtype=np.uint8)

    if mask_a is not None:
        result[mask_a > 0] = 255

    if mask_b is not None:
        result[mask_b > 0] = 255

    return result


def should_accept_horizontal_rescue(horizontal_result, principal_mask) -> bool:
    rescue_mask = horizontal_result["rescue_mask"]
    rescue_area = mask_area(rescue_mask)

    if rescue_area == 0:
        return False

    principal_area = mask_area(principal_mask)
    principal_bounds = get_mask_bounds(principal_mask)
    rescue_bounds = get_mask_bounds(rescue_mask)

    if principal_area == 0 or principal_bounds is None or rescue_bounds is None:
        return False

    image_height, _ = principal_mask.shape[:2]

    left_gain = max(0, principal_bounds["min_x"] - rescue_bounds["min_x"])
    right_gain = max(0, rescue_bounds["max_x"] - principal_bounds["max_x"])
    max_gain = max(left_gain, right_gain)

    # Daca nu extinde real nici in stanga nici in dreapta, nu e rescue util.
    if max_gain < HORIZONTAL_RESCUE_MIN_EXTENSION_PX:
        return False

    # Cazurile rele 30/31 se intind in ambele directii fata de principal.
    # Rescue-ul orizontal bun ar trebui sa continue predominant o singura parte.
    if (
        left_gain > HORIZONTAL_RESCUE_MAX_BOTH_SIDE_EXTENSION_PX
        and right_gain > HORIZONTAL_RESCUE_MAX_BOTH_SIDE_EXTENSION_PX
    ):
        return False

    max_allowed_area = max(
        350,
        int(round(HORIZONTAL_RESCUE_MAX_AREA_FACTOR * principal_area)),
    )

    if rescue_area > max_allowed_area:
        return False

    max_allowed_width = max(
        HORIZONTAL_RESCUE_MAX_WIDTH_PX,
        int(round(HORIZONTAL_RESCUE_MAX_WIDTH_FACTOR * principal_bounds["width"])),
    )

    if rescue_bounds["width"] > max_allowed_width:
        return False

    max_allowed_height = max(
        36,
        int(round(HORIZONTAL_RESCUE_MAX_HEIGHT_FRAC * image_height)),
    )

    if rescue_bounds["height"] > max_allowed_height:
        return False

    return True


def run_guarded_horizontal_rescue(binary_top2, principal_mask, traveler_points):
    horizontal_result = horizontal_rescue_before_secondary(
        binary_top2,
        principal_mask,
        traveler_points=traveler_points,
    )

    if should_accept_horizontal_rescue(horizontal_result, principal_mask):
        return horizontal_result

    empty = empty_mask_like(principal_mask)
    rejected_mask = merge_masks(
        horizontal_result["rejected_mask"],
        horizontal_result["accepted_mask"],
    )

    return {
        "rescue_mask": empty,
        "merged_mask": principal_mask.copy(),
        "binary_top2_guarded": binary_top2.copy(),
        "roi_mask": horizontal_result["roi_mask"],
        "candidate_mask": horizontal_result["candidate_mask"],
        "accepted_mask": empty,
        "rejected_mask": rejected_mask,
    }


def should_accept_secondary_rescue(
    principal_mask,
    secondary_mask_before_rescue,
    merged_before_rescue,
    secondary_rescue_result,
) -> bool:
    principal_area = mask_area(principal_mask)
    secondary_before_area = mask_area(secondary_mask_before_rescue)
    secondary_after_area = mask_area(secondary_rescue_result["secondary_mask"])
    removed_area = mask_area(secondary_rescue_result["removed_secondary_mask"])
    rescue_area = mask_area(secondary_rescue_result["rescue_mask"])

    if principal_area == 0:
        return False

    # Daca secondary normal nu a gasit nimic, acceptam rescue lateral doar daca
    # nu adauga o masca exagerata fata de principal.
    if secondary_before_area == 0:
        max_rescue_area = max(600, int(round(1.20 * principal_area)))
        return rescue_area > 0 and rescue_area <= max_rescue_area

    # Daca secondary-ul este clar overgrown, acceptam guard-ul din secondary_rescue.
    # Nu folosim doar raportul secondary/principal, pentru ca la imagini ca 29
    # principalul initial poate fi foarte mic, iar extensia buna ar parea fals
    # "overgrown" doar matematic. Cerem si masa absoluta/densitate/inaltime.
    secondary_before_bounds = get_mask_bounds(secondary_mask_before_rescue)
    secondary_before_density = mask_density(secondary_mask_before_rescue)

    overgrown_secondary = (
        secondary_before_area >= SECONDARY_RESCUE_OVERGROWTH_MIN_AREA
        and secondary_before_area > SECONDARY_RESCUE_OVERGROWTH_AREA_RATIO * principal_area
        and secondary_before_bounds is not None
        and (
            secondary_before_density >= SECONDARY_RESCUE_OVERGROWTH_MIN_DENSITY
            or secondary_before_bounds["height"] >= SECONDARY_RESCUE_OVERGROWTH_MIN_HEIGHT
        )
    )

    if overgrown_secondary:
        return True

    # Daca secondary_rescue sterge mult din secondary, dar secondary-ul nu era
    # overgrown, respingem rescue-ul. Asta protejeaza cazuri ca 30/31.
    if removed_area > 0:
        min_allowed_after_area = int(
            round(
                SECONDARY_RESCUE_MIN_KEEP_FRAC_WHEN_NOT_OVERGROWN
                * secondary_before_area
            )
        )

        if secondary_after_area < min_allowed_after_area:
            return False

        before_bounds = get_mask_bounds(merged_before_rescue)
        after_bounds = get_mask_bounds(secondary_rescue_result["merged_mask"])

        if before_bounds is not None and after_bounds is not None:
            min_allowed_width = int(
                round(
                    SECONDARY_RESCUE_MIN_WIDTH_KEEP_FRAC_WHEN_NOT_OVERGROWN
                    * before_bounds["width"]
                )
            )

            if after_bounds["width"] < min_allowed_width:
                return False

    return True


def run_guarded_secondary_rescue(
    binary_top2_guarded,
    principal_after_horizontal_mask,
    secondary_mask_before_rescue,
    merged_before_secondary_rescue,
    traveler_points,
):
    secondary_rescue_result = rescue_after_secondary(
        binary_top2=binary_top2_guarded,
        principal_mask=principal_after_horizontal_mask,
        secondary_mask=secondary_mask_before_rescue,
        merged_mask=merged_before_secondary_rescue,
        traveler_points=traveler_points,
    )

    if should_accept_secondary_rescue(
        principal_after_horizontal_mask,
        secondary_mask_before_rescue,
        merged_before_secondary_rescue,
        secondary_rescue_result,
    ):
        return secondary_rescue_result

    empty = empty_mask_like(principal_after_horizontal_mask)
    rejected_mask = merge_masks(
        secondary_rescue_result["rejected_mask"],
        secondary_rescue_result["accepted_mask"],
    )
    rejected_mask = merge_masks(
        rejected_mask,
        secondary_rescue_result["removed_secondary_mask"],
    )

    return {
        "secondary_mask": secondary_mask_before_rescue.copy(),
        "rescue_mask": empty,
        "merged_mask": merged_before_secondary_rescue.copy(),
        "removed_secondary_mask": empty,
        "roi_mask": secondary_rescue_result["roi_mask"],
        "candidate_mask": secondary_rescue_result["candidate_mask"],
        "rejected_mask": rejected_mask,
        "accepted_mask": empty,
        "changed": False,
        "guard_triggered": False,
        "added_components": [],
    }


def remove_very_small_components(mask):
    if not SMALL_COMPONENT_CLEAN_ENABLE:
        return mask, empty_mask_like(mask)

    if mask is None or mask_area(mask) == 0:
        return mask, empty_mask_like(mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )

    cleaned = np.zeros_like(mask, dtype=np.uint8)
    removed = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        component_pixels = labels == label

        is_tiny = (
            area < SMALL_COMPONENT_MIN_AREA
            or width < SMALL_COMPONENT_MIN_WIDTH
            or height < SMALL_COMPONENT_MIN_HEIGHT
        )

        is_compact_speckle = (
            area < SMALL_COMPONENT_REMOVE_COMPACT_AREA
            and width <= SMALL_COMPONENT_REMOVE_COMPACT_MAX_WIDTH
            and height <= SMALL_COMPONENT_REMOVE_COMPACT_MAX_HEIGHT
        )

        if is_tiny or is_compact_speckle:
            removed[component_pixels] = 255
        else:
            cleaned[component_pixels] = 255

    return cleaned, removed


def draw_artifact_source_overlay(
    crop,
    principal_mask,
    horizontal_rescue_mask,
    secondary_mask_normal,
    gap_rescue_mask,
    secondary_rescue_mask,
    final_mask=None,
    traveler_points=None,
):
    output = crop.copy()
    overlay = output.copy()

    # Ordinea conteaza: sursele tarzii se vad peste cele vechi.
    # Albastru   = principal / ancora
    # Cyan       = horizontal rescue
    # Verde      = secondary normal
    # Portocaliu = gap rescue
    # Magenta    = secondary rescue
    if principal_mask is not None:
        overlay[principal_mask > 0] = (255, 0, 0)

    if horizontal_rescue_mask is not None:
        overlay[horizontal_rescue_mask > 0] = (255, 255, 0)

    if secondary_mask_normal is not None:
        overlay[secondary_mask_normal > 0] = (0, 255, 0)

    if gap_rescue_mask is not None:
        overlay[gap_rescue_mask > 0] = (0, 165, 255)

    if secondary_rescue_mask is not None:
        overlay[secondary_rescue_mask > 0] = (255, 0, 255)

    output = cv2.addWeighted(overlay, 0.50, output, 0.50, 0)

    if final_mask is not None:
        final_u8 = np.zeros_like(final_mask, dtype=np.uint8)
        final_u8[final_mask > 0] = 255
        contours, _ = cv2.findContours(final_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 0, 255), 1)

    if traveler_points is not None:
        for point in traveler_points:
            px = None
            py = None

            if hasattr(point, "x") and hasattr(point, "y"):
                px = float(point.x)
                py = float(point.y)
            else:
                try:
                    values = np.asarray(point).reshape(-1)

                    if len(values) >= 2:
                        px = float(values[0])
                        py = float(values[1])
                except Exception:
                    px = None
                    py = None

            if px is None or py is None:
                continue

            x = int(round(px))
            y = int(round(py))

            if 0 <= y < output.shape[0] and 0 <= x < output.shape[1]:
                cv2.circle(output, (x, y), 1, (255, 255, 255), -1)

    cv2.putText(output, "blue=principal cyan=horizontal green=secondary orange=gap magenta=secondary_rescue", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return output


def save_focus_debug(
    index,
    crop,
    palette_7,
    binary_top1,
    binary_top2,
    traveler_debug,
    principal_roi_debug,
    principal_candidate_debug,
    principal_debug,
    horizontal_rescue_debug,
    horizontal_merged_debug,
    binary_top2_guarded,
    secondary_roi_debug,
    secondary_candidates_debug,
    secondary_debug,
    merged_before_secondary_rescue_debug,
    secondary_rescue_debug,
    merged_final_debug,
    small_component_clean_debug,
    source_origin_debug,
    final_mask_crop_debug,
    final_contour_crop_debug,
    final_mask_original_debug,
    final_contour_original_debug,
    final_binary_original,
    principal_mask,
    horizontal_rescue_mask,
    principal_after_horizontal_mask,
    secondary_mask_before_rescue,
    secondary_mask_after_rescue,
    secondary_rescue_mask,
    merged_before_secondary_rescue_mask,
    merged_final_mask,
    small_component_removed_mask,
    horizontal_roi_mask,
    horizontal_candidate_mask,
    horizontal_rejected_mask,
    horizontal_accepted_mask,
    secondary_roi_mask,
    secondary_candidate_mask,
    secondary_rejected_mask,
    secondary_accepted_mask,
    secondary_rescue_roi_mask,
    secondary_rescue_candidate_mask,
    secondary_rescue_rejected_mask,
    secondary_rescue_accepted_mask,
    secondary_rescue_removed_mask,
):
    if index not in FOCUS_DEBUG_INDICES:
        return

    image_dir = FOCUS_DEBUG_DIR / f"{index:02d}"
    raw_dir = image_dir / "RAW_MASKS"

    ensure_dir(image_dir)
    ensure_dir(raw_dir)

    save_image(image_dir / "00_crop.png", crop)
    save_image(image_dir / "01_palette_7.png", palette_7)
    save_image(image_dir / "02_binary_top1.png", binary_top1)
    save_image(image_dir / "03_binary_top2_secondary_search.png", binary_top2)
    save_image(image_dir / "04_extended_traveler.png", traveler_debug)
    save_image(image_dir / "05_principal_roi.png", principal_roi_debug)
    save_image(image_dir / "06_principal_candidate_mask_in_roi.png", principal_candidate_debug)
    save_image(image_dir / "07_principal_component.png", principal_debug)
    save_image(image_dir / "08_horizontal_rescue_debug.png", horizontal_rescue_debug)
    save_image(image_dir / "09_principal_plus_horizontal_rescue.png", horizontal_merged_debug)
    save_image(image_dir / "10_binary_top2_guarded_for_secondary.png", binary_top2_guarded)
    save_image(image_dir / "11_secondary_search_roi_top2.png", secondary_roi_debug)
    save_image(image_dir / "12_secondary_top2_candidates_in_roi.png", secondary_candidates_debug)
    save_image(image_dir / "13_principal_horizontal_secondary_components.png", secondary_debug)
    save_image(image_dir / "14_merged_before_secondary_rescue.png", merged_before_secondary_rescue_debug)
    save_image(image_dir / "15_secondary_rescue_after_secondary.png", secondary_rescue_debug)
    save_image(image_dir / "16_merged_final_after_all_rescues.png", merged_final_debug)
    save_image(image_dir / "16b_small_component_clean.png", small_component_clean_debug)
    save_image(image_dir / "16c_source_origin_debug.png", source_origin_debug)
    save_image(image_dir / "17_final_mask_on_crop.png", final_mask_crop_debug)
    save_image(image_dir / "18_final_contour_on_crop.png", final_contour_crop_debug)
    save_image(image_dir / "19_final_mask_on_original.png", final_mask_original_debug)
    save_image(image_dir / "20_final_contour_on_original.png", final_contour_original_debug)
    save_image(image_dir / "21_final_binary_mask_original.png", final_binary_original)

    save_image(raw_dir / "01_principal_mask.png", principal_mask)
    save_image(raw_dir / "02_horizontal_rescue_mask.png", horizontal_rescue_mask)
    save_image(raw_dir / "03_principal_after_horizontal_mask.png", principal_after_horizontal_mask)
    save_image(raw_dir / "04_secondary_mask_before_rescue.png", secondary_mask_before_rescue)
    save_image(raw_dir / "05_secondary_mask_after_rescue.png", secondary_mask_after_rescue)
    save_image(raw_dir / "06_secondary_rescue_mask.png", secondary_rescue_mask)
    save_image(raw_dir / "07_merged_before_secondary_rescue_mask.png", merged_before_secondary_rescue_mask)
    save_image(raw_dir / "08_merged_final_mask.png", merged_final_mask)
    save_image(raw_dir / "08b_small_component_removed_mask.png", small_component_removed_mask)
    save_image(raw_dir / "09_horizontal_roi_mask.png", horizontal_roi_mask)
    save_image(raw_dir / "10_horizontal_candidate_mask.png", horizontal_candidate_mask)
    save_image(raw_dir / "11_horizontal_rejected_mask.png", horizontal_rejected_mask)
    save_image(raw_dir / "12_horizontal_accepted_mask.png", horizontal_accepted_mask)
    save_image(raw_dir / "13_binary_top2_guarded.png", binary_top2_guarded)
    save_image(raw_dir / "14_secondary_roi_mask.png", secondary_roi_mask)
    save_image(raw_dir / "15_secondary_candidate_mask.png", secondary_candidate_mask)
    save_image(raw_dir / "16_secondary_rejected_mask.png", secondary_rejected_mask)
    save_image(raw_dir / "17_secondary_accepted_mask.png", secondary_accepted_mask)
    save_image(raw_dir / "18_secondary_rescue_roi_mask.png", secondary_rescue_roi_mask)
    save_image(raw_dir / "19_secondary_rescue_candidate_mask.png", secondary_rescue_candidate_mask)
    save_image(raw_dir / "20_secondary_rescue_rejected_mask.png", secondary_rescue_rejected_mask)
    save_image(raw_dir / "21_secondary_rescue_accepted_mask.png", secondary_rescue_accepted_mask)
    save_image(raw_dir / "22_secondary_rescue_removed_mask.png", secondary_rescue_removed_mask)


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
    principal_roi_mask = principal_result["roi_mask"]
    principal_candidate_mask = principal_result["candidate_mask"]
    rejected_principal_mask = principal_result["rejected_mask"]

    principal_selector_result = select_principal_by_lower_candidate(
        binary_top2,
        principal_mask_initial,
    )
    principal_mask = principal_selector_result["principal_mask"]

    horizontal_result = run_guarded_horizontal_rescue(
        binary_top2,
        principal_mask,
        traveler_points=extended_points,
    )

    horizontal_rescue_mask = horizontal_result["rescue_mask"]
    principal_after_horizontal_mask = horizontal_result["merged_mask"]
    binary_top2_guarded = horizontal_result["binary_top2_guarded"]
    horizontal_roi_mask = horizontal_result["roi_mask"]
    horizontal_candidate_mask = horizontal_result["candidate_mask"]
    horizontal_accepted_mask = horizontal_result["accepted_mask"]
    horizontal_rejected_mask = horizontal_result["rejected_mask"]

    secondary_result = build_secondary_components(
        binary_top1,
        binary_top2_guarded,
        principal_after_horizontal_mask,
        extended_points,
    )

    secondary_mask_normal = secondary_result["secondary_mask"]
    merged_before_gap_rescue = secondary_result["merged_mask"]
    secondary_roi_mask = secondary_result["roi_mask"]
    secondary_candidate_mask = secondary_result["candidate_mask"]
    secondary_rejected_mask = secondary_result["rejected_mask"]
    secondary_accepted_mask = secondary_result["accepted_mask"]

    gap_rescue_result = gap_rescue_after_secondary(
        binary_top2=binary_top2_guarded,
        principal_mask=principal_after_horizontal_mask,
        secondary_mask=secondary_mask_normal,
        merged_mask=merged_before_gap_rescue,
        traveler_points=extended_points,
    )

    gap_rescue_mask = gap_rescue_result["rescue_mask"]
    secondary_mask_before_rescue = gap_rescue_result["secondary_mask"]
    merged_before_secondary_rescue = gap_rescue_result["merged_mask"]
    gap_rescue_roi_mask = gap_rescue_result["roi_mask"]
    gap_rescue_candidate_mask = gap_rescue_result["candidate_mask"]
    gap_rescue_rejected_mask = gap_rescue_result["rejected_mask"]
    gap_rescue_accepted_mask = gap_rescue_result["accepted_mask"]

    secondary_rescue_result = run_guarded_secondary_rescue(
        binary_top2_guarded=binary_top2_guarded,
        principal_after_horizontal_mask=principal_after_horizontal_mask,
        secondary_mask_before_rescue=secondary_mask_before_rescue,
        merged_before_secondary_rescue=merged_before_secondary_rescue,
        traveler_points=extended_points,
    )

    secondary_mask_after_rescue = secondary_rescue_result["secondary_mask"]
    secondary_rescue_mask = secondary_rescue_result["rescue_mask"]
    merged_final_mask_before_small_clean = secondary_rescue_result["merged_mask"]
    merged_final_mask, small_component_removed_mask = remove_very_small_components(
        merged_final_mask_before_small_clean,
    )
    secondary_rescue_removed_mask = secondary_rescue_result["removed_secondary_mask"]
    secondary_rescue_roi_mask = secondary_rescue_result["roi_mask"]
    secondary_rescue_candidate_mask = secondary_rescue_result["candidate_mask"]
    secondary_rescue_rejected_mask = secondary_rescue_result["rejected_mask"]
    secondary_rescue_accepted_mask = secondary_rescue_result["accepted_mask"]

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

    principal_roi_debug = draw_principal_roi(
        crop,
        principal_roi_mask,
        traveler_points=extended_points,
    )

    principal_candidate_debug = draw_candidate_mask(
        crop,
        principal_candidate_mask,
        traveler_points=extended_points,
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

    horizontal_merged_debug = draw_horizontal_merged_debug(
        crop,
        principal_after_horizontal_mask,
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

    save_image(CROP_DIR / make_output_name(index, "crop"), crop)
    save_image(PALETTE_7_DIR / make_output_name(index, "palette_7"), palette_7)
    save_image(BINARY_TOP1_DIR / make_output_name(index, "binary_top1"), binary_top1)
    save_image(BINARY_TOP2_DIR / make_output_name(index, "binary_top2_secondary_search"), binary_top2)
    save_image(TRAVELER_DIR / make_output_name(index, "extended_traveler"), traveler_debug)
    save_image(PRINCIPAL_DIR / make_output_name(index, "principal_component"), principal_debug)
    save_image(PRINCIPAL_SELECTOR_DIR / make_output_name(index, "principal_selector"), principal_selector_debug)
    save_image(HORIZONTAL_RESCUE_DIR / make_output_name(index, "horizontal_rescue_before_secondary"), horizontal_rescue_debug)
    save_image(BINARY_TOP2_GUARDED_DIR / make_output_name(index, "binary_top2_guarded_for_secondary"), binary_top2_guarded)
    save_image(SECONDARY_ROI_DIR / make_output_name(index, "secondary_search_roi_top2"), secondary_roi_debug)
    save_image(SECONDARY_CANDIDATES_DIR / make_output_name(index, "secondary_top2_candidates_in_roi"), secondary_candidates_debug)
    save_image(SECONDARY_DIR / make_output_name(index, "principal_horizontal_secondary_components"), secondary_debug)
    save_image(GAP_RESCUE_DIR / make_output_name(index, "gap_rescue_after_secondary"), gap_rescue_debug)
    save_image(MERGED_BEFORE_SECONDARY_RESCUE_DIR / make_output_name(index, "merged_before_secondary_rescue"), merged_before_secondary_rescue_debug)
    save_image(SECONDARY_RESCUE_DIR / make_output_name(index, "secondary_rescue_after_secondary"), secondary_rescue_debug)
    save_image(MERGED_FINAL_DIR / make_output_name(index, "merged_final_after_all_rescues"), merged_final_debug)
    save_image(SMALL_COMPONENT_CLEAN_DIR / make_output_name(index, "small_component_clean"), small_component_clean_debug)
    save_image(SOURCE_ORIGIN_DEBUG_DIR / make_output_name(index, "source_origin_debug"), source_origin_debug)
    save_image(FINAL_MASK_CROP_DIR / make_output_name(index, "final_mask_on_crop"), final_mask_crop_debug)
    save_image(FINAL_CONTOUR_CROP_DIR / make_output_name(index, "final_contour_on_crop"), final_contour_crop_debug)
    save_image(FINAL_MASK_ORIGINAL_DIR / make_output_name(index, "final_mask_on_original"), final_mask_original_debug)
    save_image(FINAL_CONTOUR_ORIGINAL_DIR / make_output_name(index, "final_contour_on_original"), final_contour_original_debug)
    save_image(FINAL_BINARY_ORIGINAL_DIR / make_output_name(index, "final_binary_mask_original"), final_binary_original)

    save_focus_debug(
        index=index,
        crop=crop,
        palette_7=palette_7,
        binary_top1=binary_top1,
        binary_top2=binary_top2,
        traveler_debug=traveler_debug,
        principal_roi_debug=principal_roi_debug,
        principal_candidate_debug=principal_candidate_debug,
        principal_debug=principal_debug,
        horizontal_rescue_debug=horizontal_rescue_debug,
        horizontal_merged_debug=horizontal_merged_debug,
        binary_top2_guarded=binary_top2_guarded,
        secondary_roi_debug=secondary_roi_debug,
        secondary_candidates_debug=secondary_candidates_debug,
        secondary_debug=secondary_debug,
        merged_before_secondary_rescue_debug=merged_before_secondary_rescue_debug,
        secondary_rescue_debug=secondary_rescue_debug,
        merged_final_debug=merged_final_debug,
        small_component_clean_debug=small_component_clean_debug,
        source_origin_debug=source_origin_debug,
        final_mask_crop_debug=final_mask_crop_debug,
        final_contour_crop_debug=final_contour_crop_debug,
        final_mask_original_debug=final_mask_original_debug,
        final_contour_original_debug=final_contour_original_debug,
        final_binary_original=final_binary_original,
        principal_mask=principal_mask,
        horizontal_rescue_mask=horizontal_rescue_mask,
        principal_after_horizontal_mask=principal_after_horizontal_mask,
        secondary_mask_before_rescue=secondary_mask_before_rescue,
        secondary_mask_after_rescue=secondary_mask_after_rescue,
        secondary_rescue_mask=secondary_rescue_mask,
        merged_before_secondary_rescue_mask=merged_before_secondary_rescue,
        merged_final_mask=merged_final_mask,
        small_component_removed_mask=small_component_removed_mask,
        horizontal_roi_mask=horizontal_roi_mask,
        horizontal_candidate_mask=horizontal_candidate_mask,
        horizontal_rejected_mask=horizontal_rejected_mask,
        horizontal_accepted_mask=horizontal_accepted_mask,
        secondary_roi_mask=secondary_roi_mask,
        secondary_candidate_mask=secondary_candidate_mask,
        secondary_rejected_mask=secondary_rejected_mask,
        secondary_accepted_mask=secondary_accepted_mask,
        secondary_rescue_roi_mask=secondary_rescue_roi_mask,
        secondary_rescue_candidate_mask=secondary_rescue_candidate_mask,
        secondary_rescue_rejected_mask=secondary_rescue_rejected_mask,
        secondary_rescue_accepted_mask=secondary_rescue_accepted_mask,
        secondary_rescue_removed_mask=secondary_rescue_removed_mask,
    )



def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)
    return path

def make_rest_final_contours_contact_sheet(exclude_indices=None):
    if exclude_indices is None:
        exclude_indices = set()

    image_paths = []

    for path in sorted(FINAL_CONTOUR_ORIGINAL_DIR.glob("*_final_contour_on_original.png")):
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

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        tile = np.zeros((tile_height + label_height, tile_width, 3), dtype=np.uint8)

        y0 = label_height + (tile_height - new_height) // 2
        x0 = (tile_width - new_width) // 2
        tile[y0:y0 + new_height, x0:x0 + new_width] = resized

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

        sheet[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]] = tile

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
    ensure_dir(FOCUS_DEBUG_DIR)

    indices = get_indices_to_process()

    if len(indices) == 0:
        print("Nu s-au gasit imagini de procesat.")
        return

    total = len(indices)

    for current, index in enumerate(indices, start=1):
        process_image(index, current, total)



if __name__ == "__main__":
    main()
