# MAIN ALL IMAGES - FIX POZA 55 RIGHT ISOLATED HORIZONTAL 1
# Ruleaza toate imaginile din ORIGINAL_IMAGES.
# Salveaza rezultatele in RESULTS/06_FINAL_CONTOUR_TEST.
# Pastreaza varianta curata V16 si adauga un guard local pentru componenta orizontala izolata din dreapta la poza 55.
# Debug detaliat separat pentru poza 55.

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
REST_CONTACT_SHEET_PATH = config.RESULTS_DIR / "00_TOATE_POZELE_FINAL_CONTOUR_55_RIGHT_ISOLATED_HORIZONTAL1.jpg"

FOCUS_DEBUG_INDICES = {55}
FOCUS_DEBUG_DIR = config.RESULTS_DIR / "99_DEBUG_POZA_55_RIGHT_ISOLATED_HORIZONTAL1"


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


RIGHT_ISOLATED_SECONDARY_RESCUE_GUARD_ENABLE = True
RIGHT_ISOLATED_SECONDARY_RESCUE_MIN_GAP_PX = 60
RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_AREA = 650
RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_WIDTH = 130
RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_HEIGHT = 80
RIGHT_ISOLATED_SECONDARY_RESCUE_MIN_BELOW_MEDIAN_PX = 45


def filter_right_isolated_secondary_rescue(rescue_mask, base_mask):
    if not RIGHT_ISOLATED_SECONDARY_RESCUE_GUARD_ENABLE:
        return rescue_mask, empty_mask_like(rescue_mask)

    if rescue_mask is None or base_mask is None:
        return rescue_mask, empty_mask_like(rescue_mask)

    if mask_area(rescue_mask) == 0 or mask_area(base_mask) == 0:
        return rescue_mask, empty_mask_like(rescue_mask)

    base_bounds = get_mask_bounds(base_mask)
    base_median_y = mask_median_y(base_mask)

    if base_bounds is None or base_median_y is None:
        return rescue_mask, empty_mask_like(rescue_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (rescue_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    kept = np.zeros_like(rescue_mask, dtype=np.uint8)
    removed = np.zeros_like(rescue_mask, dtype=np.uint8)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        x2 = x + width - 1
        component_median_y = float(centroids[label][1])
        component_pixels = labels == label

        right_gap = x - base_bounds["max_x"] - 1

        is_right_isolated = right_gap >= RIGHT_ISOLATED_SECONDARY_RESCUE_MIN_GAP_PX

        is_small_enough = (
            area <= RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_AREA
            and width <= RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_WIDTH
            and height <= RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_HEIGHT
        )

        # y mai mare = mai jos in imagine.
        # La poza 46 artefactul este o insula in dreapta jos, departe de masca deja acceptata.
        is_much_lower = (
            component_median_y >= base_median_y + RIGHT_ISOLATED_SECONDARY_RESCUE_MIN_BELOW_MEDIAN_PX
        )

        if is_right_isolated and is_small_enough and is_much_lower:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    return kept, removed


GAP_FLOATING_RIGHT_GUARD_ENABLE = True
GAP_FLOATING_RIGHT_MIN_AREA = 250
GAP_FLOATING_RIGHT_MAX_AREA = 1200
GAP_FLOATING_RIGHT_MIN_WIDTH = 35
GAP_FLOATING_RIGHT_MAX_WIDTH = 140
GAP_FLOATING_RIGHT_MAX_HEIGHT = 55
GAP_FLOATING_RIGHT_MIN_RIGHT_GAP_FROM_PRINCIPAL = 18
GAP_FLOATING_RIGHT_LEFT_SUPPORT_WINDOW = 90
GAP_FLOATING_RIGHT_LEFT_SUPPORT_Y_BAND = 35
GAP_FLOATING_RIGHT_MIN_LEFT_SUPPORT_PIXELS = 8


GAP_UPPER_RIGHT_GUARD_ENABLE = True
GAP_UPPER_RIGHT_MIN_AREA = 120
GAP_UPPER_RIGHT_MAX_AREA = 900
GAP_UPPER_RIGHT_MIN_WIDTH = 12
GAP_UPPER_RIGHT_MAX_WIDTH = 120
GAP_UPPER_RIGHT_MAX_HEIGHT = 60
GAP_UPPER_RIGHT_MIN_RIGHT_GAP_FROM_PRINCIPAL = 8
GAP_UPPER_RIGHT_CONTEXT_WINDOW = 120
GAP_UPPER_RIGHT_MIN_CONTEXT_PIXELS = 20
GAP_UPPER_RIGHT_MIN_ABOVE_LOCAL_PX = 35


def filter_upper_right_gap_rescue(
    gap_rescue_mask,
    principal_after_horizontal_mask,
    secondary_mask_normal,
):
    if not GAP_UPPER_RIGHT_GUARD_ENABLE:
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    if gap_rescue_mask is None or principal_after_horizontal_mask is None:
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    if mask_area(gap_rescue_mask) == 0 or mask_area(principal_after_horizontal_mask) == 0:
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    principal_bounds = get_mask_bounds(principal_after_horizontal_mask)

    if principal_bounds is None:
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    support_mask = np.zeros_like(gap_rescue_mask, dtype=np.uint8)
    support_mask[principal_after_horizontal_mask > 0] = 255

    if secondary_mask_normal is not None:
        support_mask[secondary_mask_normal > 0] = 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (gap_rescue_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    kept = np.zeros_like(gap_rescue_mask, dtype=np.uint8)
    removed = np.zeros_like(gap_rescue_mask, dtype=np.uint8)

    image_h, image_w = gap_rescue_mask.shape[:2]

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component_pixels = labels == label
        component_median_y = float(centroids[label][1])

        right_gap_from_principal = x - principal_bounds["max_x"] - 1

        size_matches = (
            area >= GAP_UPPER_RIGHT_MIN_AREA
            and area <= GAP_UPPER_RIGHT_MAX_AREA
            and width >= GAP_UPPER_RIGHT_MIN_WIDTH
            and width <= GAP_UPPER_RIGHT_MAX_WIDTH
            and height <= GAP_UPPER_RIGHT_MAX_HEIGHT
        )

        is_right_of_principal = (
            right_gap_from_principal >= GAP_UPPER_RIGHT_MIN_RIGHT_GAP_FROM_PRINCIPAL
        )

        x1_context = max(0, x - GAP_UPPER_RIGHT_CONTEXT_WINDOW)
        x2_context = min(image_w, x + 1)

        context_region = support_mask[:, x1_context:x2_context]
        context_ys, context_xs = np.where(context_region > 0)

        has_context = len(context_ys) >= GAP_UPPER_RIGHT_MIN_CONTEXT_PIXELS

        if has_context:
            local_reference_y = float(np.median(context_ys))
        else:
            local_reference_y = None

        # In imagini, y mai mic inseamna mai sus.
        # Artefactul de la 44 este in dreapta, dar sare mult deasupra directiei locale.
        is_much_above_local_direction = (
            local_reference_y is not None
            and component_median_y <= local_reference_y - GAP_UPPER_RIGHT_MIN_ABOVE_LOCAL_PX
        )

        if size_matches and is_right_of_principal and has_context and is_much_above_local_direction:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    return kept, removed


def filter_floating_right_gap_rescue(
    gap_rescue_mask,
    principal_after_horizontal_mask,
    secondary_mask_normal,
):
    if not GAP_FLOATING_RIGHT_GUARD_ENABLE:
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    if gap_rescue_mask is None or principal_after_horizontal_mask is None:
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    if mask_area(gap_rescue_mask) == 0 or mask_area(principal_after_horizontal_mask) == 0:
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    principal_bounds = get_mask_bounds(principal_after_horizontal_mask)

    if principal_bounds is None:
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    support_mask = np.zeros_like(gap_rescue_mask, dtype=np.uint8)
    support_mask[principal_after_horizontal_mask > 0] = 255

    if secondary_mask_normal is not None:
        support_mask[secondary_mask_normal > 0] = 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (gap_rescue_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    kept = np.zeros_like(gap_rescue_mask, dtype=np.uint8)
    removed = np.zeros_like(gap_rescue_mask, dtype=np.uint8)

    height_img, width_img = gap_rescue_mask.shape[:2]

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        x2 = x + width - 1
        y2 = y + height - 1
        component_median_y = float(centroids[label][1])
        component_pixels = labels == label

        right_gap_from_principal = x - principal_bounds["max_x"] - 1

        size_matches = (
            area >= GAP_FLOATING_RIGHT_MIN_AREA
            and area <= GAP_FLOATING_RIGHT_MAX_AREA
            and width >= GAP_FLOATING_RIGHT_MIN_WIDTH
            and width <= GAP_FLOATING_RIGHT_MAX_WIDTH
            and height <= GAP_FLOATING_RIGHT_MAX_HEIGHT
        )

        is_right_of_principal = (
            right_gap_from_principal >= GAP_FLOATING_RIGHT_MIN_RIGHT_GAP_FROM_PRINCIPAL
        )

        y1_support = max(0, int(round(component_median_y)) - GAP_FLOATING_RIGHT_LEFT_SUPPORT_Y_BAND)
        y2_support = min(height_img, int(round(component_median_y)) + GAP_FLOATING_RIGHT_LEFT_SUPPORT_Y_BAND + 1)
        x1_support = max(0, x - GAP_FLOATING_RIGHT_LEFT_SUPPORT_WINDOW)
        x2_support = max(0, x)

        left_support_pixels = 0

        if x2_support > x1_support and y2_support > y1_support:
            left_support_pixels = int(
                np.count_nonzero(support_mask[y1_support:y2_support, x1_support:x2_support] > 0)
            )

        lacks_local_left_support = (
            left_support_pixels < GAP_FLOATING_RIGHT_MIN_LEFT_SUPPORT_PIXELS
        )

        # Pentru poza 46 bucata incercuita este introdusa de gap_rescue:
        # este in dreapta componentei principale, are dimensiune medie,
        # dar nu are niciun suport local in stanga pe aceeasi banda y.
        if size_matches and is_right_of_principal and lacks_local_left_support:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    return kept, removed


HORIZONTAL_LAYERED_TAIL_GUARD_ENABLE = True
HORIZONTAL_LAYERED_TAIL_MIN_COMPONENTS = 3
HORIZONTAL_LAYERED_TAIL_MIN_UPPER_AREA = 180
HORIZONTAL_LAYERED_TAIL_MAX_UPPER_HEIGHT = 24
HORIZONTAL_LAYERED_TAIL_MAX_LOWER_AREA = 900
HORIZONTAL_LAYERED_TAIL_MAX_LOWER_HEIGHT = 34
HORIZONTAL_LAYERED_TAIL_MIN_Y_GAP = 16
HORIZONTAL_LAYERED_TAIL_MIN_OVERLAP_FRAC = 0.35
HORIZONTAL_LAYERED_TAIL_MAX_RIGHT_GAP_FROM_UPPER = 95
HORIZONTAL_LAYERED_TAIL_MIN_RIGHT_REGION_GAIN = -80


def x_overlap_fraction(a, b):
    left = max(a["x"], b["x"])
    right = min(a["x2"], b["x2"])

    if right < left:
        return 0.0

    overlap = right - left + 1
    min_width = max(min(a["width"], b["width"]), 1)

    return float(overlap / min_width)


def filter_layered_horizontal_tail(horizontal_rescue_mask, principal_mask):
    if not HORIZONTAL_LAYERED_TAIL_GUARD_ENABLE:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    if horizontal_rescue_mask is None or principal_mask is None:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    if mask_area(horizontal_rescue_mask) == 0 or mask_area(principal_mask) == 0:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    principal_bounds = get_mask_bounds(principal_mask)

    if principal_bounds is None:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (horizontal_rescue_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if num_labels - 1 < HORIZONTAL_LAYERED_TAIL_MIN_COMPONENTS:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    components = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        x2 = x + width - 1
        y2 = y + height - 1

        components.append({
            "label": label,
            "area": area,
            "x": x,
            "y": y,
            "x2": x2,
            "y2": y2,
            "width": width,
            "height": height,
            "centroid_y": float(centroids[label][1]),
        })

    # Cautam o componenta superioara valida din horizontal_rescue, aflata la capatul
    # din dreapta al principalului. Aceasta este "continuarea buna".
    upper_candidates = []

    for component in components:
        is_upper_shape = (
            component["area"] >= HORIZONTAL_LAYERED_TAIL_MIN_UPPER_AREA
            and component["height"] <= HORIZONTAL_LAYERED_TAIL_MAX_UPPER_HEIGHT
        )

        is_near_right_end_of_principal = (
            component["x2"] >= principal_bounds["max_x"] + HORIZONTAL_LAYERED_TAIL_MIN_RIGHT_REGION_GAIN
        )

        if is_upper_shape and is_near_right_end_of_principal:
            upper_candidates.append(component)

    if len(upper_candidates) == 0:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    # Alegem componenta cea mai de sus dintre candidatele valide.
    upper = min(upper_candidates, key=lambda item: item["centroid_y"])

    kept = np.zeros_like(horizontal_rescue_mask, dtype=np.uint8)
    removed = np.zeros_like(horizontal_rescue_mask, dtype=np.uint8)

    for component in components:
        pixels = labels == component["label"]

        if component["label"] == upper["label"]:
            kept[pixels] = 255
            continue

        is_lower_than_upper = (
            component["centroid_y"] >= upper["centroid_y"] + HORIZONTAL_LAYERED_TAIL_MIN_Y_GAP
        )

        is_small_or_medium_tail = (
            component["area"] <= HORIZONTAL_LAYERED_TAIL_MAX_LOWER_AREA
            and component["height"] <= HORIZONTAL_LAYERED_TAIL_MAX_LOWER_HEIGHT
        )

        overlap_frac = x_overlap_fraction(component, upper)

        overlaps_upper = overlap_frac >= HORIZONTAL_LAYERED_TAIL_MIN_OVERLAP_FRAC

        right_gap_from_upper = component["x"] - upper["x2"] - 1

        is_right_tail_after_upper = (
            right_gap_from_upper >= 0
            and right_gap_from_upper <= HORIZONTAL_LAYERED_TAIL_MAX_RIGHT_GAP_FROM_UPPER
        )

        # Regula este locala pe horizontal_rescue:
        # daca exista o componenta superioara valida, stergem doar componentele
        # joase care sunt pe aceeasi extensie sau imediat dupa ea.
        # Nu atingem principal_mask si nu curatam final_mask global.
        if (
            is_lower_than_upper
            and is_small_or_medium_tail
            and (overlaps_upper or is_right_tail_after_upper)
        ):
            removed[pixels] = 255
        else:
            kept[pixels] = 255

    # Protectie: daca filtrul a sters tot sau aproape tot horizontal_rescue, revenim.
    if mask_area(kept) < 0.20 * mask_area(horizontal_rescue_mask):
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    return kept, removed


SECONDARY_AFTER_HORIZONTAL_TAIL_GUARD_ENABLE = True
SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_HORIZONTAL_AREA = 150
SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_AREA = 220
SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_WIDTH = 45
SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_HEIGHT = 28
SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_RIGHT_GAP = -2
SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_BELOW_HORIZONTAL_PX = 32



RIGHT_ISOLATED_HORIZONTAL_COMPONENT_GUARD_ENABLE = True
RIGHT_ISOLATED_HORIZONTAL_MIN_RIGHT_GAIN = 75
RIGHT_ISOLATED_HORIZONTAL_MIN_GAP_FROM_MAIN = 30
RIGHT_ISOLATED_HORIZONTAL_MAX_AREA = 220
RIGHT_ISOLATED_HORIZONTAL_MAX_WIDTH = 35
RIGHT_ISOLATED_HORIZONTAL_MAX_HEIGHT = 18
RIGHT_ISOLATED_HORIZONTAL_MIN_MAIN_AREA = 350
RIGHT_ISOLATED_HORIZONTAL_MIN_COMPONENT_Y = 200


def filter_right_isolated_horizontal_component(horizontal_rescue_mask, principal_mask):
    if not RIGHT_ISOLATED_HORIZONTAL_COMPONENT_GUARD_ENABLE:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    if horizontal_rescue_mask is None or principal_mask is None:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    if mask_area(horizontal_rescue_mask) == 0 or mask_area(principal_mask) == 0:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    principal_bounds = get_mask_bounds(principal_mask)

    if principal_bounds is None:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (horizontal_rescue_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 2:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    components = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        x2 = x + width - 1
        y2 = y + height - 1

        components.append({
            "label": label,
            "area": area,
            "x": x,
            "x2": x2,
            "y": y,
            "y2": y2,
            "width": width,
            "height": height,
        })

    main_component = max(components, key=lambda item: item["area"])

    if main_component["area"] < RIGHT_ISOLATED_HORIZONTAL_MIN_MAIN_AREA:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    kept = np.zeros_like(horizontal_rescue_mask, dtype=np.uint8)
    removed = np.zeros_like(horizontal_rescue_mask, dtype=np.uint8)

    for component in components:
        label = component["label"]
        component_pixels = labels == label

        if label == main_component["label"]:
            kept[component_pixels] = 255
            continue

        right_gain = component["x2"] - principal_bounds["max_x"]
        gap_from_main = component["x"] - main_component["x2"]

        is_small_right_isolated = (
            component["area"] <= RIGHT_ISOLATED_HORIZONTAL_MAX_AREA
            and component["width"] <= RIGHT_ISOLATED_HORIZONTAL_MAX_WIDTH
            and component["height"] <= RIGHT_ISOLATED_HORIZONTAL_MAX_HEIGHT
            and right_gain >= RIGHT_ISOLATED_HORIZONTAL_MIN_RIGHT_GAIN
            and gap_from_main >= RIGHT_ISOLATED_HORIZONTAL_MIN_GAP_FROM_MAIN
            and component["y"] >= RIGHT_ISOLATED_HORIZONTAL_MIN_COMPONENT_Y
        )

        # Caz 55:
        # componenta #4 este x=795..816, y=237..247, area=140,
        # separata de componenta principala horizontal_rescue.
        # O eliminam ca artefact izolat de dreapta, fara sa atingem componenta
        # principala sau extensiile lipite de ea.
        if is_small_right_isolated:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    return kept, removed



def filter_secondary_tail_after_horizontal(
    secondary_mask,
    horizontal_rescue_mask,
):
    if not SECONDARY_AFTER_HORIZONTAL_TAIL_GUARD_ENABLE:
        return secondary_mask, empty_mask_like(secondary_mask)

    if secondary_mask is None or horizontal_rescue_mask is None:
        return secondary_mask, empty_mask_like(secondary_mask)

    if mask_area(secondary_mask) == 0:
        return secondary_mask, empty_mask_like(secondary_mask)

    horizontal_bounds = get_mask_bounds(horizontal_rescue_mask)
    horizontal_median_y = mask_median_y(horizontal_rescue_mask)

    if horizontal_bounds is None or horizontal_median_y is None:
        return secondary_mask, empty_mask_like(secondary_mask)

    if horizontal_bounds["area"] < SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_HORIZONTAL_AREA:
        return secondary_mask, empty_mask_like(secondary_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (secondary_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    kept = np.zeros_like(secondary_mask, dtype=np.uint8)
    removed = np.zeros_like(secondary_mask, dtype=np.uint8)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        x2 = x + width - 1
        y2 = y + height - 1

        component_pixels = labels == label
        component_median_y = float(centroids[label][1])

        is_small_tail = (
            area <= SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_AREA
            and width <= SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_WIDTH
            and height <= SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_HEIGHT
        )

        starts_after_horizontal_end = (
            x >= horizontal_bounds["max_x"] + SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_RIGHT_GAP
        )

        # y mai mare inseamna mai jos in imagine.
        is_far_below_horizontal = (
            component_median_y >= horizontal_median_y + SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_BELOW_HORIZONTAL_PX
            or y2 >= horizontal_median_y + SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_BELOW_HORIZONTAL_PX + 8
        )

        vertical_overlap_top = max(y, horizontal_bounds["min_y"])
        vertical_overlap_bottom = min(y2, horizontal_bounds["max_y"])
        vertical_overlap = max(0, vertical_overlap_bottom - vertical_overlap_top + 1)
        vertical_overlap_frac = vertical_overlap / max(height, 1)

        is_thin_edge_fragment = (
            area <= 90
            and width <= 28
            and height <= 10
        )

        is_only_touching_horizontal_edge = (
            vertical_overlap_frac <= 0.30
            or y2 <= horizontal_bounds["min_y"] + 1
        )

        is_after_or_at_horizontal_end = (
            x >= horizontal_bounds["max_x"] - 1
        )

        # Prima regula: fragmente secondary mici, mult sub horizontal.
        remove_low_tail = (
            is_small_tail
            and starts_after_horizontal_end
            and is_far_below_horizontal
        )

        # A doua regula: fragment subtire care apare imediat dupa capatul horizontalului,
        # atinge doar marginea benzii si nu continua grosimea reala a pleurei.
        # Asta prinde bucata marcata acum la 54, fara sa curatam finalul global
        # si fara sa atingem principal_mask.
        remove_thin_edge_tail = (
            is_thin_edge_fragment
            and is_after_or_at_horizontal_end
            and is_only_touching_horizontal_edge
        )

        if remove_low_tail or remove_thin_edge_tail:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    return kept, removed


PRINCIPAL_UNDER_HORIZONTAL_TAIL_GUARD_ENABLE = True
PRINCIPAL_UNDER_HORIZONTAL_MAX_AREA = 520
PRINCIPAL_UNDER_HORIZONTAL_MAX_WIDTH = 70
PRINCIPAL_UNDER_HORIZONTAL_MAX_HEIGHT = 45
PRINCIPAL_UNDER_HORIZONTAL_MIN_HORIZONTAL_AREA = 180
PRINCIPAL_UNDER_HORIZONTAL_MIN_BELOW_HORIZONTAL_PX = 16
PRINCIPAL_UNDER_HORIZONTAL_MAX_LEFT_GAP_TO_HORIZONTAL = 28
PRINCIPAL_UNDER_HORIZONTAL_MIN_X_OVERLAP_FRAC = 0.10


def interval_overlap_fraction(a_min, a_max, b_min, b_max, width_a):
    left = max(a_min, b_min)
    right = min(a_max, b_max)

    if right < left:
        return 0.0

    return float((right - left + 1) / max(width_a, 1))


def filter_principal_tail_under_horizontal(principal_mask, horizontal_rescue_mask):
    if not PRINCIPAL_UNDER_HORIZONTAL_TAIL_GUARD_ENABLE:
        return principal_mask, empty_mask_like(principal_mask)

    if principal_mask is None or horizontal_rescue_mask is None:
        return principal_mask, empty_mask_like(principal_mask)

    if mask_area(principal_mask) == 0 or mask_area(horizontal_rescue_mask) == 0:
        return principal_mask, empty_mask_like(principal_mask)

    horizontal_bounds = get_mask_bounds(horizontal_rescue_mask)
    horizontal_median_y = mask_median_y(horizontal_rescue_mask)

    if horizontal_bounds is None or horizontal_median_y is None:
        return principal_mask, empty_mask_like(principal_mask)

    if horizontal_bounds["area"] < PRINCIPAL_UNDER_HORIZONTAL_MIN_HORIZONTAL_AREA:
        return principal_mask, empty_mask_like(principal_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (principal_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 2:
        return principal_mask, empty_mask_like(principal_mask)

    kept = np.zeros_like(principal_mask, dtype=np.uint8)
    removed = np.zeros_like(principal_mask, dtype=np.uint8)

    # Protejam componenta principala mare.
    areas = [
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, num_labels)
    ]
    largest_label = int(np.argmax(areas)) + 1

    for label in range(1, num_labels):
        component_pixels = labels == label

        if label == largest_label:
            kept[component_pixels] = 255
            continue

        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        x2 = x + width - 1
        y2 = y + height - 1

        component_median_y = float(centroids[label][1])

        is_small_tail_piece = (
            area <= PRINCIPAL_UNDER_HORIZONTAL_MAX_AREA
            and width <= PRINCIPAL_UNDER_HORIZONTAL_MAX_WIDTH
            and height <= PRINCIPAL_UNDER_HORIZONTAL_MAX_HEIGHT
        )

        overlap_frac = interval_overlap_fraction(
            x,
            x2,
            horizontal_bounds["min_x"],
            horizontal_bounds["max_x"],
            width,
        )

        left_gap_to_horizontal = horizontal_bounds["min_x"] - x2

        touches_horizontal_zone = (
            overlap_frac >= PRINCIPAL_UNDER_HORIZONTAL_MIN_X_OVERLAP_FRAC
            or (
                0 <= left_gap_to_horizontal <= PRINCIPAL_UNDER_HORIZONTAL_MAX_LEFT_GAP_TO_HORIZONTAL
            )
        )

        # y mai mare inseamna mai jos.
        is_below_accepted_horizontal = (
            component_median_y >= horizontal_median_y + PRINCIPAL_UNDER_HORIZONTAL_MIN_BELOW_HORIZONTAL_PX
            or y2 >= horizontal_median_y + PRINCIPAL_UNDER_HORIZONTAL_MIN_BELOW_HORIZONTAL_PX + 8
        )

        # Regula este stricta:
        # sterge doar fragmente mici din principal care se afla sub horizontal_rescue
        # acceptat, in aceeasi zona x. Nu se aplica daca nu exista horizontal_rescue valid.
        if is_small_tail_piece and touches_horizontal_zone and is_below_accepted_horizontal:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    # Protectie: daca ar sterge prea mult din principal, revenim.
    if mask_area(removed) > 0.18 * mask_area(principal_mask):
        return principal_mask, empty_mask_like(principal_mask)

    return kept, removed


HORIZONTAL_FLOATING_UPPER_STRIP_GUARD_ENABLE = True
HORIZONTAL_FLOATING_UPPER_STRIP_MAX_AREA = 650
HORIZONTAL_FLOATING_UPPER_STRIP_MIN_WIDTH = 40
HORIZONTAL_FLOATING_UPPER_STRIP_MAX_WIDTH = 130
HORIZONTAL_FLOATING_UPPER_STRIP_MAX_HEIGHT = 22
HORIZONTAL_FLOATING_UPPER_STRIP_MIN_RIGHT_GAIN = 35
HORIZONTAL_FLOATING_UPPER_STRIP_MAX_RIGHT_GAIN = 90
HORIZONTAL_FLOATING_UPPER_STRIP_MAX_START_GAP_FROM_PRINCIPAL = 35
HORIZONTAL_FLOATING_UPPER_STRIP_CONTEXT_LEFT = 130
HORIZONTAL_FLOATING_UPPER_STRIP_CONTEXT_RIGHT = 15
HORIZONTAL_FLOATING_UPPER_STRIP_MIN_CONTEXT_PIXELS = 35
HORIZONTAL_FLOATING_UPPER_STRIP_MIN_ABOVE_LOCAL_PX = 6
HORIZONTAL_FLOATING_UPPER_STRIP_MAX_DIRECT_CONTACT_PIXELS = 6


def filter_floating_upper_horizontal_strip(horizontal_rescue_mask, principal_mask):
    if not HORIZONTAL_FLOATING_UPPER_STRIP_GUARD_ENABLE:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    if horizontal_rescue_mask is None or principal_mask is None:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    if mask_area(horizontal_rescue_mask) == 0 or mask_area(principal_mask) == 0:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    principal_bounds = get_mask_bounds(principal_mask)

    if principal_bounds is None:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (horizontal_rescue_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    kept = np.zeros_like(horizontal_rescue_mask, dtype=np.uint8)
    removed = np.zeros_like(horizontal_rescue_mask, dtype=np.uint8)

    image_h, image_w = horizontal_rescue_mask.shape[:2]

    for label in range(1, num_labels):
        component_pixels = labels == label

        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        x2 = x + width - 1
        y2 = y + height - 1

        component_median_y = float(centroids[label][1])

        right_gain = x2 - principal_bounds["max_x"]

        size_matches = (
            area <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_AREA
            and width >= HORIZONTAL_FLOATING_UPPER_STRIP_MIN_WIDTH
            and width <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_WIDTH
            and height <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_HEIGHT
        )

        starts_near_principal_edge = (
            x <= principal_bounds["max_x"] + HORIZONTAL_FLOATING_UPPER_STRIP_MAX_START_GAP_FROM_PRINCIPAL
        )

        extends_right = (
            right_gain >= HORIZONTAL_FLOATING_UPPER_STRIP_MIN_RIGHT_GAIN
            and right_gain <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_RIGHT_GAIN
            and starts_near_principal_edge
        )

        x1_context = max(0, x - HORIZONTAL_FLOATING_UPPER_STRIP_CONTEXT_LEFT)
        x2_context = min(image_w, x + HORIZONTAL_FLOATING_UPPER_STRIP_CONTEXT_RIGHT + 1)

        context = principal_mask[:, x1_context:x2_context]
        context_ys, context_xs = np.where(context > 0)

        has_context = len(context_ys) >= HORIZONTAL_FLOATING_UPPER_STRIP_MIN_CONTEXT_PIXELS

        if has_context:
            local_reference_y = float(np.median(context_ys))
        else:
            local_reference_y = None

        # y mai mic inseamna mai sus in imagine.
        is_above_local_principal_band = (
            local_reference_y is not None
            and component_median_y <= local_reference_y - HORIZONTAL_FLOATING_UPPER_STRIP_MIN_ABOVE_LOCAL_PX
        )

        # Verificam daca fasia chiar are contact direct cu principalul.
        # Daca are contact real, nu o stergem.
        dilated_principal = cv2.dilate(
            (principal_mask > 0).astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        direct_contact_pixels = int(
            np.count_nonzero((component_pixels.astype(np.uint8) > 0) & (dilated_principal > 0))
        )

        lacks_direct_contact = (
            direct_contact_pixels <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_DIRECT_CONTACT_PIXELS
        )

        # Regula stricta:
        # elimina doar o fasie horizontal_rescue subtire, in dreapta, care pluteste
        # deasupra benzii principale locale si nu are contact real cu principalul.
        # Nu atinge principal_mask si nu curata final_mask global.
        if size_matches and extends_right and has_context and is_above_local_principal_band and lacks_direct_contact:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    # Protectie: daca ar sterge tot horizontal_rescue, este permis doar cand era
    # o singura fasie mica. Pentru rescue-uri mari/complexe revenim.
    if mask_area(removed) == mask_area(horizontal_rescue_mask):
        if mask_area(horizontal_rescue_mask) > HORIZONTAL_FLOATING_UPPER_STRIP_MAX_AREA:
            return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    return kept, removed


SECONDARY_FLOATING_STRIP_AFTER_HORIZONTAL_REJECT_ENABLE = True

SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_AREA = 260
SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_WIDTH = 60
SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_HEIGHT = 35
SECONDARY_LOW_TAIL_AFTER_STRIP_MIN_RIGHT_GAIN = 55


def filter_secondary_floating_strip_after_horizontal_reject(
    secondary_mask,
    principal_mask,
):
    if not SECONDARY_FLOATING_STRIP_AFTER_HORIZONTAL_REJECT_ENABLE:
        return secondary_mask, empty_mask_like(secondary_mask)

    if secondary_mask is None or principal_mask is None:
        return secondary_mask, empty_mask_like(secondary_mask)

    if mask_area(secondary_mask) == 0 or mask_area(principal_mask) == 0:
        return secondary_mask, empty_mask_like(secondary_mask)

    principal_bounds = get_mask_bounds(principal_mask)
    principal_median_y = mask_median_y(principal_mask)

    if principal_bounds is None or principal_median_y is None:
        return secondary_mask, empty_mask_like(secondary_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (secondary_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    kept = np.zeros_like(secondary_mask, dtype=np.uint8)
    removed = np.zeros_like(secondary_mask, dtype=np.uint8)

    dilated_principal = cv2.dilate(
        (principal_mask > 0).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    for label in range(1, num_labels):
        component_pixels = labels == label

        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        x2 = x + width - 1
        y2 = y + height - 1

        component_median_y = float(centroids[label][1])
        right_gain = x2 - principal_bounds["max_x"]
        starts_near_principal_edge = (
            x <= principal_bounds["max_x"] + HORIZONTAL_FLOATING_UPPER_STRIP_MAX_START_GAP_FROM_PRINCIPAL
        )
        moderate_right_gain = (
            right_gain >= HORIZONTAL_FLOATING_UPPER_STRIP_MIN_RIGHT_GAIN
            and right_gain <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_RIGHT_GAIN
            and starts_near_principal_edge
        )

        contact_pixels = int(
            np.count_nonzero((component_pixels.astype(np.uint8) > 0) & (dilated_principal > 0))
        )

        # Cazul care ramanea la 54:
        # area=528, x=667..765, y=200..209, width=99, height=10.
        # Este o fasie secondary lunga/subtire, reapărută dupa respingerea horizontal_rescue.
        is_long_thin_right_strip = (
            area >= 250
            and area <= 850
            and width >= 45
            and width <= 150
            and height <= 20
            and moderate_right_gain
            and contact_pixels <= 12
        )

        # Cozi mici joase din acelasi fenomen.
        is_low_tail_after_right_extension = (
            area <= SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_AREA
            and width <= SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_WIDTH
            and height <= SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_HEIGHT
            and moderate_right_gain
            and right_gain >= SECONDARY_LOW_TAIL_AFTER_STRIP_MIN_RIGHT_GAIN
            and (
                component_median_y >= principal_median_y + 40
                or y2 >= principal_median_y + 50
            )
        )

        if is_long_thin_right_strip or is_low_tail_after_right_extension:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    return kept, removed


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

    horizontal_rescue_mask, horizontal_layered_tail_removed_mask = filter_layered_horizontal_tail(
        horizontal_rescue_mask,
        principal_mask,
    )

    horizontal_rescue_mask, horizontal_floating_strip_removed_mask = filter_floating_upper_horizontal_strip(
        horizontal_rescue_mask,
        principal_mask,
    )

    horizontal_rescue_mask, horizontal_right_isolated_removed_mask = filter_right_isolated_horizontal_component(
        horizontal_rescue_mask,
        principal_mask,
    )

    horizontal_removed_mask = merge_masks(
        horizontal_layered_tail_removed_mask,
        horizontal_floating_strip_removed_mask,
    )
    horizontal_removed_mask = merge_masks(
        horizontal_removed_mask,
        horizontal_right_isolated_removed_mask,
    )

    principal_after_horizontal_mask = merge_masks(
        principal_mask,
        horizontal_rescue_mask,
    )

    horizontal_accepted_mask = horizontal_rescue_mask
    horizontal_rejected_mask = merge_masks(
        horizontal_rejected_mask,
        horizontal_removed_mask,
    )

    if mask_area(horizontal_removed_mask) > 0:
        binary_top2_guarded = binary_top2_guarded.copy()
        binary_top2_guarded[horizontal_removed_mask > 0] = 0

    principal_mask, principal_under_horizontal_removed_mask = filter_principal_tail_under_horizontal(
        principal_mask,
        horizontal_rescue_mask,
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
    merged_before_gap_rescue = secondary_result["merged_mask"]
    secondary_roi_mask = secondary_result["roi_mask"]
    secondary_candidate_mask = secondary_result["candidate_mask"]
    secondary_rejected_mask = secondary_result["rejected_mask"]
    secondary_accepted_mask = secondary_result["accepted_mask"]

    secondary_mask_normal, secondary_tail_after_horizontal_removed_mask = filter_secondary_tail_after_horizontal(
        secondary_mask_normal,
        horizontal_rescue_mask,
    )

    secondary_mask_normal, secondary_floating_readd_removed_mask = filter_secondary_floating_strip_after_horizontal_reject(
        secondary_mask_normal,
        principal_mask,
    )

    secondary_total_removed_mask = merge_masks(
        secondary_tail_after_horizontal_removed_mask,
        secondary_floating_readd_removed_mask,
    )

    secondary_accepted_mask = secondary_mask_normal
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

    # Important pentru poza 54:
    # dupa ce horizontal_rescue este respins si secondary este filtrat,
    # gap_rescue poate readauga aceeasi fasie lunga/subtire.
    # De aceea aplicam aceeasi regula stricta si pe masca combinata secondary+gap.
    secondary_mask_before_rescue, combined_secondary_gap_removed_mask = filter_secondary_floating_strip_after_horizontal_reject(
        secondary_mask_before_rescue,
        principal_mask,
    )

    gap_rescue_rejected_mask = empty_mask_like(principal_after_horizontal_mask)
    gap_rescue_rejected_mask = merge_masks(
        gap_rescue_rejected_mask,
        combined_secondary_gap_removed_mask,
    )

    if mask_area(combined_secondary_gap_removed_mask) > 0:
        binary_top2_guarded = binary_top2_guarded.copy()
        binary_top2_guarded[combined_secondary_gap_removed_mask > 0] = 0

    merged_before_secondary_rescue = np.zeros_like(principal_after_horizontal_mask, dtype=np.uint8)
    merged_before_secondary_rescue[principal_after_horizontal_mask > 0] = 255
    merged_before_secondary_rescue[secondary_mask_before_rescue > 0] = 255

    gap_rescue_roi_mask = gap_rescue_result["roi_mask"]
    gap_rescue_candidate_mask = gap_rescue_result["candidate_mask"]
    gap_rescue_rejected_mask = gap_rescue_result["rejected_mask"]
    gap_rescue_accepted_mask = gap_rescue_result["accepted_mask"]

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

    secondary_rescue_mask, right_isolated_rescue_removed_mask = filter_right_isolated_secondary_rescue(
        secondary_rescue_mask_raw,
        merged_before_secondary_rescue,
    )

    merged_final_mask_before_small_clean = np.zeros_like(merged_before_secondary_rescue, dtype=np.uint8)
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

    make_rest_final_contours_contact_sheet(exclude_indices=set())


if __name__ == "__main__":
    main()
