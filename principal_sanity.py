import cv2
import numpy as np


PRINCIPAL_SANITY_ENABLE = True

SUSPECT_MAX_MEDIAN_Y_FRAC = 0.31
SUSPECT_MAX_BOTTOM_Y_FRAC = 0.38
SUSPECT_MIN_AREA = 250
SUSPECT_MIN_WIDTH_PX = 45

LOWER_SEARCH_MIN_EXTRA_Y = 22
LOWER_SEARCH_MIN_Y_FRAC = 0.27
LOWER_SEARCH_MAX_Y_FRAC = 0.64
LOWER_CANDIDATE_MIN_WIDTH_PX = 70
LOWER_CANDIDATE_MIN_AREA = 220
LOWER_CANDIDATE_MAX_HEIGHT_FRAC = 0.24
LOWER_CANDIDATE_MIN_MEDIAN_DELTA_Y = 34

LINK_KERNEL_W = 35
LINK_KERNEL_H = 5
LINK_DILATE_W = 13
LINK_DILATE_H = 7

TOP_BAND_KEEP_ABOVE_PX = 2
TOP_BAND_KEEP_BELOW_PX = 15
TOP_BAND_MIN_COLUMN_PIXELS = 1
TOP_BAND_CLOSE_W = 7
TOP_BAND_CLOSE_H = 3
TOP_BAND_MIN_COMPONENT_AREA = 18


def normalize_mask(mask):
    result = np.zeros_like(mask, dtype=np.uint8)
    result[mask > 0] = 255
    return result


def empty_mask_like(mask):
    return np.zeros_like(mask, dtype=np.uint8)


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


def mask_median_y(mask):
    if mask is None:
        return None

    ys, _ = np.where(mask > 0)

    if len(ys) == 0:
        return None

    return float(np.median(ys))


def remove_small_components(mask, min_area):
    mask = normalize_mask(mask)
    result = np.zeros_like(mask, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area >= min_area:
            result[labels == label] = 255

    return result


def merge_masks(mask_a, mask_b):
    result = np.zeros_like(mask_a, dtype=np.uint8)

    if mask_a is not None:
        result[mask_a > 0] = 255

    if mask_b is not None:
        result[mask_b > 0] = 255

    return result


def is_upper_principal_suspicious(principal_mask):
    bounds = get_mask_bounds(principal_mask)
    median_y = mask_median_y(principal_mask)

    if bounds is None or median_y is None:
        return False

    height = principal_mask.shape[0]

    if bounds["area"] < SUSPECT_MIN_AREA:
        return False

    if bounds["width"] < SUSPECT_MIN_WIDTH_PX:
        return False

    if median_y > SUSPECT_MAX_MEDIAN_Y_FRAC * height:
        return False

    if bounds["max_y"] > SUSPECT_MAX_BOTTOM_Y_FRAC * height:
        return False

    return True


def build_lower_search_roi(binary_top2, principal_mask):
    height, width = binary_top2.shape[:2]
    principal_bounds = get_mask_bounds(principal_mask)

    roi = np.zeros((height, width), dtype=np.uint8)

    if principal_bounds is None:
        return roi

    start_y = max(
        int(principal_bounds["max_y"] + LOWER_SEARCH_MIN_EXTRA_Y),
        int(round(LOWER_SEARCH_MIN_Y_FRAC * height)),
    )
    end_y = int(round(LOWER_SEARCH_MAX_Y_FRAC * height))

    start_y = max(0, min(height - 1, start_y))
    end_y = max(start_y + 1, min(height, end_y))

    roi[start_y:end_y, :] = 255

    return roi


def score_lower_candidate(component_mask, original_candidate_mask, principal_mask):
    bounds = get_mask_bounds(component_mask)
    original_bounds = get_mask_bounds(original_candidate_mask)

    if bounds is None or original_bounds is None:
        return None

    height, width = principal_mask.shape[:2]
    principal_median_y = mask_median_y(principal_mask)
    candidate_median_y = mask_median_y(original_candidate_mask)

    if principal_median_y is None or candidate_median_y is None:
        return None

    original_area = mask_area(original_candidate_mask)

    if bounds["width"] < LOWER_CANDIDATE_MIN_WIDTH_PX:
        return None

    if original_area < LOWER_CANDIDATE_MIN_AREA:
        return None

    if bounds["height"] > LOWER_CANDIDATE_MAX_HEIGHT_FRAC * height:
        return None

    if candidate_median_y < principal_median_y + LOWER_CANDIDATE_MIN_MEDIAN_DELTA_Y:
        return None

    center_x = width / 2.0
    candidate_center_x = (bounds["min_x"] + bounds["max_x"]) / 2.0
    center_penalty = abs(candidate_center_x - center_x) / max(width, 1)

    thinness_bonus = bounds["width"] / max(bounds["height"], 1)
    vertical_penalty = bounds["height"] * 1.25
    y_penalty = abs(candidate_median_y - 0.38 * height) * 0.35

    score = (
        2.7 * bounds["width"]
        + 1.0 * original_area
        + 24.0 * thinness_bonus
        - vertical_penalty
        - y_penalty
        - 180.0 * center_penalty
    )

    return float(score)


def find_lower_principal_candidate(binary_top2, principal_mask):
    binary_top2 = normalize_mask(binary_top2)
    roi_mask = build_lower_search_roi(binary_top2, principal_mask)
    search_mask = cv2.bitwise_and(binary_top2, roi_mask)

    if mask_area(search_mask) == 0:
        return {
            "candidate_mask": empty_mask_like(binary_top2),
            "roi_mask": roi_mask,
            "rejected_mask": empty_mask_like(binary_top2),
            "score": None,
        }

    link_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (LINK_KERNEL_W, LINK_KERNEL_H),
    )
    linked_mask = cv2.morphologyEx(search_mask, cv2.MORPH_CLOSE, link_kernel, iterations=1)

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(linked_mask, 8)

    best_score = None
    best_mask = empty_mask_like(binary_top2)
    rejected_mask = empty_mask_like(binary_top2)

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (LINK_DILATE_W, LINK_DILATE_H),
    )

    for label in range(1, num_labels):
        component_linked = np.zeros_like(binary_top2, dtype=np.uint8)
        component_linked[labels == label] = 255

        component_region = cv2.dilate(component_linked, dilate_kernel, iterations=1)
        component_original = cv2.bitwise_and(search_mask, component_region)

        score = score_lower_candidate(
            component_linked,
            component_original,
            principal_mask,
        )

        if score is None:
            rejected_mask = merge_masks(rejected_mask, component_original)
            continue

        if best_score is None or score > best_score:
            if mask_area(best_mask) > 0:
                rejected_mask = merge_masks(rejected_mask, best_mask)

            best_score = score
            best_mask = component_original
        else:
            rejected_mask = merge_masks(rejected_mask, component_original)

    return {
        "candidate_mask": best_mask,
        "roi_mask": roi_mask,
        "rejected_mask": rejected_mask,
        "score": best_score,
    }


def keep_top_band_by_column(mask, keep_above_px, keep_below_px):
    mask = normalize_mask(mask)
    height, width = mask.shape[:2]
    result = np.zeros_like(mask, dtype=np.uint8)

    for x in range(width):
        ys = np.where(mask[:, x] > 0)[0]

        if len(ys) < TOP_BAND_MIN_COLUMN_PIXELS:
            continue

        top_y = int(np.min(ys))
        top = max(0, top_y - keep_above_px)
        bottom = min(height, top_y + keep_below_px + 1)

        column = mask[top:bottom, x]
        result[top:bottom, x][column > 0] = 255

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (TOP_BAND_CLOSE_W, TOP_BAND_CLOSE_H),
    )
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    result = remove_small_components(result, TOP_BAND_MIN_COMPONENT_AREA)

    return result


def repair_principal_if_upper_artifact(binary_top2, principal_mask):
    principal_mask = normalize_mask(principal_mask)
    binary_top2 = normalize_mask(binary_top2)

    empty = empty_mask_like(principal_mask)

    if not PRINCIPAL_SANITY_ENABLE:
        return {
            "principal_mask": principal_mask,
            "changed": False,
            "roi_mask": empty,
            "candidate_mask": empty,
            "rejected_mask": empty,
            "replacement_mask": empty,
        }

    if not is_upper_principal_suspicious(principal_mask):
        return {
            "principal_mask": principal_mask,
            "changed": False,
            "roi_mask": empty,
            "candidate_mask": empty,
            "rejected_mask": empty,
            "replacement_mask": empty,
        }

    candidate_result = find_lower_principal_candidate(binary_top2, principal_mask)

    candidate_mask = candidate_result["candidate_mask"]
    roi_mask = candidate_result["roi_mask"]
    rejected_mask = candidate_result["rejected_mask"]

    if mask_area(candidate_mask) == 0:
        return {
            "principal_mask": principal_mask,
            "changed": False,
            "roi_mask": roi_mask,
            "candidate_mask": candidate_mask,
            "rejected_mask": rejected_mask,
            "replacement_mask": empty,
        }

    replacement_mask = keep_top_band_by_column(
        candidate_mask,
        keep_above_px=TOP_BAND_KEEP_ABOVE_PX,
        keep_below_px=TOP_BAND_KEEP_BELOW_PX,
    )

    replacement_bounds = get_mask_bounds(replacement_mask)
    principal_median_y = mask_median_y(principal_mask)
    replacement_median_y = mask_median_y(replacement_mask)

    valid_replacement = (
        replacement_bounds is not None
        and mask_area(replacement_mask) >= 60
        and replacement_bounds["width"] >= LOWER_CANDIDATE_MIN_WIDTH_PX
        and replacement_median_y is not None
        and principal_median_y is not None
        and replacement_median_y >= principal_median_y + LOWER_CANDIDATE_MIN_MEDIAN_DELTA_Y
    )

    if not valid_replacement:
        rejected_mask = merge_masks(rejected_mask, candidate_mask)

        return {
            "principal_mask": principal_mask,
            "changed": False,
            "roi_mask": roi_mask,
            "candidate_mask": candidate_mask,
            "rejected_mask": rejected_mask,
            "replacement_mask": replacement_mask,
        }

    return {
        "principal_mask": replacement_mask,
        "changed": True,
        "roi_mask": roi_mask,
        "candidate_mask": candidate_mask,
        "rejected_mask": rejected_mask,
        "replacement_mask": replacement_mask,
    }


def to_bgr(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image.copy()


def draw_mask_overlay(image_bgr, mask, color, alpha=0.65):
    output = to_bgr(image_bgr)
    mask = normalize_mask(mask)

    if mask_area(mask) == 0:
        return output

    overlay = output.copy()
    overlay[mask > 0] = color

    return cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0)


def draw_principal_sanity_debug(
    image_bgr,
    original_principal_mask,
    repaired_principal_mask,
    roi_mask,
    candidate_mask,
    rejected_mask,
):
    output = to_bgr(image_bgr)

    output = draw_mask_overlay(output, roi_mask, (180, 0, 180), alpha=0.20)
    output = draw_mask_overlay(output, rejected_mask, (0, 0, 255), alpha=0.60)
    output = draw_mask_overlay(output, candidate_mask, (0, 255, 255), alpha=0.60)
    output = draw_mask_overlay(output, original_principal_mask, (255, 255, 0), alpha=0.55)
    output = draw_mask_overlay(output, repaired_principal_mask, (0, 255, 0), alpha=0.70)

    return output
