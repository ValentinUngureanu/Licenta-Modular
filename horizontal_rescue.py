import cv2
import numpy as np

from gap_rescue import mask_area
from postprocessing import empty_mask_like, get_mask_bounds

PRINCIPAL_COLOR = (0, 255, 0)
RESCUE_COLOR = (255, 0, 255)
ROI_COLOR = (255, 0, 0)
CANDIDATE_COLOR = (255, 255, 0)
REJECTED_COLOR = (0, 0, 255)
TRAVELER_COLOR = (255, 180, 0)
MERGED_COLOR = (0, 255, 0)

HORIZONTAL_ENABLE = True

HORIZONTAL_MIN_WIDTH_FRAC = 0.22
HORIZONTAL_MAX_HEIGHT_FRAC = 0.16
HORIZONTAL_MAX_VERTICALITY = 0.38
HORIZONTAL_MAX_ABS_SLOPE = 0.18
HORIZONTAL_MIN_POINTS = 25

HORIZONTAL_OVERLAP_PX = 45
HORIZONTAL_SEARCH_WIDTH_PX = 260
HORIZONTAL_SEARCH_WIDTH_FRAC = 0.38

HORIZONTAL_UP_HALF_HEIGHT_PX = 30
HORIZONTAL_DOWN_HALF_HEIGHT_PX = 9

HORIZONTAL_MIN_AREA = 7
HORIZONTAL_MIN_WIDTH_PX = 5
HORIZONTAL_MIN_EXTENSION_GAIN_PX = 8
HORIZONTAL_MAX_COMPONENT_VERTICALITY = 1.60
HORIZONTAL_MAX_ACCEPTED_COMPONENTS_PER_SIDE = 4

HORIZONTAL_SIDE_MIN_MISSING_FRAC = 0.04

HORIZONTAL_RESCUE_MAX_AREA_FACTOR = 0.85
HORIZONTAL_RESCUE_MAX_WIDTH_FACTOR = 1.75
HORIZONTAL_RESCUE_MAX_WIDTH_PX = 240
HORIZONTAL_RESCUE_MAX_HEIGHT_FRAC = 0.18
HORIZONTAL_RESCUE_MIN_EXTENSION_PX = 8
HORIZONTAL_RESCUE_MAX_BOTH_SIDE_EXTENSION_PX = 50

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

RIGHT_ISOLATED_HORIZONTAL_COMPONENT_GUARD_ENABLE = True
RIGHT_ISOLATED_HORIZONTAL_MIN_RIGHT_GAIN = 75
RIGHT_ISOLATED_HORIZONTAL_MIN_GAP_FROM_MAIN = 30
RIGHT_ISOLATED_HORIZONTAL_MAX_AREA = 220
RIGHT_ISOLATED_HORIZONTAL_MAX_WIDTH = 35
RIGHT_ISOLATED_HORIZONTAL_MAX_HEIGHT = 18
RIGHT_ISOLATED_HORIZONTAL_MIN_MAIN_AREA = 350
RIGHT_ISOLATED_HORIZONTAL_MIN_COMPONENT_Y = 200

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


def normalize_mask(mask):
    if mask is None:
        return None

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return (mask > 0).astype(np.uint8) * 255


def to_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image.copy()


def draw_mask_overlay(base_image, mask, color, alpha=0.60):
    result = to_bgr(base_image)

    if mask is None:
        return result

    mask = normalize_mask(mask)
    mask_bool = mask > 0

    if np.count_nonzero(mask_bool) == 0:
        return result

    color_array = np.array(color, dtype=np.float32)
    result_float = result.astype(np.float32)

    result_float[mask_bool] = (1.0 - alpha) * result_float[
        mask_bool
    ] + alpha * color_array

    return np.clip(result_float, 0, 255).astype(np.uint8)


def draw_points(image, points, color, radius=1):
    result = to_bgr(image)

    if points is None or len(points) == 0:
        return result

    points = np.asarray(points, dtype=np.int32)

    for x, y in points:
        cv2.circle(
            result,
            (int(x), int(y)),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )

    return result


def merge_masks(mask_a, mask_b):
    mask_a = normalize_mask(mask_a)
    mask_b = normalize_mask(mask_b)

    result = np.zeros_like(mask_a, dtype=np.uint8)
    result[mask_a > 0] = 255
    result[mask_b > 0] = 255

    return result


def mask_bounds(mask):
    mask = normalize_mask(mask)
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
        "median_y": float(np.median(ys)),
    }


def component_stats(component_mask):
    component_mask = normalize_mask(component_mask)
    ys, xs = np.where(component_mask > 0)

    if len(xs) == 0:
        return None

    min_x = int(np.min(xs))
    max_x = int(np.max(xs))
    min_y = int(np.min(ys))
    max_y = int(np.max(ys))

    width = max(1, max_x - min_x + 1)
    height = max(1, max_y - min_y + 1)
    area = int(len(xs))
    verticality = float(height / max(width, 1))

    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "width": width,
        "height": height,
        "area": area,
        "verticality": verticality,
        "median_y": float(np.median(ys)),
    }


def mask_to_bottom_points(mask):
    mask = normalize_mask(mask)
    _, width = mask.shape[:2]

    points = []

    for x in range(width):
        ys = np.flatnonzero(mask[:, x] > 0)

        if len(ys) == 0:
            continue

        points.append((x, int(ys[-1])))

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32)

    return np.array(points, dtype=np.int32)


def get_edge_y(mask, side):
    mask = normalize_mask(mask)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return 0.0

    if side == "right":
        edge_x = int(np.max(xs))
        selected = ys[xs >= edge_x - 18]
    else:
        edge_x = int(np.min(xs))
        selected = ys[xs <= edge_x + 18]

    if len(selected) == 0:
        return float(np.median(ys))

    return float(np.median(selected))


def get_principal_slope(principal_mask):
    points = mask_to_bottom_points(principal_mask)

    if len(points) < 2:
        return 0.0

    xs = points[:, 0].astype(np.float32)
    ys = points[:, 1].astype(np.float32)

    if np.max(xs) <= np.min(xs):
        return 0.0

    try:
        return float(np.polyfit(xs, ys, 1)[0])
    except Exception:
        return 0.0


def is_horizontal_principal(principal_mask):
    principal_mask = normalize_mask(principal_mask)
    height, width = principal_mask.shape[:2]
    bounds = mask_bounds(principal_mask)

    if bounds is None:
        return False

    points = mask_to_bottom_points(principal_mask)

    if len(points) < HORIZONTAL_MIN_POINTS:
        return False

    width_frac = bounds["width"] / max(width, 1)
    height_frac = bounds["height"] / max(height, 1)
    verticality = bounds["height"] / max(bounds["width"], 1)
    slope = get_principal_slope(principal_mask)

    if width_frac < HORIZONTAL_MIN_WIDTH_FRAC:
        return False

    if height_frac > HORIZONTAL_MAX_HEIGHT_FRAC:
        return False

    if verticality > HORIZONTAL_MAX_VERTICALITY:
        return False

    if abs(slope) > HORIZONTAL_MAX_ABS_SLOPE:
        return False

    return True


def get_horizontal_search_x_range(binary_top2, principal_mask, side):
    binary_top2 = normalize_mask(binary_top2)
    principal_mask = normalize_mask(principal_mask)

    _, width = binary_top2.shape[:2]
    bounds = mask_bounds(principal_mask)

    if bounds is None:
        return 0, -1

    search_width = max(
        HORIZONTAL_SEARCH_WIDTH_PX,
        int(round(HORIZONTAL_SEARCH_WIDTH_FRAC * width)),
    )

    if side == "right":
        x1 = max(0, bounds["max_x"] - HORIZONTAL_OVERLAP_PX)
        x2 = min(width - 1, bounds["max_x"] + search_width)
    else:
        x1 = max(0, bounds["min_x"] - search_width)
        x2 = min(width - 1, bounds["min_x"] + HORIZONTAL_OVERLAP_PX)

    return x1, x2


def side_has_space(binary_top2, principal_mask, side):
    binary_top2 = normalize_mask(binary_top2)
    _, width = binary_top2.shape[:2]
    bounds = mask_bounds(principal_mask)

    if bounds is None:
        return False

    if side == "right":
        missing_frac = (width - 1 - bounds["max_x"]) / max(width, 1)
    else:
        missing_frac = bounds["min_x"] / max(width, 1)

    return missing_frac >= HORIZONTAL_SIDE_MIN_MISSING_FRAC


def build_horizontal_rescue_roi(binary_top2, principal_mask, side):
    binary_top2 = normalize_mask(binary_top2)
    principal_mask = normalize_mask(principal_mask)

    height, _ = binary_top2.shape[:2]
    bounds = mask_bounds(principal_mask)

    roi_mask = np.zeros_like(binary_top2, dtype=np.uint8)

    if bounds is None:
        return roi_mask, 0.0

    y_center = int(round(get_edge_y(principal_mask, side)))

    top = max(0, y_center - HORIZONTAL_UP_HALF_HEIGHT_PX)
    bottom = min(height, y_center + HORIZONTAL_DOWN_HALF_HEIGHT_PX + 1)

    x1, x2 = get_horizontal_search_x_range(binary_top2, principal_mask, side)

    if x2 < x1:
        return roi_mask, float(y_center)

    roi_mask[top:bottom, x1 : x2 + 1] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)

    return roi_mask, float(y_center)


def build_guarded_binary_top2_for_horizontal_side(binary_top2, principal_mask, side):
    binary_top2 = normalize_mask(binary_top2)
    guarded = binary_top2.copy()

    height, _ = binary_top2.shape[:2]
    y_center = int(round(get_edge_y(principal_mask, side)))
    x1, x2 = get_horizontal_search_x_range(binary_top2, principal_mask, side)

    if x2 < x1:
        return guarded

    lower_limit = min(height, y_center + HORIZONTAL_DOWN_HALF_HEIGHT_PX + 1)
    guarded[lower_limit:height, x1 : x2 + 1] = 0

    return guarded


def score_horizontal_candidate(component_mask, principal_mask, y_center, side):
    component_mask = normalize_mask(component_mask)
    principal_mask = normalize_mask(principal_mask)

    stats = component_stats(component_mask)
    bounds = mask_bounds(principal_mask)

    if stats is None or bounds is None:
        return None

    if stats["area"] < HORIZONTAL_MIN_AREA:
        return None

    if stats["width"] < HORIZONTAL_MIN_WIDTH_PX:
        return None

    if (
        stats["verticality"] > HORIZONTAL_MAX_COMPONENT_VERTICALITY
        and stats["width"] < 35
    ):
        return None

    if side == "right":
        extension_gain = stats["max_x"] - bounds["max_x"]
    else:
        extension_gain = bounds["min_x"] - stats["min_x"]

    if extension_gain < HORIZONTAL_MIN_EXTENSION_GAIN_PX:
        return None

    ys, _ = np.where(component_mask > 0)
    signed_distances = ys.astype(np.float32) - float(y_center)

    if float(np.median(signed_distances)) > HORIZONTAL_DOWN_HALF_HEIGHT_PX:
        return None

    if float(np.percentile(signed_distances, 85)) > HORIZONTAL_DOWN_HALF_HEIGHT_PX + 2:
        return None

    abs_distances = np.abs(signed_distances)
    median_dist = float(np.median(abs_distances))

    width_score = min(1.0, stats["width"] / 120.0)
    area_score = min(1.0, stats["area"] / 220.0)
    distance_score = 1.0 - min(
        1.0,
        median_dist / max(HORIZONTAL_UP_HALF_HEIGHT_PX, 1),
    )
    gain_score = min(1.0, extension_gain / 120.0)
    vertical_penalty = min(1.4, stats["verticality"])

    score = 0.0
    score += 2.2 * width_score
    score += 1.2 * area_score
    score += 2.0 * distance_score
    score += 1.6 * gain_score
    score -= 0.65 * vertical_penalty

    return float(score)


def horizontal_rescue_for_side(binary_top2, principal_mask, side):
    binary_top2 = normalize_mask(binary_top2)
    principal_mask = normalize_mask(principal_mask)

    roi_mask, y_center = build_horizontal_rescue_roi(
        binary_top2,
        principal_mask,
        side,
    )

    not_principal = cv2.bitwise_not(principal_mask)

    candidate_mask = cv2.bitwise_and(binary_top2, roi_mask)
    candidate_mask = cv2.bitwise_and(candidate_mask, not_principal)

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        candidate_mask,
        connectivity=8,
    )

    accepted = []
    rejected_mask = np.zeros_like(binary_top2, dtype=np.uint8)

    for label in range(1, num_labels):
        component_mask = np.zeros_like(binary_top2, dtype=np.uint8)
        component_mask[labels == label] = 255

        score = score_horizontal_candidate(
            component_mask,
            principal_mask,
            y_center,
            side,
        )

        if score is None:
            rejected_mask[component_mask > 0] = 255
            continue

        accepted.append((score, component_mask))

    accepted.sort(key=lambda item: item[0], reverse=True)
    accepted = accepted[:HORIZONTAL_MAX_ACCEPTED_COMPONENTS_PER_SIDE]

    accepted_mask = np.zeros_like(binary_top2, dtype=np.uint8)

    for _, component_mask in accepted:
        accepted_mask[component_mask > 0] = 255

    return {
        "roi_mask": roi_mask,
        "candidate_mask": candidate_mask,
        "accepted_mask": accepted_mask,
        "rejected_mask": rejected_mask,
        "y_center": y_center,
        "used": int(np.count_nonzero(accepted_mask > 0)) > 0,
    }


def horizontal_rescue_before_secondary(
    binary_top2, principal_mask, traveler_points=None
):
    binary_top2 = normalize_mask(binary_top2)
    principal_mask = normalize_mask(principal_mask)

    rescue_mask = np.zeros_like(binary_top2, dtype=np.uint8)
    roi_mask = np.zeros_like(binary_top2, dtype=np.uint8)
    candidate_mask = np.zeros_like(binary_top2, dtype=np.uint8)
    accepted_mask = np.zeros_like(binary_top2, dtype=np.uint8)
    rejected_mask = np.zeros_like(binary_top2, dtype=np.uint8)

    if not HORIZONTAL_ENABLE or not is_horizontal_principal(principal_mask):
        return {
            "used": False,
            "rescue_mask": rescue_mask,
            "merged_mask": principal_mask.copy(),
            "binary_top2_guarded": binary_top2.copy(),
            "roi_mask": roi_mask,
            "candidate_mask": candidate_mask,
            "accepted_mask": accepted_mask,
            "rejected_mask": rejected_mask,
        }

    binary_top2_guarded = binary_top2.copy()

    for side in ["left", "right"]:
        if not side_has_space(binary_top2, principal_mask, side):
            continue

        side_result = horizontal_rescue_for_side(
            binary_top2,
            principal_mask,
            side,
        )

        roi_mask[side_result["roi_mask"] > 0] = 255
        candidate_mask[side_result["candidate_mask"] > 0] = 255
        accepted_mask[side_result["accepted_mask"] > 0] = 255
        rejected_mask[side_result["rejected_mask"] > 0] = 255
        rescue_mask[side_result["accepted_mask"] > 0] = 255

        if side_result["used"]:
            binary_top2_guarded = build_guarded_binary_top2_for_horizontal_side(
                binary_top2_guarded,
                merge_masks(principal_mask, rescue_mask),
                side,
            )

    merged_mask = merge_masks(principal_mask, rescue_mask)

    return {
        "used": int(np.count_nonzero(rescue_mask > 0)) > 0,
        "rescue_mask": rescue_mask,
        "merged_mask": merged_mask,
        "binary_top2_guarded": binary_top2_guarded,
        "roi_mask": roi_mask,
        "candidate_mask": candidate_mask,
        "accepted_mask": accepted_mask,
        "rejected_mask": rejected_mask,
    }


def draw_horizontal_rescue_debug(
    crop,
    principal_mask,
    rescue_mask,
    roi_mask,
    candidate_mask,
    accepted_mask,
    rejected_mask,
    traveler_points=None,
):
    result = to_bgr(crop)

    result = draw_mask_overlay(result, roi_mask, ROI_COLOR, alpha=0.22)
    result = draw_mask_overlay(result, candidate_mask, CANDIDATE_COLOR, alpha=0.38)
    result = draw_mask_overlay(result, rejected_mask, REJECTED_COLOR, alpha=0.38)
    result = draw_mask_overlay(result, principal_mask, PRINCIPAL_COLOR, alpha=0.65)
    result = draw_mask_overlay(result, accepted_mask, RESCUE_COLOR, alpha=0.85)

    if traveler_points is not None:
        result = draw_points(result, traveler_points, TRAVELER_COLOR, radius=1)

    return result


def draw_horizontal_merged_debug(crop, merged_mask, traveler_points=None):
    result = draw_mask_overlay(crop, merged_mask, MERGED_COLOR, alpha=0.75)

    if traveler_points is not None:
        result = draw_points(result, traveler_points, TRAVELER_COLOR, radius=1)

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

    if max_gain < HORIZONTAL_RESCUE_MIN_EXTENSION_PX:
        return False

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

        components.append(
            {
                "label": label,
                "area": area,
                "x": x,
                "y": y,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
                "centroid_y": float(centroids[label][1]),
            }
        )

    upper_candidates = []

    for component in components:
        is_upper_shape = (
            component["area"] >= HORIZONTAL_LAYERED_TAIL_MIN_UPPER_AREA
            and component["height"] <= HORIZONTAL_LAYERED_TAIL_MAX_UPPER_HEIGHT
        )

        is_near_right_end_of_principal = (
            component["x2"]
            >= principal_bounds["max_x"] + HORIZONTAL_LAYERED_TAIL_MIN_RIGHT_REGION_GAIN
        )

        if is_upper_shape and is_near_right_end_of_principal:
            upper_candidates.append(component)

    if len(upper_candidates) == 0:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    upper = min(upper_candidates, key=lambda item: item["centroid_y"])

    kept = np.zeros_like(horizontal_rescue_mask, dtype=np.uint8)
    removed = np.zeros_like(horizontal_rescue_mask, dtype=np.uint8)

    for component in components:
        pixels = labels == component["label"]

        if component["label"] == upper["label"]:
            kept[pixels] = 255
            continue

        is_lower_than_upper = (
            component["centroid_y"]
            >= upper["centroid_y"] + HORIZONTAL_LAYERED_TAIL_MIN_Y_GAP
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

        if (
            is_lower_than_upper
            and is_small_or_medium_tail
            and (overlaps_upper or is_right_tail_after_upper)
        ):
            removed[pixels] = 255
        else:
            kept[pixels] = 255

    if mask_area(kept) < 0.20 * mask_area(horizontal_rescue_mask):
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    return kept, removed


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

        components.append(
            {
                "label": label,
                "area": area,
                "x": x,
                "x2": x2,
                "y": y,
                "y2": y2,
                "width": width,
                "height": height,
            }
        )

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

        if is_small_right_isolated:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    return kept, removed


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
        y + height - 1

        component_median_y = float(centroids[label][1])

        right_gain = x2 - principal_bounds["max_x"]

        size_matches = (
            area <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_AREA
            and width >= HORIZONTAL_FLOATING_UPPER_STRIP_MIN_WIDTH
            and width <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_WIDTH
            and height <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_HEIGHT
        )

        starts_near_principal_edge = (
            x
            <= principal_bounds["max_x"]
            + HORIZONTAL_FLOATING_UPPER_STRIP_MAX_START_GAP_FROM_PRINCIPAL
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

        has_context = (
            len(context_ys) >= HORIZONTAL_FLOATING_UPPER_STRIP_MIN_CONTEXT_PIXELS
        )

        if has_context:
            local_reference_y = float(np.median(context_ys))
        else:
            local_reference_y = None

        is_above_local_principal_band = (
            local_reference_y is not None
            and component_median_y
            <= local_reference_y - HORIZONTAL_FLOATING_UPPER_STRIP_MIN_ABOVE_LOCAL_PX
        )

        dilated_principal = cv2.dilate(
            (principal_mask > 0).astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        direct_contact_pixels = int(
            np.count_nonzero(
                (component_pixels.astype(np.uint8) > 0) & (dilated_principal > 0)
            )
        )

        lacks_direct_contact = (
            direct_contact_pixels
            <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_DIRECT_CONTACT_PIXELS
        )

        if (
            size_matches
            and extends_right
            and has_context
            and is_above_local_principal_band
            and lacks_direct_contact
        ):
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    if mask_area(removed) == mask_area(horizontal_rescue_mask):
        if mask_area(horizontal_rescue_mask) > HORIZONTAL_FLOATING_UPPER_STRIP_MAX_AREA:
            return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    return kept, removed
