import cv2
import numpy as np

PRINCIPAL_COLOR = (0, 255, 0)
SECONDARY_COLOR = (0, 255, 255)
RESCUE_COLOR = (255, 0, 255)
ROI_COLOR = (255, 0, 0)
CANDIDATE_COLOR = (255, 255, 0)
REJECTED_COLOR = (0, 0, 255)
TRAVELER_COLOR = (255, 180, 0)
MERGED_COLOR = (0, 255, 0)

GAP_RESCUE_ENABLE = True

GAP_TRIGGER_MAX_SECONDARY_AREA = 650
GAP_TRIGGER_MAX_SECONDARY_TO_PRINCIPAL = 0.22
GAP_TRIGGER_MIN_PRINCIPAL_WIDTH = 250

GAP_OVERLAP_PX = 35
GAP_SEARCH_WIDTH_PX = 340
GAP_SEARCH_WIDTH_FRAC = 0.42
GAP_SIDE_MIN_MISSING_FRAC = 0.04

GAP_UP_HALF_HEIGHT_PX = 36
GAP_DOWN_HALF_HEIGHT_PX = 32

GAP_MIN_AREA = 55
GAP_MIN_WIDTH_PX = 18
GAP_MIN_EXTENSION_GAIN_PX = 35
GAP_MAX_START_GAP_PX = 80
GAP_MAX_COMPONENT_HEIGHT_PX = 70
GAP_MAX_AREA_FACTOR = 1.70
GAP_MAX_MEDIAN_DISTANCE_PX = 32
GAP_MAX_P90_DISTANCE_PX = 44
GAP_MAX_ACCEPTED_COMPONENTS_PER_SIDE = 1

GAP_LOCAL_EDGE_GUARD_ENABLE = True
GAP_LOCAL_EDGE_ACTIVATE_DIFF_PX = 34
GAP_LOCAL_EDGE_MAX_ABOVE_EDGE_PX = 28

GAP_LOCAL_EDGE_PROTECT_SMALL_BRIDGE_AREA = 80
GAP_LOCAL_EDGE_PROTECT_SMALL_BRIDGE_WIDTH = 22
GAP_LOCAL_EDGE_PROTECT_SMALL_BRIDGE_HEIGHT = 9


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

    result_float[mask_bool] = (
            (1.0 - alpha) * result_float[mask_bool]
            + alpha * color_array
    )

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

    if mask_a is not None:
        result[mask_a > 0] = 255

    if mask_b is not None:
        result[mask_b > 0] = 255

    return result


def mask_area(mask) -> int:
    if mask is None:
        return 0

    return int(np.count_nonzero(mask > 0))


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

    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "width": width,
        "height": height,
        "area": area,
        "median_x": float(np.median(xs)),
        "median_y": float(np.median(ys)),
        "center_x": float(np.mean(xs)),
        "center_y": float(np.mean(ys)),
        "verticality": float(height / max(width, 1)),
    }


def edge_y_from_mask(mask, side):
    mask = normalize_mask(mask)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return 0.0

    if side == "left":
        edge_x = int(np.min(xs))
        selected = ys[xs <= edge_x + 8]
    else:
        edge_x = int(np.max(xs))
        selected = ys[xs >= edge_x - 8]

    if len(selected) == 0:
        return float(np.median(ys))

    return float(np.median(selected))


def is_gap_candidate_wrong_for_local_edge(component_mask, principal_mask, y_center, side):
    if not GAP_LOCAL_EDGE_GUARD_ENABLE:
        return False

    if side != "left":
        return False

    component_mask = normalize_mask(component_mask)
    principal_mask = normalize_mask(principal_mask)

    stats = component_stats(component_mask)

    if stats is None:
        return False

    is_small_bridge = (
            stats["area"] <= GAP_LOCAL_EDGE_PROTECT_SMALL_BRIDGE_AREA
            and stats["width"] <= GAP_LOCAL_EDGE_PROTECT_SMALL_BRIDGE_WIDTH
            and stats["height"] <= GAP_LOCAL_EDGE_PROTECT_SMALL_BRIDGE_HEIGHT
    )

    if is_small_bridge:
        return False

    principal_edge_y = edge_y_from_mask(principal_mask, side)

    edge_is_much_lower_than_center = (
            principal_edge_y > float(y_center) + GAP_LOCAL_EDGE_ACTIVATE_DIFF_PX
    )

    if not edge_is_much_lower_than_center:
        return False

    component_too_high_for_edge = (
            stats["median_y"] < principal_edge_y - GAP_LOCAL_EDGE_MAX_ABOVE_EDGE_PX
    )

    return bool(component_too_high_for_edge)


def should_try_gap_rescue(principal_mask, secondary_mask):
    if not GAP_RESCUE_ENABLE:
        return False

    principal_mask = normalize_mask(principal_mask)
    secondary_mask = normalize_mask(secondary_mask)

    principal_bounds = mask_bounds(principal_mask)

    if principal_bounds is None:
        return False

    principal_area = mask_area(principal_mask)
    secondary_area = mask_area(secondary_mask)

    if principal_area == 0:
        return False

    if principal_bounds["width"] < GAP_TRIGGER_MIN_PRINCIPAL_WIDTH:
        return False

    if secondary_area > GAP_TRIGGER_MAX_SECONDARY_AREA:
        return False

    if secondary_area > GAP_TRIGGER_MAX_SECONDARY_TO_PRINCIPAL * principal_area:
        return False

    return True


def side_has_space(binary_top2, principal_mask, side):
    binary_top2 = normalize_mask(binary_top2)
    principal_bounds = mask_bounds(principal_mask)

    if principal_bounds is None:
        return False

    _, width = binary_top2.shape[:2]

    if side == "right":
        missing_frac = (width - 1 - principal_bounds["max_x"]) / max(width, 1)
    else:
        missing_frac = principal_bounds["min_x"] / max(width, 1)

    return missing_frac >= GAP_SIDE_MIN_MISSING_FRAC


def build_gap_roi_for_side(binary_top2, principal_mask, side):
    binary_top2 = normalize_mask(binary_top2)
    principal_mask = normalize_mask(principal_mask)

    height, width = binary_top2.shape[:2]
    principal_bounds = mask_bounds(principal_mask)

    roi_mask = np.zeros_like(binary_top2, dtype=np.uint8)

    if principal_bounds is None:
        return roi_mask, 0.0

    y_center = int(round(principal_bounds["median_y"]))

    search_width = max(
        GAP_SEARCH_WIDTH_PX,
        int(round(GAP_SEARCH_WIDTH_FRAC * width)),
    )

    if side == "right":
        x1 = max(0, principal_bounds["max_x"] - GAP_OVERLAP_PX)
        x2 = min(width - 1, principal_bounds["max_x"] + search_width)
    else:
        x1 = max(0, principal_bounds["min_x"] - search_width)
        x2 = min(width - 1, principal_bounds["min_x"] + GAP_OVERLAP_PX)

    top = max(0, y_center - GAP_UP_HALF_HEIGHT_PX)
    bottom = min(height, y_center + GAP_DOWN_HALF_HEIGHT_PX + 1)

    if x2 < x1 or bottom <= top:
        return roi_mask, float(y_center)

    roi_mask[top:bottom, x1:x2 + 1] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)

    return roi_mask, float(y_center)


def score_gap_candidate(component_mask, principal_mask, y_center, side):
    component_mask = normalize_mask(component_mask)
    principal_mask = normalize_mask(principal_mask)

    stats = component_stats(component_mask)
    principal_bounds = mask_bounds(principal_mask)

    if stats is None or principal_bounds is None:
        return None

    principal_area = mask_area(principal_mask)

    if stats["area"] < GAP_MIN_AREA:
        return None

    if stats["width"] < GAP_MIN_WIDTH_PX:
        return None

    if stats["height"] > GAP_MAX_COMPONENT_HEIGHT_PX:
        return None

    if stats["area"] > GAP_MAX_AREA_FACTOR * max(principal_area, 1):
        return None

    if side == "right":
        extension_gain = stats["max_x"] - principal_bounds["max_x"]
        start_gap = stats["min_x"] - principal_bounds["max_x"]
    else:
        extension_gain = principal_bounds["min_x"] - stats["min_x"]
        start_gap = principal_bounds["min_x"] - stats["max_x"]

    if extension_gain < GAP_MIN_EXTENSION_GAIN_PX:
        return None

    if start_gap > GAP_MAX_START_GAP_PX:
        return None

    if is_gap_candidate_wrong_for_local_edge(
            component_mask,
            principal_mask,
            y_center,
            side,
    ):
        return None

    ys, _ = np.where(component_mask > 0)
    distances = np.abs(ys.astype(np.float32) - float(y_center))

    median_dist = float(np.median(distances))
    p90_dist = float(np.percentile(distances, 90))

    if median_dist > GAP_MAX_MEDIAN_DISTANCE_PX:
        return None

    if p90_dist > GAP_MAX_P90_DISTANCE_PX:
        return None

    width_score = min(1.0, stats["width"] / 220.0)
    area_score = min(1.0, stats["area"] / 2500.0)
    gain_score = min(1.0, extension_gain / 260.0)
    distance_score = 1.0 - min(1.0, median_dist / max(GAP_MAX_MEDIAN_DISTANCE_PX, 1))
    height_penalty = min(1.0, stats["height"] / max(GAP_MAX_COMPONENT_HEIGHT_PX, 1))

    score = 0.0
    score += 2.2 * width_score
    score += 1.6 * area_score
    score += 1.7 * gain_score
    score += 2.0 * distance_score
    score -= 0.7 * height_penalty

    return float(score)


def gap_rescue_for_side(binary_top2, principal_mask, already_merged_mask, side):
    binary_top2 = normalize_mask(binary_top2)
    principal_mask = normalize_mask(principal_mask)
    already_merged_mask = normalize_mask(already_merged_mask)

    roi_mask, y_center = build_gap_roi_for_side(
        binary_top2,
        principal_mask,
        side,
    )

    not_already_merged = cv2.bitwise_not(already_merged_mask)
    candidate_mask = cv2.bitwise_and(binary_top2, roi_mask)
    candidate_mask = cv2.bitwise_and(candidate_mask, not_already_merged)

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        candidate_mask,
        connectivity=8,
    )

    accepted = []
    rejected_mask = np.zeros_like(binary_top2, dtype=np.uint8)

    for label in range(1, num_labels):
        component_mask = np.zeros_like(binary_top2, dtype=np.uint8)
        component_mask[labels == label] = 255

        score = score_gap_candidate(
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
    accepted = accepted[:GAP_MAX_ACCEPTED_COMPONENTS_PER_SIDE]

    accepted_mask = np.zeros_like(binary_top2, dtype=np.uint8)

    for _, component_mask in accepted:
        accepted_mask[component_mask > 0] = 255

    return {
        "roi_mask": roi_mask,
        "candidate_mask": candidate_mask,
        "accepted_mask": accepted_mask,
        "rejected_mask": rejected_mask,
        "used": int(np.count_nonzero(accepted_mask > 0)) > 0,
    }


def gap_rescue_after_secondary(
        binary_top2,
        principal_mask,
        secondary_mask,
        merged_mask,
        traveler_points=None,
):
    binary_top2 = normalize_mask(binary_top2)
    principal_mask = normalize_mask(principal_mask)
    secondary_mask = normalize_mask(secondary_mask)
    merged_mask = normalize_mask(merged_mask)

    rescue_mask = np.zeros_like(binary_top2, dtype=np.uint8)
    roi_mask = np.zeros_like(binary_top2, dtype=np.uint8)
    candidate_mask = np.zeros_like(binary_top2, dtype=np.uint8)
    accepted_mask = np.zeros_like(binary_top2, dtype=np.uint8)
    rejected_mask = np.zeros_like(binary_top2, dtype=np.uint8)

    if not should_try_gap_rescue(principal_mask, secondary_mask):
        return {
            "used": False,
            "secondary_mask": secondary_mask.copy(),
            "rescue_mask": rescue_mask,
            "merged_mask": merged_mask.copy(),
            "roi_mask": roi_mask,
            "candidate_mask": candidate_mask,
            "accepted_mask": accepted_mask,
            "rejected_mask": rejected_mask,
        }

    for side in ["left", "right"]:
        if not side_has_space(binary_top2, principal_mask, side):
            continue

        side_result = gap_rescue_for_side(
            binary_top2,
            principal_mask,
            merge_masks(merged_mask, rescue_mask),
            side,
        )

        roi_mask[side_result["roi_mask"] > 0] = 255
        candidate_mask[side_result["candidate_mask"] > 0] = 255
        accepted_mask[side_result["accepted_mask"] > 0] = 255
        rejected_mask[side_result["rejected_mask"] > 0] = 255
        rescue_mask[side_result["accepted_mask"] > 0] = 255

    secondary_after_gap = merge_masks(secondary_mask, rescue_mask)
    merged_after_gap = merge_masks(merged_mask, rescue_mask)

    return {
        "used": int(np.count_nonzero(rescue_mask > 0)) > 0,
        "secondary_mask": secondary_after_gap,
        "rescue_mask": rescue_mask,
        "merged_mask": merged_after_gap,
        "roi_mask": roi_mask,
        "candidate_mask": candidate_mask,
        "accepted_mask": accepted_mask,
        "rejected_mask": rejected_mask,
    }


def draw_gap_rescue_debug(
        crop,
        principal_mask,
        secondary_mask,
        rescue_mask,
        roi_mask,
        candidate_mask,
        accepted_mask,
        rejected_mask,
        traveler_points=None,
):
    result = to_bgr(crop)

    result = draw_mask_overlay(result, roi_mask, ROI_COLOR, alpha=0.22)
    result = draw_mask_overlay(result, candidate_mask, CANDIDATE_COLOR, alpha=0.36)
    result = draw_mask_overlay(result, rejected_mask, REJECTED_COLOR, alpha=0.36)
    result = draw_mask_overlay(result, principal_mask, PRINCIPAL_COLOR, alpha=0.65)
    result = draw_mask_overlay(result, secondary_mask, SECONDARY_COLOR, alpha=0.65)
    result = draw_mask_overlay(result, accepted_mask, RESCUE_COLOR, alpha=0.85)

    if traveler_points is not None:
        result = draw_points(result, traveler_points, TRAVELER_COLOR, radius=1)

    return result


def draw_gap_merged_debug(crop, merged_mask, traveler_points=None):
    result = draw_mask_overlay(crop, merged_mask, MERGED_COLOR, alpha=0.75)

    if traveler_points is not None:
        result = draw_points(result, traveler_points, TRAVELER_COLOR, radius=1)

    return result
