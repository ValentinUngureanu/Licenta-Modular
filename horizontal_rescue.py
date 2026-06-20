import cv2
import numpy as np


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

    roi_mask[top:bottom, x1:x2 + 1] = 255

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
    guarded[lower_limit:height, x1:x2 + 1] = 0

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


def horizontal_rescue_before_secondary(binary_top2, principal_mask, traveler_points=None):
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
