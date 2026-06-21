import cv2
import numpy as np

from postprocessing import empty_mask_like, get_mask_bounds, mask_area, mask_median_y

PRINCIPAL_COLOR = (0, 255, 0)
SECONDARY_COLOR = (0, 255, 255)
MERGED_COLOR = (0, 255, 0)
ROI_COLOR = (255, 0, 0)
CANDIDATE_COLOR = (255, 255, 0)
REJECTED_COLOR = (0, 0, 255)
TRAVELER_COLOR = (255, 180, 0)

SECONDARY_MAX_ITERATIONS = 8
SECONDARY_MAX_COMPONENTS_PER_SIDE_PER_ITERATION = 6

SECONDARY_SEARCH_WIDTH_FRAC = 0.12
SECONDARY_SEARCH_WIDTH_PX = 80

SECONDARY_MAX_EDGE_GAP_FRAC = 0.08
SECONDARY_MAX_EDGE_GAP_PX = 55

SECONDARY_MIN_AREA = 7
SECONDARY_MIN_WIDTH_FRAC = 0.006
SECONDARY_MIN_WIDTH_PX = 4

SECONDARY_MAX_VERTICALITY = 1.75
SECONDARY_VERTICALITY_WIDTH_FRAC = 0.10

SECONDARY_BAND_MIN_HALF_HEIGHT_PX = 20
SECONDARY_BAND_MAX_HALF_HEIGHT_FRAC = 0.12
SECONDARY_BAND_EXTRA_PX = 13
SECONDARY_BAND_MAD_SCALE = 3.4

SECONDARY_MAX_MEDIAN_DIST_FACTOR = 1.00
SECONDARY_MAX_P90_DIST_FACTOR = 1.42

SECONDARY_ACCEPT_MIN_SCORE = 1.00

SECONDARY_UPPER_ARTIFACT_GUARD_ENABLE = True

SECONDARY_UPPER_ARTIFACT_MEDIAN_FACTOR = 0.18
SECONDARY_UPPER_ARTIFACT_P90_FACTOR = 0.25
SECONDARY_UPPER_ARTIFACT_EDGE_FACTOR = 0.22

SECONDARY_UPPER_ARTIFACT_MIN_MEDIAN_PX = 5
SECONDARY_UPPER_ARTIFACT_MIN_P90_PX = 8
SECONDARY_UPPER_ARTIFACT_MIN_EDGE_PX = 7

SECONDARY_UPPER_ARTIFACT_MIN_GAP_PX = 5

MODEL_DEGREE_2_MIN_WIDTH_FRAC = 0.22
MODEL_DEGREE_2_MIN_POINTS = 45

SECONDARY_AFTER_HORIZONTAL_TAIL_GUARD_ENABLE = True
SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_HORIZONTAL_AREA = 150
SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_AREA = 220
SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_WIDTH = 45
SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_HEIGHT = 28
SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_RIGHT_GAP = -2
SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_BELOW_HORIZONTAL_PX = 32

SECONDARY_FLOATING_STRIP_AFTER_HORIZONTAL_REJECT_ENABLE = True

SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_AREA = 260
SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_WIDTH = 60
SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_HEIGHT = 35
SECONDARY_LOW_TAIL_AFTER_STRIP_MIN_RIGHT_GAIN = 55

HORIZONTAL_FLOATING_UPPER_STRIP_MIN_RIGHT_GAIN = 35
HORIZONTAL_FLOATING_UPPER_STRIP_MAX_RIGHT_GAIN = 90
HORIZONTAL_FLOATING_UPPER_STRIP_MAX_START_GAP_FROM_PRINCIPAL = 35


def normalize_binary_mask(binary_mask):
    if binary_mask.ndim == 3:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

    return (binary_mask > 0).astype(np.uint8) * 255


def to_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image.copy()


def draw_points(image, points, color, radius=2):
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


def draw_mask_overlay(base_image, mask, color, alpha=0.65):
    result = to_bgr(base_image)

    if mask is None:
        return result

    mask = normalize_binary_mask(mask)
    mask_bool = mask > 0

    if np.count_nonzero(mask_bool) == 0:
        return result

    color_array = np.array(color, dtype=np.float32)
    result_float = result.astype(np.float32)

    result_float[mask_bool] = (1.0 - alpha) * result_float[
        mask_bool
    ] + alpha * color_array

    return np.clip(result_float, 0, 255).astype(np.uint8)


def mask_to_bottom_points(mask):
    mask = normalize_binary_mask(mask)
    _, width = mask.shape[:2]

    points = []

    for x in range(width):
        ys = np.flatnonzero(mask[:, x] > 0)

        if len(ys) == 0:
            continue

        y = int(ys[-1])
        points.append((x, y))

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32)

    return np.array(points, dtype=np.int32)


def mask_bounds(mask):
    mask = normalize_binary_mask(mask)
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


def fit_secondary_model(points, image_shape):
    _, width = image_shape[:2]

    if points is None or len(points) == 0:

        def predict_empty(x_values):
            x_values = np.asarray(x_values, dtype=np.float32)
            return np.zeros_like(x_values, dtype=np.float32)

        return predict_empty, 0

    points = np.asarray(points, dtype=np.int32)

    xs = points[:, 0].astype(np.float32)
    ys = points[:, 1].astype(np.float32)

    min_x = int(np.min(xs))
    max_x = int(np.max(xs))
    point_width = max(1, max_x - min_x + 1)

    if (
        len(points) >= MODEL_DEGREE_2_MIN_POINTS
        and point_width >= MODEL_DEGREE_2_MIN_WIDTH_FRAC * width
    ):
        degree = 2
    elif len(points) >= 2:
        degree = 1
    else:
        degree = 0

    if degree == 0:
        median_y = float(np.median(ys))

        def predict_constant(x_values):
            x_values = np.asarray(x_values, dtype=np.float32)
            return np.full_like(x_values, median_y, dtype=np.float32)

        return predict_constant, degree

    try:
        coeffs = np.polyfit(xs, ys, degree)
    except Exception:
        median_y = float(np.median(ys))

        def predict_fallback(x_values):
            x_values = np.asarray(x_values, dtype=np.float32)
            return np.full_like(x_values, median_y, dtype=np.float32)

        return predict_fallback, 0

    def predict_poly(x_values):
        x_values = np.asarray(x_values, dtype=np.float32)
        return np.polyval(coeffs, x_values).astype(np.float32)

    return predict_poly, degree


def estimate_secondary_band_half_height(points, predict, image_shape):
    height, _ = image_shape[:2]

    max_half_height = max(
        SECONDARY_BAND_MIN_HALF_HEIGHT_PX,
        int(round(SECONDARY_BAND_MAX_HALF_HEIGHT_FRAC * height)),
    )

    if points is None or len(points) < 8:
        return max(
            SECONDARY_BAND_MIN_HALF_HEIGHT_PX,
            min(max_half_height, int(round(0.06 * height))),
        )

    points = np.asarray(points, dtype=np.int32)

    xs = points[:, 0].astype(np.float32)
    ys = points[:, 1].astype(np.float32)

    predicted = predict(xs)
    residuals = np.abs(ys - predicted)

    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))

    half_height = int(
        round(
            SECONDARY_BAND_EXTRA_PX
            + median_residual
            + SECONDARY_BAND_MAD_SCALE * max(mad, 1.0)
        )
    )

    half_height = max(SECONDARY_BAND_MIN_HALF_HEIGHT_PX, half_height)
    half_height = min(max_half_height, half_height)

    return int(half_height)


def build_side_roi(search_binary_mask, current_mask, current_points, side):
    search_binary = normalize_binary_mask(search_binary_mask)
    current_mask = normalize_binary_mask(current_mask)

    height, width = search_binary.shape[:2]

    bounds = mask_bounds(current_mask)

    roi_mask = np.zeros_like(search_binary, dtype=np.uint8)

    if bounds is None:
        return roi_mask, None, 0

    predict, _ = fit_secondary_model(current_points, search_binary.shape)
    half_height = estimate_secondary_band_half_height(
        current_points,
        predict,
        search_binary.shape,
    )

    search_width = max(
        SECONDARY_SEARCH_WIDTH_PX,
        int(round(SECONDARY_SEARCH_WIDTH_FRAC * width)),
    )

    if side == "left":
        x1 = max(0, bounds["min_x"] - search_width)
        x2 = max(0, bounds["min_x"] - 1)
    else:
        x1 = min(width - 1, bounds["max_x"] + 1)
        x2 = min(width - 1, bounds["max_x"] + search_width)

    if x2 < x1:
        return roi_mask, predict, half_height

    xs = np.arange(x1, x2 + 1, dtype=np.float32)
    ys = predict(xs)

    for x, center_y in zip(xs.astype(np.int32), ys):
        cy = int(round(center_y))

        top = max(0, cy - half_height)
        bottom = min(height, cy + half_height + 1)

        roi_mask[top:bottom, x] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)

    return roi_mask, predict, half_height


def component_mask_stats(component_mask):
    component_mask = normalize_binary_mask(component_mask)
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
    density = float(area / max(width * height, 1))

    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "width": width,
        "height": height,
        "area": area,
        "verticality": verticality,
        "density": density,
        "median_y": float(np.median(ys)),
    }


def edge_y_from_mask(mask, side):
    mask = normalize_binary_mask(mask)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return 0.0

    if side == "left":
        edge_x = int(np.min(xs))
        selected = ys[xs <= edge_x + 6]
    else:
        edge_x = int(np.max(xs))
        selected = ys[xs >= edge_x - 6]

    if len(selected) == 0:
        return float(np.median(ys))

    return float(np.median(selected))


def is_secondary_candidate_too_high(
    ys_float,
    predicted,
    candidate_edge_y,
    expected_edge_y,
    current_edge_y,
    half_height,
    gap,
):
    if not SECONDARY_UPPER_ARTIFACT_GUARD_ENABLE:
        return False

    if gap < SECONDARY_UPPER_ARTIFACT_MIN_GAP_PX:
        return False

    signed_distances = ys_float - predicted

    above_values = np.maximum(0.0, -signed_distances)

    median_signed = float(np.median(signed_distances))
    p90_above = float(np.percentile(above_values, 90))

    median_limit = max(
        SECONDARY_UPPER_ARTIFACT_MIN_MEDIAN_PX,
        SECONDARY_UPPER_ARTIFACT_MEDIAN_FACTOR * half_height,
    )

    p90_limit = max(
        SECONDARY_UPPER_ARTIFACT_MIN_P90_PX,
        SECONDARY_UPPER_ARTIFACT_P90_FACTOR * half_height,
    )

    edge_limit = max(
        SECONDARY_UPPER_ARTIFACT_MIN_EDGE_PX,
        SECONDARY_UPPER_ARTIFACT_EDGE_FACTOR * half_height,
    )

    clearly_above_model = median_signed < -median_limit and p90_above > p90_limit

    edge_too_high = (
        candidate_edge_y < expected_edge_y - edge_limit
        and candidate_edge_y < current_edge_y - edge_limit
    )

    return bool(clearly_above_model and edge_too_high)


def score_secondary_candidate(
    component_mask,
    current_mask,
    predict,
    half_height,
    image_shape,
    side,
):
    _, width = image_shape[:2]

    stats = component_mask_stats(component_mask)
    current_bounds = mask_bounds(current_mask)

    if stats is None or current_bounds is None:
        return None

    min_width = max(
        SECONDARY_MIN_WIDTH_PX,
        int(round(SECONDARY_MIN_WIDTH_FRAC * width)),
    )

    if stats["area"] < SECONDARY_MIN_AREA:
        return None

    if stats["width"] < min_width and stats["area"] < 2 * SECONDARY_MIN_AREA:
        return None

    if (
        stats["verticality"] > SECONDARY_MAX_VERTICALITY
        and stats["width"] < SECONDARY_VERTICALITY_WIDTH_FRAC * width
    ):
        return None

    if side == "left":
        gap = current_bounds["min_x"] - stats["max_x"]
        edge_x = stats["max_x"]
        current_edge_y = edge_y_from_mask(current_mask, "left")
        candidate_edge_y = edge_y_from_mask(component_mask, "right")
    else:
        gap = stats["min_x"] - current_bounds["max_x"]
        edge_x = stats["min_x"]
        current_edge_y = edge_y_from_mask(current_mask, "right")
        candidate_edge_y = edge_y_from_mask(component_mask, "left")

    if gap < 0:
        return None

    max_gap = max(
        SECONDARY_MAX_EDGE_GAP_PX,
        int(round(SECONDARY_MAX_EDGE_GAP_FRAC * width)),
    )

    if gap > max_gap:
        return None

    ys, xs = np.where(component_mask > 0)

    xs_float = xs.astype(np.float32)
    ys_float = ys.astype(np.float32)

    predicted = predict(xs_float)
    distances = np.abs(ys_float - predicted)

    median_dist = float(np.median(distances))
    p90_dist = float(np.percentile(distances, 90))

    if median_dist > SECONDARY_MAX_MEDIAN_DIST_FACTOR * half_height:
        return None

    if p90_dist > SECONDARY_MAX_P90_DIST_FACTOR * half_height:
        return None

    expected_edge_y = float(predict(np.array([edge_x], dtype=np.float32))[0])

    if is_secondary_candidate_too_high(
        ys_float=ys_float,
        predicted=predicted,
        candidate_edge_y=candidate_edge_y,
        expected_edge_y=expected_edge_y,
        current_edge_y=current_edge_y,
        half_height=half_height,
        gap=gap,
    ):
        return None

    edge_dist_to_model = abs(candidate_edge_y - expected_edge_y)
    edge_dist_to_current = abs(candidate_edge_y - current_edge_y)

    edge_dist = min(edge_dist_to_model, edge_dist_to_current)

    if edge_dist > half_height * 1.20:
        return None

    width_score = min(1.0, stats["width"] / max(0.18 * width, 1))
    area_score = min(1.0, stats["area"] / 240.0)
    distance_score = 1.0 - min(1.0, median_dist / max(half_height, 1))
    gap_score = 1.0 - min(1.0, gap / max(max_gap, 1))
    edge_score = 1.0 - min(1.0, edge_dist / max(half_height, 1))

    score = 0.0
    score += 2.2 * width_score
    score += 1.4 * area_score
    score += 2.0 * distance_score
    score += 1.2 * gap_score
    score += 1.2 * edge_score
    score += 0.5 * min(1.0, stats["density"])
    score -= 0.7 * min(1.6, stats["verticality"])

    if score < SECONDARY_ACCEPT_MIN_SCORE:
        return None

    return float(score)


def find_secondary_components_for_side(
    search_binary_mask,
    current_mask,
    current_points,
    side,
):
    search_binary = normalize_binary_mask(search_binary_mask)
    current_mask = normalize_binary_mask(current_mask)

    roi_mask, predict, half_height = build_side_roi(
        search_binary,
        current_mask,
        current_points,
        side,
    )

    if predict is None:
        empty = np.zeros_like(search_binary, dtype=np.uint8)

        return {
            "accepted_components": [],
            "accepted_mask": empty,
            "roi_mask": roi_mask,
            "candidate_mask": empty,
            "rejected_mask": empty,
        }

    not_current = cv2.bitwise_not(current_mask)

    candidate_mask = cv2.bitwise_and(search_binary, roi_mask)
    candidate_mask = cv2.bitwise_and(candidate_mask, not_current)

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        candidate_mask,
        connectivity=8,
    )

    accepted = []
    rejected_mask = np.zeros_like(search_binary, dtype=np.uint8)

    for label in range(1, num_labels):
        component_mask = np.zeros_like(search_binary, dtype=np.uint8)
        component_mask[labels == label] = 255

        score = score_secondary_candidate(
            component_mask,
            current_mask,
            predict,
            half_height,
            search_binary.shape,
            side,
        )

        if score is None:
            rejected_mask[component_mask > 0] = 255
            continue

        accepted.append((score, component_mask))

    accepted.sort(key=lambda item: item[0], reverse=True)

    accepted = accepted[:SECONDARY_MAX_COMPONENTS_PER_SIDE_PER_ITERATION]

    accepted_mask = np.zeros_like(search_binary, dtype=np.uint8)

    for _, component_mask in accepted:
        accepted_mask[component_mask > 0] = 255

    return {
        "accepted_components": accepted,
        "accepted_mask": accepted_mask,
        "roi_mask": roi_mask,
        "candidate_mask": candidate_mask,
        "rejected_mask": rejected_mask,
    }


def build_secondary_components(
    binary_top1, binary_top2, principal_mask, traveler_points
):
    binary_top1 = normalize_binary_mask(binary_top1)
    binary_top2 = normalize_binary_mask(binary_top2)
    principal_mask = normalize_binary_mask(principal_mask)

    current_mask = principal_mask.copy()
    secondary_mask = np.zeros_like(binary_top1, dtype=np.uint8)

    all_roi_mask = np.zeros_like(binary_top1, dtype=np.uint8)
    all_candidate_mask = np.zeros_like(binary_top1, dtype=np.uint8)
    all_rejected_mask = np.zeros_like(binary_top1, dtype=np.uint8)
    all_accepted_mask = np.zeros_like(binary_top1, dtype=np.uint8)

    added_components = []

    for iteration in range(SECONDARY_MAX_ITERATIONS):
        added_this_round = False

        for side in ["left", "right"]:
            current_points = mask_to_bottom_points(current_mask)

            if len(current_points) < 8 and traveler_points is not None:
                current_points = np.asarray(traveler_points, dtype=np.int32)

            result = find_secondary_components_for_side(
                binary_top2,
                current_mask,
                current_points,
                side,
            )

            all_roi_mask[result["roi_mask"] > 0] = 255
            all_candidate_mask[result["candidate_mask"] > 0] = 255
            all_rejected_mask[result["rejected_mask"] > 0] = 255
            all_accepted_mask[result["accepted_mask"] > 0] = 255

            accepted_components = result["accepted_components"]

            if len(accepted_components) == 0:
                continue

            for score, component_mask in accepted_components:
                current_mask[component_mask > 0] = 255
                secondary_mask[component_mask > 0] = 255

                added_components.append(
                    {
                        "iteration": iteration + 1,
                        "side": side,
                        "score": score,
                    }
                )

            added_this_round = True

        if not added_this_round:
            break

    merged_mask = np.zeros_like(binary_top1, dtype=np.uint8)
    merged_mask[principal_mask > 0] = 255
    merged_mask[secondary_mask > 0] = 255

    return {
        "secondary_mask": secondary_mask,
        "merged_mask": merged_mask,
        "roi_mask": all_roi_mask,
        "candidate_mask": all_candidate_mask,
        "rejected_mask": all_rejected_mask,
        "accepted_mask": all_accepted_mask,
        "added_components": added_components,
    }


def draw_secondary_roi(base_image, roi_mask, principal_mask=None, traveler_points=None):
    result = draw_mask_overlay(base_image, roi_mask, ROI_COLOR, alpha=0.28)

    if principal_mask is not None:
        result = draw_mask_overlay(result, principal_mask, PRINCIPAL_COLOR, alpha=0.55)

    if traveler_points is not None:
        result = draw_points(result, traveler_points, TRAVELER_COLOR, radius=1)

    return result


def draw_secondary_candidates(
    base_image,
    candidate_mask,
    rejected_mask=None,
    traveler_points=None,
):
    result = draw_mask_overlay(base_image, candidate_mask, CANDIDATE_COLOR, alpha=0.55)

    if rejected_mask is not None:
        result = draw_mask_overlay(result, rejected_mask, REJECTED_COLOR, alpha=0.45)

    if traveler_points is not None:
        result = draw_points(result, traveler_points, TRAVELER_COLOR, radius=1)

    return result


def draw_secondary_components(
    base_image,
    principal_mask,
    secondary_mask,
    traveler_points=None,
):
    result = to_bgr(base_image)

    result = draw_mask_overlay(result, principal_mask, PRINCIPAL_COLOR, alpha=0.65)
    result = draw_mask_overlay(result, secondary_mask, SECONDARY_COLOR, alpha=0.80)

    if traveler_points is not None:
        result = draw_points(result, traveler_points, TRAVELER_COLOR, radius=1)

    return result


def draw_merged_components(base_image, merged_mask, traveler_points=None):
    result = draw_mask_overlay(base_image, merged_mask, MERGED_COLOR, alpha=0.75)

    if traveler_points is not None:
        result = draw_points(result, traveler_points, TRAVELER_COLOR, radius=1)

    return result


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
        y2 = y + height - 1

        component_pixels = labels == label
        component_median_y = float(centroids[label][1])

        is_small_tail = (
            area <= SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_AREA
            and width <= SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_WIDTH
            and height <= SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_HEIGHT
        )

        starts_after_horizontal_end = (
            x
            >= horizontal_bounds["max_x"]
            + SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_RIGHT_GAP
        )

        is_far_below_horizontal = (
            component_median_y
            >= horizontal_median_y
            + SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_BELOW_HORIZONTAL_PX
            or y2
            >= horizontal_median_y
            + SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_BELOW_HORIZONTAL_PX
            + 8
        )

        vertical_overlap_top = max(y, horizontal_bounds["min_y"])
        vertical_overlap_bottom = min(y2, horizontal_bounds["max_y"])
        vertical_overlap = max(0, vertical_overlap_bottom - vertical_overlap_top + 1)
        vertical_overlap_frac = vertical_overlap / max(height, 1)

        is_thin_edge_fragment = area <= 90 and width <= 28 and height <= 10

        is_only_touching_horizontal_edge = (
            vertical_overlap_frac <= 0.30 or y2 <= horizontal_bounds["min_y"] + 1
        )

        is_after_or_at_horizontal_end = x >= horizontal_bounds["max_x"] - 1

        remove_low_tail = (
            is_small_tail and starts_after_horizontal_end and is_far_below_horizontal
        )

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
            x
            <= principal_bounds["max_x"]
            + HORIZONTAL_FLOATING_UPPER_STRIP_MAX_START_GAP_FROM_PRINCIPAL
        )
        moderate_right_gain = (
            right_gain >= HORIZONTAL_FLOATING_UPPER_STRIP_MIN_RIGHT_GAIN
            and right_gain <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_RIGHT_GAIN
            and starts_near_principal_edge
        )

        contact_pixels = int(
            np.count_nonzero(
                (component_pixels.astype(np.uint8) > 0) & (dilated_principal > 0)
            )
        )

        is_long_thin_right_strip = (
            250 <= area <= 850
            and 45 <= width <= 150
            and height <= 20
            and moderate_right_gain
            and contact_pixels <= 12
        )

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
