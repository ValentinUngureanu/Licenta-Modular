import cv2
import numpy as np

from postprocessing import (
    empty_mask_like,
    get_mask_bounds,
    mask_area,
    mask_density,
    mask_median_y,
    merge_masks,
)

PRINCIPAL_COLOR = (0, 255, 0)
SECONDARY_COLOR = (0, 255, 255)
RESCUE_COLOR = (255, 0, 255)
MERGED_COLOR = (0, 255, 0)
ROI_COLOR = (255, 0, 0)
CANDIDATE_COLOR = (255, 255, 0)
REJECTED_COLOR = (0, 0, 255)
TRAVELER_COLOR = (255, 180, 0)

RESCUE_MAX_ITERATIONS = 3

RESCUE_TRIGGER_MIN_WIDTH_FRAC = 0.55
RESCUE_TRIGGER_MIN_SIDE_FREE_FRAC = 0.18

RESCUE_SEARCH_WIDTH_FRAC = 0.34
RESCUE_SEARCH_WIDTH_PX = 230

RESCUE_MAX_GAP_FRAC = 0.20
RESCUE_MAX_GAP_PX = 170

RESCUE_MIN_EXTENSION_GAIN_FRAC = 0.018
RESCUE_MIN_EXTENSION_GAIN_PX = 14

RESCUE_BAND_MIN_HALF_HEIGHT_PX = 22
RESCUE_BAND_MAX_HALF_HEIGHT_FRAC = 0.12
RESCUE_BAND_EXTRA_PX = 14
RESCUE_BAND_MAD_SCALE = 3.2

RESCUE_ENDPOINT_FIT_WINDOW_FRAC = 0.12
RESCUE_ENDPOINT_FIT_WINDOW_PX = 80
RESCUE_MAX_ENDPOINT_SLOPE = 0.35

RESCUE_MIN_AREA = 6
RESCUE_MIN_WIDTH_PX = 5
RESCUE_MIN_WIDTH_FRAC = 0.006
RESCUE_MAX_VERTICALITY = 1.9
RESCUE_VERTICALITY_WIDTH_FRAC = 0.10

RESCUE_MAX_MEDIAN_DIST_FACTOR = 1.15
RESCUE_MAX_P90_DIST_FACTOR = 1.65
RESCUE_MAX_EDGE_DIST_FACTOR = 1.45
RESCUE_ACCEPT_MIN_SCORE = 1.05
RESCUE_MAX_COMPONENTS_PER_SIDE = 2

RESCUE_UPPER_ARTIFACT_GUARD_ENABLE = True
RESCUE_UPPER_ARTIFACT_MIN_GAP_PX = 8
RESCUE_UPPER_ARTIFACT_EDGE_ABOVE_MODEL_PX = 14
RESCUE_UPPER_ARTIFACT_EDGE_ABOVE_CURRENT_PX = 14
RESCUE_UPPER_ARTIFACT_MEDIAN_ABOVE_MODEL_PX = 12
RESCUE_UPPER_ARTIFACT_P90_ABOVE_MODEL_PX = 18

GUARD_ENABLE = True
GUARD_MAX_SECONDARY_TO_PRINCIPAL_AREA = 2.75
GUARD_MAX_SECONDARY_TO_PRINCIPAL_WIDTH = 1.45
GUARD_KEEP_NEAR_HALF_HEIGHT_MULTIPLIER = 1.10

SECONDARY_RESCUE_OVERGROWTH_AREA_RATIO = 3.0
SECONDARY_RESCUE_OVERGROWTH_MIN_AREA = 9000
SECONDARY_RESCUE_OVERGROWTH_MIN_DENSITY = 0.18
SECONDARY_RESCUE_OVERGROWTH_MIN_HEIGHT = 125
SECONDARY_RESCUE_MIN_KEEP_FRAC_WHEN_NOT_OVERGROWN = 0.65
SECONDARY_RESCUE_MIN_WIDTH_KEEP_FRAC_WHEN_NOT_OVERGROWN = 0.88

RIGHT_ISOLATED_SECONDARY_RESCUE_GUARD_ENABLE = True
RIGHT_ISOLATED_SECONDARY_RESCUE_MIN_GAP_PX = 60
RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_AREA = 650
RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_WIDTH = 130
RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_HEIGHT = 80
RIGHT_ISOLATED_SECONDARY_RESCUE_MIN_BELOW_MEDIAN_PX = 45


def normalize_binary_mask(mask):
    if mask is None:
        return None

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return (mask > 0).astype(np.uint8) * 255


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


def build_guidance_points(principal_mask, traveler_points):
    principal_points = mask_to_bottom_points(principal_mask)

    if traveler_points is None:
        traveler_points = np.empty((0, 2), dtype=np.int32)
    else:
        traveler_points = np.asarray(traveler_points, dtype=np.int32)

    if len(principal_points) == 0 and len(traveler_points) == 0:
        return np.empty((0, 2), dtype=np.int32)

    if len(principal_points) == 0:
        points = traveler_points.copy()
    elif len(traveler_points) == 0:
        points = principal_points.copy()
    else:
        points = np.vstack([principal_points, traveler_points]).astype(np.int32)

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32)

    points = points[np.argsort(points[:, 0])]

    grouped = {}

    for x, y in points:
        x = int(x)
        y = int(y)

        if x not in grouped:
            grouped[x] = []

        grouped[x].append(y)

    result = []

    for x in sorted(grouped.keys()):
        y = int(round(np.median(grouped[x])))
        result.append((x, y))

    return np.array(result, dtype=np.int32)


def edge_y_from_mask(mask, side):
    mask = normalize_binary_mask(mask)
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


def build_endpoint_predictor(
    guidance_points, current_mask, current_bounds, side, image_shape
):
    height, width = image_shape[:2]

    if side == "left":
        edge_x = int(current_bounds["min_x"])
    else:
        edge_x = int(current_bounds["max_x"])

    current_edge_y = edge_y_from_mask(current_mask, side)

    if guidance_points is None or len(guidance_points) == 0:
        anchor_y = current_edge_y
        slope = 0.0
    else:
        points = np.asarray(guidance_points, dtype=np.int32)

        fit_window = max(
            RESCUE_ENDPOINT_FIT_WINDOW_PX,
            int(round(RESCUE_ENDPOINT_FIT_WINDOW_FRAC * width)),
        )

        if side == "left":
            local_mask = points[:, 0] <= edge_x + fit_window
        else:
            local_mask = points[:, 0] >= edge_x - fit_window

        local_points = points[local_mask]

        if len(local_points) < 6:
            local_points = points

        xs = local_points[:, 0].astype(np.float32)
        ys = local_points[:, 1].astype(np.float32)

        if len(local_points) >= 2 and np.max(xs) > np.min(xs):
            try:
                coeffs = np.polyfit(xs, ys, 1)
                slope = float(coeffs[0])
                guidance_edge_y = float(np.polyval(coeffs, edge_x))
            except Exception:
                slope = 0.0
                guidance_edge_y = float(np.median(ys))
        else:
            slope = 0.0
            guidance_edge_y = float(np.median(ys))

        slope = float(
            np.clip(
                slope,
                -RESCUE_MAX_ENDPOINT_SLOPE,
                RESCUE_MAX_ENDPOINT_SLOPE,
            )
        )

        if current_edge_y <= 0:
            anchor_y = guidance_edge_y
        elif abs(current_edge_y - guidance_edge_y) > 35:
            anchor_y = guidance_edge_y
        else:
            anchor_y = 0.70 * guidance_edge_y + 0.30 * current_edge_y

    anchor_y = float(np.clip(anchor_y, 0, height - 1))

    def predict(x_values):
        x_values = np.asarray(x_values, dtype=np.float32)
        y_values = anchor_y + slope * (x_values - float(edge_x))
        y_values = np.clip(y_values, 0, height - 1)
        return y_values.astype(np.float32)

    return predict, edge_x, anchor_y, slope


def estimate_band_half_height(guidance_points, predict, edge_x, side, image_shape):
    height, width = image_shape[:2]

    max_half_height = max(
        RESCUE_BAND_MIN_HALF_HEIGHT_PX,
        int(round(RESCUE_BAND_MAX_HALF_HEIGHT_FRAC * height)),
    )

    if guidance_points is None or len(guidance_points) < 8:
        return max(
            RESCUE_BAND_MIN_HALF_HEIGHT_PX,
            min(max_half_height, int(round(0.06 * height))),
        )

    points = np.asarray(guidance_points, dtype=np.int32)

    fit_window = max(
        RESCUE_ENDPOINT_FIT_WINDOW_PX,
        int(round(RESCUE_ENDPOINT_FIT_WINDOW_FRAC * width)),
    )

    if side == "left":
        local_mask = points[:, 0] <= edge_x + fit_window
    else:
        local_mask = points[:, 0] >= edge_x - fit_window

    local_points = points[local_mask]

    if len(local_points) < 8:
        local_points = points

    xs = local_points[:, 0].astype(np.float32)
    ys = local_points[:, 1].astype(np.float32)

    predicted = predict(xs)
    residuals = np.abs(ys - predicted)

    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))

    half_height = int(
        round(
            RESCUE_BAND_EXTRA_PX
            + median_residual
            + RESCUE_BAND_MAD_SCALE * max(mad, 1.0)
        )
    )

    half_height = max(RESCUE_BAND_MIN_HALF_HEIGHT_PX, half_height)
    half_height = min(max_half_height, half_height)

    return int(half_height)


def build_side_roi(binary_top2, current_mask, guidance_points, side):
    binary_top2 = normalize_binary_mask(binary_top2)
    current_mask = normalize_binary_mask(current_mask)

    height, width = binary_top2.shape[:2]
    current_bounds = mask_bounds(current_mask)

    roi_mask = np.zeros_like(binary_top2, dtype=np.uint8)

    if current_bounds is None:
        return roi_mask, None, 0

    predict, edge_x, _, _ = build_endpoint_predictor(
        guidance_points,
        current_mask,
        current_bounds,
        side,
        binary_top2.shape,
    )

    half_height = estimate_band_half_height(
        guidance_points,
        predict,
        edge_x,
        side,
        binary_top2.shape,
    )

    search_width = max(
        RESCUE_SEARCH_WIDTH_PX,
        int(round(RESCUE_SEARCH_WIDTH_FRAC * width)),
    )

    if side == "left":
        x1 = max(0, current_bounds["min_x"] - search_width)
        x2 = max(0, current_bounds["min_x"] - 1)
    else:
        x1 = min(width - 1, current_bounds["max_x"] + 1)
        x2 = min(width - 1, current_bounds["max_x"] + search_width)

    if x2 < x1:
        return roi_mask, predict, half_height

    xs = np.arange(x1, x2 + 1, dtype=np.float32)
    ys = predict(xs)

    for x, center_y in zip(xs.astype(np.int32), ys, strict=False):
        cy = int(round(center_y))

        top = max(0, cy - half_height)
        bottom = min(height, cy + half_height + 1)

        roi_mask[top:bottom, x] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)

    return roi_mask, predict, half_height


def component_stats(component_mask):
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


def is_rescue_candidate_too_high(
    ys_float,
    predicted,
    candidate_edge_y,
    expected_edge_y,
    current_edge_y,
    gap,
):
    if not RESCUE_UPPER_ARTIFACT_GUARD_ENABLE:
        return False

    if gap < RESCUE_UPPER_ARTIFACT_MIN_GAP_PX:
        return False

    signed = ys_float - predicted
    above = np.maximum(0.0, -signed)

    median_signed = float(np.median(signed))
    p90_above = float(np.percentile(above, 90))

    edge_above_model = (
        candidate_edge_y < expected_edge_y - RESCUE_UPPER_ARTIFACT_EDGE_ABOVE_MODEL_PX
    )
    edge_above_current = (
        candidate_edge_y < current_edge_y - RESCUE_UPPER_ARTIFACT_EDGE_ABOVE_CURRENT_PX
    )

    median_above_model = median_signed < -RESCUE_UPPER_ARTIFACT_MEDIAN_ABOVE_MODEL_PX
    p90_above_model = p90_above > RESCUE_UPPER_ARTIFACT_P90_ABOVE_MODEL_PX

    return bool(
        edge_above_model
        and edge_above_current
        and (median_above_model or p90_above_model)
    )


def score_rescue_candidate(
    component_mask, current_mask, predict, half_height, image_shape, side
):
    _, width = image_shape[:2]

    stats = component_stats(component_mask)
    current_bounds = mask_bounds(current_mask)

    if stats is None or current_bounds is None:
        return None

    min_width = max(
        RESCUE_MIN_WIDTH_PX,
        int(round(RESCUE_MIN_WIDTH_FRAC * width)),
    )

    if stats["area"] < RESCUE_MIN_AREA:
        return None

    if stats["width"] < min_width and stats["area"] < 2 * RESCUE_MIN_AREA:
        return None

    if (
        stats["verticality"] > RESCUE_MAX_VERTICALITY
        and stats["width"] < RESCUE_VERTICALITY_WIDTH_FRAC * width
    ):
        return None

    if side == "left":
        gap = current_bounds["min_x"] - stats["max_x"]
        edge_x = stats["max_x"]
        current_edge_y = edge_y_from_mask(current_mask, "left")
        candidate_edge_y = edge_y_from_mask(component_mask, "right")
        extension_gain = current_bounds["min_x"] - stats["min_x"]
    else:
        gap = stats["min_x"] - current_bounds["max_x"]
        edge_x = stats["min_x"]
        current_edge_y = edge_y_from_mask(current_mask, "right")
        candidate_edge_y = edge_y_from_mask(component_mask, "left")
        extension_gain = stats["max_x"] - current_bounds["max_x"]

    if gap < 0:
        return None

    max_gap = max(
        RESCUE_MAX_GAP_PX,
        int(round(RESCUE_MAX_GAP_FRAC * width)),
    )

    if gap > max_gap:
        return None

    min_gain = max(
        RESCUE_MIN_EXTENSION_GAIN_PX,
        int(round(RESCUE_MIN_EXTENSION_GAIN_FRAC * width)),
    )

    if extension_gain < min_gain:
        return None

    ys, xs = np.where(component_mask > 0)

    xs_float = xs.astype(np.float32)
    ys_float = ys.astype(np.float32)

    predicted = predict(xs_float)
    distances = np.abs(ys_float - predicted)

    median_dist = float(np.median(distances))
    p90_dist = float(np.percentile(distances, 90))

    if median_dist > RESCUE_MAX_MEDIAN_DIST_FACTOR * half_height:
        return None

    if p90_dist > RESCUE_MAX_P90_DIST_FACTOR * half_height:
        return None

    expected_edge_y = float(predict(np.array([edge_x], dtype=np.float32))[0])

    if is_rescue_candidate_too_high(
        ys_float=ys_float,
        predicted=predicted,
        candidate_edge_y=candidate_edge_y,
        expected_edge_y=expected_edge_y,
        current_edge_y=current_edge_y,
        gap=gap,
    ):
        return None

    edge_dist_to_model = abs(candidate_edge_y - expected_edge_y)
    edge_dist_to_current = abs(candidate_edge_y - current_edge_y)
    edge_dist = min(edge_dist_to_model, edge_dist_to_current)

    if edge_dist > RESCUE_MAX_EDGE_DIST_FACTOR * half_height:
        return None

    width_score = min(1.0, stats["width"] / max(0.18 * width, 1))
    area_score = min(1.0, stats["area"] / 230.0)
    dist_score = 1.0 - min(1.0, median_dist / max(half_height, 1))
    gap_score = 1.0 - min(1.0, gap / max(max_gap, 1))
    edge_score = 1.0 - min(1.0, edge_dist / max(half_height, 1))
    gain_score = min(1.0, extension_gain / max(0.18 * width, 1))

    score = 0.0
    score += 2.4 * width_score
    score += 1.3 * area_score
    score += 2.0 * dist_score
    score += 1.3 * gap_score
    score += 1.4 * edge_score
    score += 1.6 * gain_score
    score += 0.4 * min(1.0, stats["density"])
    score -= 0.65 * min(1.7, stats["verticality"])

    if score < RESCUE_ACCEPT_MIN_SCORE:
        return None

    return float(score)


def find_rescue_for_side(binary_top2, current_mask, guidance_points, side):
    binary_top2 = normalize_binary_mask(binary_top2)
    current_mask = normalize_binary_mask(current_mask)

    roi_mask, predict, half_height = build_side_roi(
        binary_top2,
        current_mask,
        guidance_points,
        side,
    )

    if predict is None:
        empty = np.zeros_like(binary_top2, dtype=np.uint8)
        return [], empty, roi_mask, empty, empty

    not_current = cv2.bitwise_not(current_mask)

    candidate_mask = cv2.bitwise_and(binary_top2, roi_mask)
    candidate_mask = cv2.bitwise_and(candidate_mask, not_current)

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        candidate_mask,
        connectivity=8,
    )

    accepted = []
    rejected_mask = np.zeros_like(binary_top2, dtype=np.uint8)

    for label in range(1, num_labels):
        component_mask = np.zeros_like(binary_top2, dtype=np.uint8)
        component_mask[labels == label] = 255

        score = score_rescue_candidate(
            component_mask,
            current_mask,
            predict,
            half_height,
            binary_top2.shape,
            side,
        )

        if score is None:
            rejected_mask[component_mask > 0] = 255
            continue

        accepted.append((score, component_mask))

    accepted.sort(key=lambda item: item[0], reverse=True)
    accepted = accepted[:RESCUE_MAX_COMPONENTS_PER_SIDE]

    accepted_mask = np.zeros_like(binary_top2, dtype=np.uint8)

    for _, component_mask in accepted:
        accepted_mask[component_mask > 0] = 255

    return accepted, accepted_mask, roi_mask, candidate_mask, rejected_mask


def should_try_lateral_rescue(merged_mask):
    bounds = mask_bounds(merged_mask)

    if bounds is None:
        return False

    _, width = merged_mask.shape[:2]
    width_frac = bounds["width"] / max(width, 1)
    left_free = bounds["min_x"] / max(width, 1)
    right_free = (width - 1 - bounds["max_x"]) / max(width, 1)

    if width_frac < RESCUE_TRIGGER_MIN_WIDTH_FRAC:
        return True

    if left_free > RESCUE_TRIGGER_MIN_SIDE_FREE_FRAC:
        return True

    if right_free > RESCUE_TRIGGER_MIN_SIDE_FREE_FRAC:
        return True

    return False


def guard_overgrown_secondary(principal_mask, secondary_mask):
    principal_mask = normalize_binary_mask(principal_mask)
    secondary_mask = normalize_binary_mask(secondary_mask)

    if not GUARD_ENABLE:
        empty = np.zeros_like(principal_mask, dtype=np.uint8)
        return secondary_mask, empty, False

    principal_bounds = mask_bounds(principal_mask)
    secondary_bounds = mask_bounds(secondary_mask)

    if principal_bounds is None or secondary_bounds is None:
        empty = np.zeros_like(principal_mask, dtype=np.uint8)
        return secondary_mask, empty, False

    principal_area = max(principal_bounds["area"], 1)
    principal_width = max(principal_bounds["width"], 1)

    area_ratio = secondary_bounds["area"] / principal_area
    width_ratio = secondary_bounds["width"] / principal_width

    if (
        area_ratio <= GUARD_MAX_SECONDARY_TO_PRINCIPAL_AREA
        and width_ratio <= GUARD_MAX_SECONDARY_TO_PRINCIPAL_WIDTH
    ):
        empty = np.zeros_like(principal_mask, dtype=np.uint8)
        return secondary_mask, empty, False

    removed_mask = secondary_mask.copy()
    filtered_secondary = np.zeros_like(secondary_mask, dtype=np.uint8)

    return filtered_secondary, removed_mask, True


def rescue_after_secondary(
    binary_top2, principal_mask, secondary_mask, traveler_points
):
    binary_top2 = normalize_binary_mask(binary_top2)
    principal_mask = normalize_binary_mask(principal_mask)
    secondary_mask = normalize_binary_mask(secondary_mask)

    secondary_mask, removed_secondary_mask, guard_triggered = guard_overgrown_secondary(
        principal_mask,
        secondary_mask,
    )

    current_mask = np.zeros_like(principal_mask, dtype=np.uint8)
    current_mask[principal_mask > 0] = 255
    current_mask[secondary_mask > 0] = 255

    rescue_mask = np.zeros_like(principal_mask, dtype=np.uint8)
    all_roi_mask = np.zeros_like(principal_mask, dtype=np.uint8)
    all_candidate_mask = np.zeros_like(principal_mask, dtype=np.uint8)
    all_rejected_mask = np.zeros_like(principal_mask, dtype=np.uint8)
    all_accepted_mask = np.zeros_like(principal_mask, dtype=np.uint8)

    added = []

    guidance_points = build_guidance_points(
        principal_mask,
        traveler_points,
    )

    if not guard_triggered and should_try_lateral_rescue(current_mask):
        for iteration in range(RESCUE_MAX_ITERATIONS):
            added_this_round = False

            for side in ["left", "right"]:
                accepted, accepted_mask, roi_mask, candidate_mask, rejected_mask = (
                    find_rescue_for_side(
                        binary_top2,
                        current_mask,
                        guidance_points,
                        side,
                    )
                )

                all_roi_mask[roi_mask > 0] = 255
                all_candidate_mask[candidate_mask > 0] = 255
                all_rejected_mask[rejected_mask > 0] = 255
                all_accepted_mask[accepted_mask > 0] = 255

                if len(accepted) == 0:
                    continue

                for score, component_mask in accepted:
                    current_mask[component_mask > 0] = 255
                    rescue_mask[component_mask > 0] = 255
                    added.append(
                        {
                            "iteration": iteration + 1,
                            "side": side,
                            "score": score,
                        }
                    )

                added_this_round = True

            if not added_this_round:
                break

    merged_after_rescue = np.zeros_like(principal_mask, dtype=np.uint8)
    merged_after_rescue[principal_mask > 0] = 255
    merged_after_rescue[secondary_mask > 0] = 255
    merged_after_rescue[rescue_mask > 0] = 255

    changed = bool(guard_triggered or len(added) > 0)

    return {
        "secondary_mask": secondary_mask,
        "rescue_mask": rescue_mask,
        "merged_mask": merged_after_rescue,
        "removed_secondary_mask": removed_secondary_mask,
        "roi_mask": all_roi_mask,
        "candidate_mask": all_candidate_mask,
        "rejected_mask": all_rejected_mask,
        "accepted_mask": all_accepted_mask,
        "changed": changed,
        "guard_triggered": guard_triggered,
        "added_components": added,
    }


def draw_rescue_debug(
    base_image,
    principal_mask,
    secondary_mask,
    rescue_mask,
    removed_secondary_mask=None,
    roi_mask=None,
    candidate_mask=None,
    rejected_mask=None,
    traveler_points=None,
):
    result = to_bgr(base_image)

    if roi_mask is not None:
        result = draw_mask_overlay(result, roi_mask, ROI_COLOR, alpha=0.22)

    if candidate_mask is not None:
        result = draw_mask_overlay(result, candidate_mask, CANDIDATE_COLOR, alpha=0.38)

    if rejected_mask is not None:
        result = draw_mask_overlay(result, rejected_mask, REJECTED_COLOR, alpha=0.32)

    result = draw_mask_overlay(result, principal_mask, PRINCIPAL_COLOR, alpha=0.65)
    result = draw_mask_overlay(result, secondary_mask, SECONDARY_COLOR, alpha=0.75)
    result = draw_mask_overlay(result, rescue_mask, RESCUE_COLOR, alpha=0.90)

    if removed_secondary_mask is not None:
        result = draw_mask_overlay(
            result, removed_secondary_mask, REJECTED_COLOR, alpha=0.40
        )

    if traveler_points is not None:
        result = draw_points(result, traveler_points, TRAVELER_COLOR, radius=1)

    return result


def draw_merged_after_rescue(base_image, merged_mask, traveler_points=None):
    result = draw_mask_overlay(base_image, merged_mask, MERGED_COLOR, alpha=0.75)

    if traveler_points is not None:
        result = draw_points(result, traveler_points, TRAVELER_COLOR, radius=1)

    return result


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

    if secondary_before_area == 0:
        max_rescue_area = max(600, int(round(1.20 * principal_area)))
        return 0 < rescue_area <= max_rescue_area

    secondary_before_bounds = get_mask_bounds(secondary_mask_before_rescue)
    secondary_before_density = mask_density(secondary_mask_before_rescue)

    overgrown_secondary = (
        secondary_before_area >= SECONDARY_RESCUE_OVERGROWTH_MIN_AREA
        and secondary_before_area
        > SECONDARY_RESCUE_OVERGROWTH_AREA_RATIO * principal_area
        and secondary_before_bounds is not None
        and (
            secondary_before_density >= SECONDARY_RESCUE_OVERGROWTH_MIN_DENSITY
            or secondary_before_bounds["height"]
            >= SECONDARY_RESCUE_OVERGROWTH_MIN_HEIGHT
        )
    )

    if overgrown_secondary:
        return True

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
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        component_median_y = float(centroids[label][1])
        component_pixels = labels == label

        right_gap = x - base_bounds["max_x"] - 1

        is_right_isolated = right_gap >= RIGHT_ISOLATED_SECONDARY_RESCUE_MIN_GAP_PX

        is_small_enough = (
            area <= RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_AREA
            and width <= RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_WIDTH
            and height <= RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_HEIGHT
        )

        is_much_lower = (
            component_median_y
            >= base_median_y + RIGHT_ISOLATED_SECONDARY_RESCUE_MIN_BELOW_MEDIAN_PX
        )

        if is_right_isolated and is_small_enough and is_much_lower:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    return kept, removed
