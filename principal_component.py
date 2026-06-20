import cv2
import numpy as np


ROI_COLOR = (255, 0, 0)
CANDIDATE_COLOR = (0, 255, 255)
PRINCIPAL_COLOR = (0, 255, 0)
REJECTED_COLOR = (0, 0, 255)
TRAVELER_COLOR = (255, 180, 0)

PRINCIPAL_X_PADDING_FRAC = 0.045
PRINCIPAL_X_PADDING_PX = 35

PRINCIPAL_BAND_MIN_HALF_HEIGHT_PX = 18
PRINCIPAL_BAND_MAX_HALF_HEIGHT_FRAC = 0.095
PRINCIPAL_BAND_EXTRA_PX = 10
PRINCIPAL_BAND_MAD_SCALE = 3.0

PRINCIPAL_POLY_DEGREE_2_MIN_WIDTH_FRAC = 0.22
PRINCIPAL_POLY_DEGREE_2_MIN_POINTS = 45

PRINCIPAL_MIN_AREA = 10
PRINCIPAL_MIN_WIDTH_FRAC = 0.010
PRINCIPAL_MIN_WIDTH_PX = 6

PRINCIPAL_MAX_MEDIAN_DIST_FACTOR = 0.92
PRINCIPAL_MAX_P90_DIST_FACTOR = 1.25

PRINCIPAL_MAX_VERTICALITY = 1.45
PRINCIPAL_VERTICALITY_WIDTH_FRAC = 0.10

PRINCIPAL_KEEP_TOP_COMPONENTS = 12


def normalize_binary_mask(binary_mask):
    if binary_mask.ndim == 3:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

    return (binary_mask > 0).astype(np.uint8) * 255


def to_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image.copy()


def fit_principal_model(traveler_points, image_shape):
    _, width = image_shape[:2]

    if traveler_points is None or len(traveler_points) == 0:
        def predict_empty(x_values):
            x_values = np.asarray(x_values, dtype=np.float32)
            return np.zeros_like(x_values, dtype=np.float32)

        return predict_empty, 0

    points = np.asarray(traveler_points, dtype=np.int32)
    xs = points[:, 0].astype(np.float32)
    ys = points[:, 1].astype(np.float32)

    min_x = int(np.min(xs))
    max_x = int(np.max(xs))
    traveler_width = max(1, max_x - min_x + 1)

    if (
        len(points) >= PRINCIPAL_POLY_DEGREE_2_MIN_POINTS
        and traveler_width >= PRINCIPAL_POLY_DEGREE_2_MIN_WIDTH_FRAC * width
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


def estimate_band_half_height(traveler_points, predict, image_shape):
    height, _ = image_shape[:2]

    max_half_height = max(
        PRINCIPAL_BAND_MIN_HALF_HEIGHT_PX,
        int(round(PRINCIPAL_BAND_MAX_HALF_HEIGHT_FRAC * height)),
    )

    if traveler_points is None or len(traveler_points) < 8:
        return max(
            PRINCIPAL_BAND_MIN_HALF_HEIGHT_PX,
            min(max_half_height, int(round(0.055 * height))),
        )

    points = np.asarray(traveler_points, dtype=np.int32)
    xs = points[:, 0].astype(np.float32)
    ys = points[:, 1].astype(np.float32)

    predicted = predict(xs)
    residuals = np.abs(ys - predicted)

    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))

    half_height = int(
        round(
            PRINCIPAL_BAND_EXTRA_PX
            + median_residual
            + PRINCIPAL_BAND_MAD_SCALE * max(mad, 1.0)
        )
    )

    half_height = max(PRINCIPAL_BAND_MIN_HALF_HEIGHT_PX, half_height)
    half_height = min(max_half_height, half_height)

    return int(half_height)


def build_principal_roi_mask(binary_mask, traveler_points):
    binary = normalize_binary_mask(binary_mask)
    height, width = binary.shape[:2]

    roi_mask = np.zeros_like(binary, dtype=np.uint8)

    if traveler_points is None or len(traveler_points) == 0:
        return roi_mask, None, 0, 0, 0

    points = np.asarray(traveler_points, dtype=np.int32)

    predict, degree = fit_principal_model(points, binary.shape)
    half_height = estimate_band_half_height(points, predict, binary.shape)

    min_x = int(np.min(points[:, 0]))
    max_x = int(np.max(points[:, 0]))

    x_padding = max(
        PRINCIPAL_X_PADDING_PX,
        int(round(PRINCIPAL_X_PADDING_FRAC * width)),
    )

    x1 = max(0, min_x - x_padding)
    x2 = min(width - 1, max_x + x_padding)

    xs = np.arange(x1, x2 + 1, dtype=np.float32)
    ys = predict(xs)

    for x, center_y in zip(xs.astype(np.int32), ys):
        cy = int(round(center_y))
        top = max(0, cy - half_height)
        bottom = min(height, cy + half_height + 1)
        roi_mask[top:bottom, x] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)

    return roi_mask, predict, degree, half_height, x_padding


def component_stats_from_mask(component_mask):
    ys, xs = np.where(component_mask > 0)

    if len(xs) == 0:
        return {
            "area": 0,
            "min_x": 0,
            "max_x": 0,
            "min_y": 0,
            "max_y": 0,
            "width": 0,
            "height": 0,
            "verticality": 0.0,
            "median_y": 0.0,
        }

    min_x = int(np.min(xs))
    max_x = int(np.max(xs))
    min_y = int(np.min(ys))
    max_y = int(np.max(ys))

    width = max(1, max_x - min_x + 1)
    height = max(1, max_y - min_y + 1)
    area = int(len(xs))
    verticality = float(height / max(width, 1))
    median_y = float(np.median(ys))

    return {
        "area": area,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "width": width,
        "height": height,
        "verticality": verticality,
        "median_y": median_y,
    }


def score_principal_fragment(component_mask, predict, half_height, image_shape):
    _, width = image_shape[:2]

    stats = component_stats_from_mask(component_mask)

    if stats["area"] <= 0:
        return None

    min_width = max(
        PRINCIPAL_MIN_WIDTH_PX,
        int(round(PRINCIPAL_MIN_WIDTH_FRAC * width)),
    )

    if stats["area"] < PRINCIPAL_MIN_AREA:
        return None

    if stats["width"] < min_width and stats["area"] < 2 * PRINCIPAL_MIN_AREA:
        return None

    if (
        stats["verticality"] > PRINCIPAL_MAX_VERTICALITY
        and stats["width"] < PRINCIPAL_VERTICALITY_WIDTH_FRAC * width
    ):
        return None

    ys, xs = np.where(component_mask > 0)
    xs_float = xs.astype(np.float32)
    ys_float = ys.astype(np.float32)

    predicted = predict(xs_float)
    distances = np.abs(ys_float - predicted)

    median_dist = float(np.median(distances))
    p90_dist = float(np.percentile(distances, 90))

    if median_dist > PRINCIPAL_MAX_MEDIAN_DIST_FACTOR * half_height:
        return None

    if p90_dist > PRINCIPAL_MAX_P90_DIST_FACTOR * half_height:
        return None

    width_frac = stats["width"] / max(width, 1)
    area_score = min(1.0, stats["area"] / 350.0)
    width_score = min(1.0, width_frac / 0.22)
    dist_score = 1.0 - min(1.0, median_dist / max(half_height, 1))
    vertical_penalty = min(1.5, stats["verticality"])

    score = 0.0
    score += 2.4 * width_score
    score += 1.7 * area_score
    score += 2.0 * dist_score
    score -= 0.65 * vertical_penalty

    return float(score)


def extract_principal_fragments(binary_mask, roi_mask, predict, half_height):
    binary = normalize_binary_mask(binary_mask)
    candidate_mask = cv2.bitwise_and(binary, roi_mask)

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        candidate_mask,
        connectivity=8,
    )

    scored_components = []

    for label in range(1, num_labels):
        component_mask = np.zeros_like(binary, dtype=np.uint8)
        component_mask[labels == label] = 255

        score = score_principal_fragment(
            component_mask,
            predict,
            half_height,
            binary.shape,
        )

        if score is None:
            continue

        scored_components.append((score, label, component_mask))

    scored_components.sort(key=lambda item: item[0], reverse=True)

    principal_mask = np.zeros_like(binary, dtype=np.uint8)
    kept = scored_components[:PRINCIPAL_KEEP_TOP_COMPONENTS]

    for _, _, component_mask in kept:
        principal_mask[component_mask > 0] = 255

    rejected_mask = np.zeros_like(binary, dtype=np.uint8)
    kept_labels = {label for _, label, _ in kept}

    for label in range(1, num_labels):
        if label in kept_labels:
            continue

        rejected_mask[labels == label] = 255

    return {
        "candidate_mask": candidate_mask,
        "principal_mask": principal_mask,
        "rejected_mask": rejected_mask,
        "scored_components": scored_components,
    }


def build_principal_component(binary_mask, traveler_points):
    binary = normalize_binary_mask(binary_mask)

    roi_mask, predict, degree, half_height, x_padding = build_principal_roi_mask(
        binary,
        traveler_points,
    )

    if predict is None:
        empty = np.zeros_like(binary, dtype=np.uint8)

        return {
            "roi_mask": empty,
            "candidate_mask": empty,
            "principal_mask": empty,
            "rejected_mask": empty,
            "degree": degree,
            "half_height": half_height,
            "x_padding": x_padding,
            "scored_components": [],
        }

    fragments_result = extract_principal_fragments(
        binary,
        roi_mask,
        predict,
        half_height,
    )

    return {
        "roi_mask": roi_mask,
        "candidate_mask": fragments_result["candidate_mask"],
        "principal_mask": fragments_result["principal_mask"],
        "rejected_mask": fragments_result["rejected_mask"],
        "degree": degree,
        "half_height": half_height,
        "x_padding": x_padding,
        "scored_components": fragments_result["scored_components"],
    }


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


def draw_principal_roi(base_image, roi_mask, traveler_points=None):
    result = draw_mask_overlay(base_image, roi_mask, ROI_COLOR, alpha=0.32)

    if traveler_points is not None:
        result = draw_points(result, traveler_points, TRAVELER_COLOR, radius=2)

    return result


def draw_candidate_mask(base_image, candidate_mask, traveler_points=None):
    result = draw_mask_overlay(base_image, candidate_mask, CANDIDATE_COLOR, alpha=0.65)

    if traveler_points is not None:
        result = draw_points(result, traveler_points, TRAVELER_COLOR, radius=1)

    return result


def draw_principal_component(
    base_image,
    principal_mask,
    rejected_mask=None,
    traveler_points=None,
):
    result = to_bgr(base_image)

    if rejected_mask is not None:
        result = draw_mask_overlay(result, rejected_mask, REJECTED_COLOR, alpha=0.45)

    result = draw_mask_overlay(result, principal_mask, PRINCIPAL_COLOR, alpha=0.75)

    if traveler_points is not None:
        result = draw_points(result, traveler_points, TRAVELER_COLOR, radius=1)

    return result
