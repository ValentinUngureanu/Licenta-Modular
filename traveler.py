from dataclasses import dataclass

import cv2
import numpy as np

RAW_POINT_COLOR = (255, 180, 0)
SELECTED_COLOR = (0, 255, 0)
CLEANED_COLOR = (0, 255, 0)
REMOVED_COLOR = (0, 0, 255)
EXTENDED_COLOR = (0, 255, 255)
GRAY_COLOR = (120, 120, 120)

TRAVELER_MAX_X_GAP = 7
TRAVELER_MAX_Y_JUMP = 22

MIN_COMPONENT_POINTS = 8
MIN_COMPONENT_WIDTH_FRAC = 0.025

CLEAN_LOCAL_WINDOW_FRAC = 0.035
CLEAN_MIN_BAND_PX = 9
CLEAN_MAX_BAND_FRAC = 0.065
CLEAN_MAD_SCALE = 3.0
CLEAN_MIN_KEEP_FRAC = 0.45

EXTEND_MAX_ITERATIONS = 8
EXTEND_MAX_EDGE_GAP_FRAC = 0.08
EXTEND_MAX_EDGE_GAP_PX = 45
EXTEND_MAX_EDGE_Y_DIST_FRAC = 0.075
EXTEND_MAX_EDGE_Y_DIST_PX = 28
EXTEND_MIN_CAND_WIDTH_FRAC = 0.018
EXTEND_MIN_CAND_POINTS = 5
EXTEND_MAX_CAND_VERTICALITY = 0.95
EXTEND_MAX_SLOPE_DELTA = 0.85
EXTEND_EDGE_WINDOW_PX = 8

MODEL_DEGREE_2_MIN_WIDTH_FRAC = 0.22
MODEL_DEGREE_2_MIN_POINTS = 45


@dataclass
class TravelerComponent:
    component_id: int
    points: np.ndarray


@dataclass
class TravelerStats:
    n: int
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    width: int
    height: int
    density: float
    median_y: float
    slope: float
    verticality: float


def normalize_binary_mask(binary_mask):
    if binary_mask.ndim == 3:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

    return (binary_mask > 0).astype(np.uint8) * 255


def extract_raw_traveler_points(binary_mask):
    binary = normalize_binary_mask(binary_mask)
    _, width = binary.shape[:2]

    points = []

    for x in range(width):
        ys = np.flatnonzero(binary[:, x] > 0)

        if len(ys) == 0:
            continue

        y = int(ys[-1])
        points.append((x, y))

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32)

    return np.array(points, dtype=np.int32)


def split_traveler_components(points):
    if points is None or len(points) == 0:
        return []

    points = np.asarray(points, dtype=np.int32)
    order = np.argsort(points[:, 0])
    points = points[order]

    components = []
    current = [points[0]]
    component_id = 0

    for point in points[1:]:
        previous = current[-1]

        x_gap = int(point[0] - previous[0])
        y_jump = int(abs(point[1] - previous[1]))

        if x_gap > TRAVELER_MAX_X_GAP or y_jump > TRAVELER_MAX_Y_JUMP:
            components.append(
                TravelerComponent(
                    component_id=component_id,
                    points=np.array(current, dtype=np.int32),
                )
            )
            component_id += 1
            current = [point]
        else:
            current.append(point)

    components.append(
        TravelerComponent(
            component_id=component_id,
            points=np.array(current, dtype=np.int32),
        )
    )

    return components


def get_component_stats(points):
    if points is None or len(points) == 0:
        return TravelerStats(
            n=0,
            min_x=0,
            max_x=0,
            min_y=0,
            max_y=0,
            width=0,
            height=0,
            density=0.0,
            median_y=0.0,
            slope=0.0,
            verticality=0.0,
        )

    points = np.asarray(points, dtype=np.int32)
    xs = points[:, 0].astype(np.float32)
    ys = points[:, 1].astype(np.float32)

    min_x = int(np.min(xs))
    max_x = int(np.max(xs))
    min_y = int(np.min(ys))
    max_y = int(np.max(ys))

    width = max(1, max_x - min_x + 1)
    height = max(1, max_y - min_y + 1)
    n = int(len(points))
    density = float(n / width)
    median_y = float(np.median(ys))

    if n >= 2 and width >= 2:
        try:
            slope = float(np.polyfit(xs, ys, 1)[0])
        except Exception:
            slope = 0.0
    else:
        slope = 0.0

    verticality = float(height / max(width, 1))

    return TravelerStats(
        n=n,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        width=width,
        height=height,
        density=density,
        median_y=median_y,
        slope=slope,
        verticality=verticality,
    )


def score_traveler_component(component, image_shape):
    height, width = image_shape[:2]
    stats = get_component_stats(component.points)

    if stats.n == 0:
        return -9999.0

    width_frac = stats.width / max(width, 1)
    height_frac = stats.height / max(height, 1)
    median_y_frac = stats.median_y / max(height, 1)

    score = 0.0

    score += 4.5 * min(1.0, width_frac / 0.45)
    score += 1.8 * min(1.0, stats.density)
    score += 1.2 * min(1.0, stats.n / 120.0)

    if 0.18 <= median_y_frac <= 0.82:
        score += 0.8
    else:
        score -= 1.0

    score -= 2.5 * min(2.0, stats.verticality)
    score -= 2.0 * max(0.0, height_frac - 0.18)
    score -= 1.2 * min(1.5, abs(stats.slope))

    min_width = max(8, int(MIN_COMPONENT_WIDTH_FRAC * width))

    if stats.width < min_width:
        score -= 2.5

    if stats.n < MIN_COMPONENT_POINTS:
        score -= 2.0

    if stats.width < 12 and stats.height > stats.width:
        score -= 3.0

    return float(score)


def select_main_traveler_component(components, image_shape):
    if len(components) == 0:
        return None, []

    scored = []

    for component in components:
        score = score_traveler_component(component, image_shape)
        scored.append((score, component))

    scored.sort(key=lambda item: item[0], reverse=True)

    return scored[0][1], scored


def clean_component_core(points, image_shape):
    if points is None or len(points) == 0:
        empty = np.empty((0, 2), dtype=np.int32)
        return empty, empty

    points = np.asarray(points, dtype=np.int32)

    if len(points) < 12:
        return points.copy(), np.empty((0, 2), dtype=np.int32)

    height, width = image_shape[:2]

    order = np.argsort(points[:, 0])
    ordered = points[order]
    xs = ordered[:, 0]
    ys = ordered[:, 1].astype(np.float32)

    local_window = max(10, int(round(CLEAN_LOCAL_WINDOW_FRAC * width)))
    local_medians = np.zeros(len(ordered), dtype=np.float32)

    for i, x in enumerate(xs):
        mask = np.abs(xs - x) <= local_window
        local_medians[i] = float(np.median(ys[mask]))

    residuals = np.abs(ys - local_medians)
    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))

    max_band = max(
        CLEAN_MIN_BAND_PX,
        int(round(CLEAN_MAX_BAND_FRAC * height)),
    )

    allowed_band = max(
        CLEAN_MIN_BAND_PX,
        median_residual + CLEAN_MAD_SCALE * max(mad, 1.0),
    )

    allowed_band = min(max_band, allowed_band)

    keep_mask = residuals <= allowed_band

    min_keep = max(8, int(round(CLEAN_MIN_KEEP_FRAC * len(ordered))))

    if int(np.count_nonzero(keep_mask)) < min_keep:
        return ordered.copy(), np.empty((0, 2), dtype=np.int32)

    cleaned = ordered[keep_mask].copy()
    removed = ordered[~keep_mask].copy()

    return cleaned, removed


def fit_traveler_model(points, image_shape):
    _, width = image_shape[:2]

    if points is None or len(points) == 0:

        def predict_empty(x_values):
            x_values = np.asarray(x_values, dtype=np.float32)
            return np.zeros_like(x_values, dtype=np.float32)

        return predict_empty, 0

    points = np.asarray(points, dtype=np.int32)
    xs = points[:, 0].astype(np.float32)
    ys = points[:, 1].astype(np.float32)

    stats = get_component_stats(points)

    if (
        len(points) >= MODEL_DEGREE_2_MIN_POINTS
        and stats.width >= MODEL_DEGREE_2_MIN_WIDTH_FRAC * width
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


def get_edge_y(points, side):
    if points is None or len(points) == 0:
        return 0.0

    points = np.asarray(points, dtype=np.int32)

    if side == "left":
        edge_x = int(np.min(points[:, 0]))
        mask = points[:, 0] <= edge_x + EXTEND_EDGE_WINDOW_PX
    else:
        edge_x = int(np.max(points[:, 0]))
        mask = points[:, 0] >= edge_x - EXTEND_EDGE_WINDOW_PX

    if np.count_nonzero(mask) == 0:
        return float(np.median(points[:, 1]))

    return float(np.median(points[mask, 1]))


def evaluate_extension_candidate(candidate_points, current_points, image_shape, side):
    height, width = image_shape[:2]

    candidate_cleaned, _ = clean_component_core(candidate_points, image_shape)

    if len(candidate_cleaned) == 0:
        candidate_cleaned = np.asarray(candidate_points, dtype=np.int32)

    cand_stats = get_component_stats(candidate_cleaned)
    cur_stats = get_component_stats(current_points)

    if cand_stats.n < EXTEND_MIN_CAND_POINTS:
        return None

    min_cand_width = max(6, int(round(EXTEND_MIN_CAND_WIDTH_FRAC * width)))

    if cand_stats.width < min_cand_width:
        return None

    if (
        cand_stats.verticality > EXTEND_MAX_CAND_VERTICALITY
        and cand_stats.width < 0.12 * width
    ):
        return None

    if side == "left":
        gap = cur_stats.min_x - cand_stats.max_x
        edge_x = cand_stats.max_x
        current_edge_y = get_edge_y(current_points, "left")
        candidate_edge_y = get_edge_y(candidate_cleaned, "right")
    else:
        gap = cand_stats.min_x - cur_stats.max_x
        edge_x = cand_stats.min_x
        current_edge_y = get_edge_y(current_points, "right")
        candidate_edge_y = get_edge_y(candidate_cleaned, "left")

    if gap < 0:
        return None

    max_gap = max(
        EXTEND_MAX_EDGE_GAP_PX,
        int(round(EXTEND_MAX_EDGE_GAP_FRAC * width)),
    )

    if gap > max_gap:
        return None

    predict, _ = fit_traveler_model(current_points, image_shape)
    expected_y = float(predict(np.array([edge_x], dtype=np.float32))[0])

    y_dist_to_model = abs(candidate_edge_y - expected_y)
    y_dist_to_current_edge = abs(candidate_edge_y - current_edge_y)

    max_y_dist = max(
        EXTEND_MAX_EDGE_Y_DIST_PX,
        int(round(EXTEND_MAX_EDGE_Y_DIST_FRAC * height)),
    )

    if min(y_dist_to_model, y_dist_to_current_edge) > max_y_dist:
        return None

    slope_delta = abs(cand_stats.slope - cur_stats.slope)

    if slope_delta > EXTEND_MAX_SLOPE_DELTA and cand_stats.width < 0.15 * width:
        return None

    gap_score = 1.0 - min(1.0, gap / max(max_gap, 1))
    y_score = 1.0 - min(
        1.0,
        min(y_dist_to_model, y_dist_to_current_edge) / max(max_y_dist, 1),
    )
    width_score = min(1.0, cand_stats.width / max(0.18 * width, 1))
    density_score = min(1.0, cand_stats.density)

    score = 0.0
    score += 2.0 * width_score
    score += 1.6 * y_score
    score += 1.2 * gap_score
    score += 0.9 * density_score
    score -= 1.2 * min(1.5, cand_stats.verticality)
    score -= 0.5 * min(1.5, slope_delta)

    return score, candidate_cleaned


def extend_component_left_right(
    selected_points,
    components,
    image_shape,
    selected_component_id=None,
):
    if selected_points is None or len(selected_points) == 0:
        return np.empty((0, 2), dtype=np.int32), []

    current = np.asarray(selected_points, dtype=np.int32).copy()
    used_ids = set()

    if selected_component_id is not None:
        used_ids.add(selected_component_id)

    added_ids = []

    for _ in range(EXTEND_MAX_ITERATIONS):
        added_this_round = False

        for side in ["left", "right"]:
            best = None

            for component in components:
                if component.component_id in used_ids:
                    continue

                candidate_result = evaluate_extension_candidate(
                    component.points,
                    current,
                    image_shape,
                    side,
                )

                if candidate_result is None:
                    continue

                score, candidate_cleaned = candidate_result

                if best is None or score > best[0]:
                    best = (score, component.component_id, candidate_cleaned)

            if best is None:
                continue

            _, component_id, candidate_cleaned = best

            current = np.vstack([current, candidate_cleaned]).astype(np.int32)
            current = current[np.argsort(current[:, 0])]

            cleaned_current, _ = clean_component_core(current, image_shape)

            if len(cleaned_current) > 0:
                current = cleaned_current

            used_ids.add(component_id)
            added_ids.append(component_id)
            added_this_round = True

        if not added_this_round:
            break

    current = current[np.argsort(current[:, 0])]

    return current.astype(np.int32), added_ids


def build_traveler(binary_mask):
    binary = normalize_binary_mask(binary_mask)
    image_shape = binary.shape[:2]

    raw_points = extract_raw_traveler_points(binary)
    components = split_traveler_components(raw_points)
    selected_component, scored_components = select_main_traveler_component(
        components,
        image_shape,
    )

    if selected_component is None:
        empty = np.empty((0, 2), dtype=np.int32)

        return {
            "raw_points": raw_points,
            "components": components,
            "scored_components": scored_components,
            "selected_component": None,
            "selected_points": empty,
            "cleaned_points": empty,
            "removed_points": empty,
            "extended_points": empty,
            "added_component_ids": [],
        }

    cleaned_points, removed_points = clean_component_core(
        selected_component.points,
        image_shape,
    )

    extended_points, added_component_ids = extend_component_left_right(
        cleaned_points,
        components,
        image_shape,
        selected_component_id=selected_component.component_id,
    )

    return {
        "raw_points": raw_points,
        "components": components,
        "scored_components": scored_components,
        "selected_component": selected_component,
        "selected_points": selected_component.points,
        "cleaned_points": cleaned_points,
        "removed_points": removed_points,
        "extended_points": extended_points,
        "added_component_ids": added_component_ids,
    }


def to_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image.copy()


def draw_points(image, points, color, radius=2):
    result = to_bgr(image)

    if points is None or len(points) == 0:
        return result

    for x, y in np.asarray(points, dtype=np.int32):
        cv2.circle(result, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

    return result


def draw_raw_traveler_points(base_image, raw_points):
    return draw_points(base_image, raw_points, RAW_POINT_COLOR, radius=1)


def component_color(component_id):
    colors = [
        (0, 255, 255),
        (0, 180, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
        (255, 255, 0),
        (80, 200, 255),
        (255, 120, 80),
        (120, 255, 80),
        (200, 80, 255),
    ]

    return colors[component_id % len(colors)]


def draw_traveler_components_colored(base_image, components):
    result = to_bgr(base_image)

    for component in components:
        color = component_color(component.component_id)
        result = draw_points(result, component.points, color, radius=2)

    return result


def draw_selected_component(base_image, raw_points, selected_points):
    result = draw_points(base_image, raw_points, GRAY_COLOR, radius=1)
    result = draw_points(result, selected_points, SELECTED_COLOR, radius=2)

    return result


def draw_clean_selected_component(
    base_image,
    selected_points,
    cleaned_points,
    removed_points,
):
    result = draw_points(base_image, selected_points, GRAY_COLOR, radius=1)
    result = draw_points(result, removed_points, REMOVED_COLOR, radius=2)
    result = draw_points(result, cleaned_points, CLEANED_COLOR, radius=2)

    return result


def draw_extended_component(
    base_image,
    raw_points,
    extended_points,
    components,
    added_component_ids,
):
    result = draw_points(base_image, raw_points, GRAY_COLOR, radius=1)

    for component in components:
        if component.component_id in added_component_ids:
            result = draw_points(result, component.points, EXTENDED_COLOR, radius=2)

    result = draw_points(result, extended_points, SELECTED_COLOR, radius=2)

    return result
