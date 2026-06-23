from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

MIN_COMPONENT_AREA_PX = 1
BOUNDARY_DILATE_KERNEL = 3
MAX_BOUNDARY_POINTS_FOR_DISTANCE = 1000
BRIDGE_THICKNESS_PX = 3
BRIDGE_ENDPOINT_OVERLAP_PX = 2

# RESTART 5: upper-envelope pentru zone fragmentate.
# - Pentru gap-uri simple: pastram bridge-ul cu grosime locala + convex hull local.
# - Pentru zone fragmentate: dezactivam hull-ul si folosim doar o banda controlata
#   cu grosime locala, ca sa nu umplem agresiv intre multe fragmente mici.
BRIDGE_DYNAMIC_THICKNESS_ENABLE = True
BRIDGE_LOCAL_THICKNESS_RADIUS_X = 18
BRIDGE_LOCAL_THICKNESS_MAX_VERTICAL_DISTANCE = 14
BRIDGE_LOCAL_THICKNESS_MIN_PX = 2
BRIDGE_LOCAL_THICKNESS_MAX_PX = 18
BRIDGE_LOCAL_THICKNESS_SCALE = 1.0

BRIDGE_LOCAL_CONVEX_HULL_ENABLE = True
BRIDGE_HULL_LOCAL_RADIUS_X = 22
BRIDGE_HULL_LOCAL_RADIUS_Y = 18
BRIDGE_HULL_MIN_POINTS = 6

# Detector pentru zone fragmentate. In aceste zone hull-ul local produce pete
# prea groase, deci folosim doar linie ingrosata controlat.
FRAGMENTED_MODE_ENABLE = True
FRAGMENTED_GLOBAL_COMPONENT_COUNT_MIN = 8
FRAGMENTED_LOCAL_COMPONENT_COUNT_MIN = 3
FRAGMENTED_LOCAL_SMALL_COMPONENT_COUNT_MIN = 1
FRAGMENTED_LOCAL_RADIUS_X = 36
FRAGMENTED_LOCAL_RADIUS_Y = 24
FRAGMENTED_SMALL_COMPONENT_AREA_LT = 220
FRAGMENTED_PAIR_SMALL_AREA_LT = 180
FRAGMENTED_PAIR_BOTH_AREA_LT = 600
FRAGMENTED_BRIDGE_THICKNESS_SCALE = 0.80
FRAGMENTED_BRIDGE_THICKNESS_MAX_PX = 9

# RESTART 5: cand top2_final_mask este prea groasa/ramificata, lucram pe
# muchia superioara a mastii, nu pe toata insula de pixeli. Asta ajuta cazuri
# precum poza 52, unde masca are ramuri in jos si convex hull-ul/bridge-ul
# contureaza o pata, nu linia pleurala.
UPPER_ENVELOPE_MODE_ENABLE = True
UPPER_ENVELOPE_COMPONENT_COUNT_GE = 8
UPPER_ENVELOPE_BBOX_HEIGHT_MIN_PX = 45
UPPER_ENVELOPE_BBOX_WIDTH_MIN_PX = 70
UPPER_ENVELOPE_Q90_THICKNESS_MIN_PX = 18
UPPER_ENVELOPE_BAND_THICKNESS_PX = 7
UPPER_ENVELOPE_SMOOTH_RADIUS_X = 5
UPPER_ENVELOPE_MAX_INTERPOLATION_GAP_X = 65
UPPER_ENVELOPE_TOP_MARGIN_PX = 0
UPPER_ENVELOPE_BOTTOM_MARGIN_PX = 1


Point = Tuple[int, int]


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

        return True


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.zeros_like(mask, dtype=np.uint8)
    binary[mask > 0] = 255
    return binary


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _compute_mask_shape_metrics(mask: np.ndarray) -> Dict[str, object]:
    binary = _as_binary_mask(mask)
    ys, xs = np.where(binary > 0)

    if len(xs) == 0:
        return {
            "has_pixels": False,
            "bbox_left": 0,
            "bbox_top": 0,
            "bbox_right": 0,
            "bbox_bottom": 0,
            "bbox_width": 0,
            "bbox_height": 0,
            "valid_column_count": 0,
            "median_column_thickness": 0.0,
            "q90_column_thickness": 0.0,
        }

    left = int(xs.min())
    right = int(xs.max())
    top = int(ys.min())
    bottom = int(ys.max())

    thickness_values: List[int] = []
    for x in range(left, right + 1):
        col_ys = np.where(binary[:, x] > 0)[0]
        if len(col_ys) == 0:
            continue
        thickness_values.append(int(col_ys.max() - col_ys.min() + 1))

    if len(thickness_values) == 0:
        median_thickness = 0.0
        q90_thickness = 0.0
    else:
        arr = np.array(thickness_values, dtype=np.float32)
        median_thickness = float(np.median(arr))
        q90_thickness = float(np.percentile(arr, 90))

    return {
        "has_pixels": True,
        "bbox_left": left,
        "bbox_top": top,
        "bbox_right": right,
        "bbox_bottom": bottom,
        "bbox_width": int(right - left + 1),
        "bbox_height": int(bottom - top + 1),
        "valid_column_count": int(len(thickness_values)),
        "median_column_thickness": median_thickness,
        "q90_column_thickness": q90_thickness,
    }


def _should_use_upper_envelope_mode(
    base_mask: np.ndarray,
    initial_components: List[Dict[str, object]],
) -> Dict[str, object]:
    metrics = _compute_mask_shape_metrics(base_mask)
    component_count = int(len(initial_components))

    reason_parts: List[str] = []
    apply_mode = False

    if not UPPER_ENVELOPE_MODE_ENABLE:
        return {
            "apply": False,
            "reason": "upper_envelope_mode_disabled",
            "component_count": component_count,
            **metrics,
        }

    if not bool(metrics.get("has_pixels", False)):
        return {
            "apply": False,
            "reason": "empty_mask",
            "component_count": component_count,
            **metrics,
        }

    if component_count >= UPPER_ENVELOPE_COMPONENT_COUNT_GE:
        apply_mode = True
        reason_parts.append("many_components")

    bbox_height = int(metrics.get("bbox_height", 0))
    bbox_width = int(metrics.get("bbox_width", 0))
    q90_thickness = float(metrics.get("q90_column_thickness", 0.0))

    vertically_ramified = (
        bbox_height >= UPPER_ENVELOPE_BBOX_HEIGHT_MIN_PX
        and bbox_width >= UPPER_ENVELOPE_BBOX_WIDTH_MIN_PX
        and q90_thickness >= UPPER_ENVELOPE_Q90_THICKNESS_MIN_PX
    )

    if vertically_ramified:
        apply_mode = True
        reason_parts.append("vertical_ramification")

    if len(reason_parts) == 0:
        reason_parts.append("normal_mask_restart4_behavior")

    return {
        "apply": bool(apply_mode),
        "reason": "+".join(reason_parts),
        "component_count": component_count,
        **metrics,
    }


def _interpolate_upper_y_values(
    valid_xs: np.ndarray,
    valid_ys: np.ndarray,
    x_start: int,
    x_end: int,
) -> Dict[int, float]:
    result: Dict[int, float] = {}

    if len(valid_xs) == 0:
        return result

    order = np.argsort(valid_xs)
    xs_sorted = valid_xs[order].astype(np.int32)
    ys_sorted = valid_ys[order].astype(np.float32)

    # In caz ca exista duplicate pe aceeasi coloana, pastram mediana.
    unique_xs: List[int] = []
    unique_ys: List[float] = []
    current_x = int(xs_sorted[0])
    bucket: List[float] = []

    for x, y in zip(xs_sorted, ys_sorted):
        x = int(x)
        y = float(y)
        if x != current_x:
            unique_xs.append(current_x)
            unique_ys.append(float(np.median(np.array(bucket, dtype=np.float32))))
            current_x = x
            bucket = [y]
        else:
            bucket.append(y)

    unique_xs.append(current_x)
    unique_ys.append(float(np.median(np.array(bucket, dtype=np.float32))))

    ux = np.array(unique_xs, dtype=np.int32)
    uy = np.array(unique_ys, dtype=np.float32)

    for x in range(int(x_start), int(x_end) + 1):
        pos = int(np.searchsorted(ux, x))

        if pos < len(ux) and int(ux[pos]) == x:
            result[x] = float(uy[pos])
            continue

        if pos == 0 or pos >= len(ux):
            continue

        left_x = int(ux[pos - 1])
        right_x = int(ux[pos])
        gap = int(right_x - left_x)

        if gap > UPPER_ENVELOPE_MAX_INTERPOLATION_GAP_X:
            continue

        t = float(x - left_x) / max(float(gap), 1.0)
        y = float(uy[pos - 1]) * (1.0 - t) + float(uy[pos]) * t
        result[x] = y

    return result


def _smooth_upper_y_values(y_by_x: Dict[int, float]) -> Dict[int, float]:
    if len(y_by_x) == 0:
        return {}

    xs = sorted(y_by_x.keys())
    smoothed: Dict[int, float] = {}
    radius = int(UPPER_ENVELOPE_SMOOTH_RADIUS_X)

    for x in xs:
        values: List[float] = []
        for nx in range(x - radius, x + radius + 1):
            if nx in y_by_x:
                values.append(float(y_by_x[nx]))

        if len(values) == 0:
            smoothed[x] = float(y_by_x[x])
        else:
            smoothed[x] = float(np.median(np.array(values, dtype=np.float32)))

    return smoothed


def _build_upper_envelope_mask(base_mask: np.ndarray) -> np.ndarray:
    """
    Extrage o banda subtire sub muchia superioara a mastii.

    Pentru cazuri ca poza 52, masca initiala poate contine ramuri in jos.
    Unificarea pe masca intreaga contureaza o pata. Aici pastram doar partea de
    sus a fiecarei coloane, adica zona care corespunde mai bine liniei pleurale.
    """
    binary = _as_binary_mask(base_mask)
    height, width = binary.shape[:2]
    ys_all, xs_all = np.where(binary > 0)

    if len(xs_all) == 0:
        return binary

    x_start = int(xs_all.min())
    x_end = int(xs_all.max())

    valid_xs: List[int] = []
    upper_ys: List[int] = []

    for x in range(x_start, x_end + 1):
        ys = np.where(binary[:, x] > 0)[0]
        if len(ys) == 0:
            continue

        valid_xs.append(int(x))
        upper_ys.append(int(ys.min()))

    if len(valid_xs) == 0:
        return binary

    y_by_x = _interpolate_upper_y_values(
        np.array(valid_xs, dtype=np.int32),
        np.array(upper_ys, dtype=np.float32),
        x_start,
        x_end,
    )
    y_by_x = _smooth_upper_y_values(y_by_x)

    envelope = np.zeros_like(binary, dtype=np.uint8)
    band = max(1, int(UPPER_ENVELOPE_BAND_THICKNESS_PX))
    top_margin = max(0, int(UPPER_ENVELOPE_TOP_MARGIN_PX))
    bottom_margin = max(0, int(UPPER_ENVELOPE_BOTTOM_MARGIN_PX))

    for x, y_float in y_by_x.items():
        y_top = int(round(float(y_float))) - top_margin
        y_bottom = y_top + band - 1 + bottom_margin
        y_top = max(0, min(height - 1, y_top))
        y_bottom = max(0, min(height - 1, y_bottom))

        if y_bottom < y_top:
            continue

        envelope[y_top : y_bottom + 1, int(x)] = 255

    if int(np.count_nonzero(envelope)) == 0:
        return binary

    # Mic close orizontal ca sa nu ramana gauri de 1 pixel dupa netezire.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    envelope = cv2.morphologyEx(envelope, cv2.MORPH_CLOSE, kernel, iterations=1)
    envelope[envelope > 0] = 255
    return envelope


def _draw_upper_envelope_decision(
    crop_bgr: np.ndarray,
    original_mask: np.ndarray,
    envelope_mask: np.ndarray,
    decision: Dict[str, object],
) -> np.ndarray:
    output = _ensure_bgr(crop_bgr)
    original = _as_binary_mask(original_mask)
    envelope = _as_binary_mask(envelope_mask)

    original_contours, _ = cv2.findContours(
        original,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    cv2.drawContours(output, original_contours, -1, (0, 255, 0), 1)

    overlay = output.copy()
    overlay[envelope > 0] = (255, 0, 255)
    output = cv2.addWeighted(overlay, 0.55, output, 0.45, 0)

    label_1 = "UPPER_ENVELOPE=ON" if bool(decision.get("apply", False)) else "UPPER_ENVELOPE=OFF"
    label_2 = (
        f"reason={decision.get('reason')} "
        f"components={decision.get('component_count')} "
        f"bbox_h={decision.get('bbox_height')} "
        f"q90_th={float(decision.get('q90_column_thickness', 0.0)):.1f}"
    )

    cv2.putText(
        output,
        label_1,
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        label_2,
        (12, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


def _draw_mask_on_crop(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = _ensure_bgr(crop_bgr)
    binary = _as_binary_mask(mask)
    output[binary > 0] = (0, 180, 0)
    return output


def _draw_contour_on_crop(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = _ensure_bgr(crop_bgr)
    binary = _as_binary_mask(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, (0, 255, 0), 1)
    return output


def _draw_added_pixels_on_crop(
    crop_bgr: np.ndarray,
    base_mask: np.ndarray,
    bridge_mask: np.ndarray,
) -> np.ndarray:
    output = _ensure_bgr(crop_bgr)
    base = _as_binary_mask(base_mask)
    bridge = _as_binary_mask(bridge_mask)

    contours, _ = cv2.findContours(base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, (0, 255, 0), 1)
    output[bridge > 0] = (255, 0, 255)
    return output


def _component_color(component_index: int) -> Tuple[int, int, int]:
    r = (37 * component_index + 80) % 256
    g = (83 * component_index + 120) % 256
    b = (149 * component_index + 180) % 256
    return int(b), int(g), int(r)


def _bbox_gap_and_overlap(
    component_a: Dict[str, object],
    component_b: Dict[str, object],
) -> Dict[str, object]:
    a_left = int(component_a["left"])
    a_right = int(component_a["right"])
    a_top = int(component_a["top"])
    a_bottom = int(component_a["bottom"])

    b_left = int(component_b["left"])
    b_right = int(component_b["right"])
    b_top = int(component_b["top"])
    b_bottom = int(component_b["bottom"])

    if a_right < b_left:
        gap_x = b_left - a_right - 1
    elif b_right < a_left:
        gap_x = a_left - b_right - 1
    else:
        gap_x = 0

    if a_bottom < b_top:
        gap_y = b_top - a_bottom - 1
    elif b_bottom < a_top:
        gap_y = a_top - b_bottom - 1
    else:
        gap_y = 0

    x_overlap = max(0, min(a_right, b_right) - max(a_left, b_left) + 1)
    y_overlap = max(0, min(a_bottom, b_bottom) - max(a_top, b_top) + 1)
    bbox_distance = float((gap_x * gap_x + gap_y * gap_y) ** 0.5)

    if x_overlap > 0 and gap_y > 0:
        relation = "overlap_x_vertical_gap"
    elif y_overlap > 0 and gap_x > 0:
        relation = "horizontal_gap"
    elif gap_x > 0 and gap_y > 0:
        relation = "diagonal_gap"
    elif x_overlap > 0 and y_overlap > 0:
        relation = "bbox_overlap_but_separate_masks"
    else:
        relation = "near_unknown"

    return {
        "gap_x": gap_x,
        "gap_y": gap_y,
        "x_overlap": x_overlap,
        "y_overlap": y_overlap,
        "bbox_distance": bbox_distance,
        "relation": relation,
    }


def _component_boundary_points(label_mask: np.ndarray) -> np.ndarray:
    binary = _as_binary_mask(label_mask)
    if int(np.count_nonzero(binary)) == 0:
        return np.empty((0, 2), dtype=np.int32)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (BOUNDARY_DILATE_KERNEL, BOUNDARY_DILATE_KERNEL),
    )
    eroded = cv2.erode(binary, kernel, iterations=1)
    boundary = cv2.subtract(binary, eroded)
    ys, xs = np.where(boundary > 0)

    if len(xs) == 0:
        ys, xs = np.where(binary > 0)

    return np.column_stack([xs, ys]).astype(np.int32)


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points

    step = max(1, len(points) // max_points)
    return points[::step]


def _closest_points(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> Tuple[Optional[Point], Optional[Point], float]:
    if points_a.size == 0 or points_b.size == 0:
        return None, None, 999999.0

    points_a = _sample_points(points_a, MAX_BOUNDARY_POINTS_FOR_DISTANCE)
    points_b = _sample_points(points_b, MAX_BOUNDARY_POINTS_FOR_DISTANCE)

    best_distance = 999999.0
    best_a: Optional[Point] = None
    best_b: Optional[Point] = None

    for start in range(0, len(points_a), 250):
        chunk = points_a[start : start + 250]
        diff = chunk[:, None, :] - points_b[None, :, :]
        dist2 = np.sum(diff.astype(np.float32) ** 2, axis=2)
        flat_index = int(np.argmin(dist2))
        local_distance = float(dist2.flat[flat_index])

        if local_distance < best_distance:
            row, col = np.unravel_index(flat_index, dist2.shape)
            best_distance = local_distance
            best_a = (int(chunk[row, 0]), int(chunk[row, 1]))
            best_b = (int(points_b[col, 0]), int(points_b[col, 1]))

    return best_a, best_b, float(best_distance ** 0.5)


def _line_empty_ratio(base_mask: np.ndarray, p1: Point, p2: Point) -> Tuple[float, int, int]:
    line_mask = np.zeros_like(base_mask, dtype=np.uint8)
    cv2.line(line_mask, p1, p2, 255, 1, cv2.LINE_8)

    ys, xs = np.where(line_mask > 0)
    if len(xs) <= 2:
        return 0.0, 0, 0

    inner_xs = xs[1:-1]
    inner_ys = ys[1:-1]
    total = len(inner_xs)
    empty = int(np.count_nonzero(base_mask[inner_ys, inner_xs] == 0))
    ratio = empty / max(total, 1)
    return float(ratio), empty, total



def _extend_line_endpoints(p1: Point, p2: Point, overlap_px: int) -> Tuple[Point, Point]:
    if overlap_px <= 0:
        return p1, p2

    x1, y1 = p1
    x2, y2 = p2
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = float((dx * dx + dy * dy) ** 0.5)

    if length < 1.0:
        return p1, p2

    ux = dx / length
    uy = dy / length

    new_p1 = (
        int(round(x1 - ux * overlap_px)),
        int(round(y1 - uy * overlap_px)),
    )
    new_p2 = (
        int(round(x2 + ux * overlap_px)),
        int(round(y2 + uy * overlap_px)),
    )

    return new_p1, new_p2


def _clip_point(point: Point, width: int, height: int) -> Point:
    x, y = point
    x = max(0, min(width - 1, int(x)))
    y = max(0, min(height - 1, int(y)))
    return x, y



def _local_mask_points(
    mask: np.ndarray,
    anchor: Point,
    radius_x: int,
    radius_y: int,
) -> np.ndarray:
    """
    Extrage doar pixelii componentei dintr-o zona mica in jurul capatului.

    Folosim acesti pixeli ca suport pentru hull-ul convex local. Daca am folosi
    toata componenta, hull-ul ar putea acoperi zone mari gresite. Asa, hull-ul
    se muleaza doar pe grosimea pleurei de langa gap.
    """
    binary = _as_binary_mask(mask)
    if int(np.count_nonzero(binary)) == 0:
        return np.empty((0, 2), dtype=np.int32)

    height, width = binary.shape[:2]
    anchor_x, anchor_y = anchor
    anchor_x = max(0, min(width - 1, int(anchor_x)))
    anchor_y = max(0, min(height - 1, int(anchor_y)))

    x1 = max(0, anchor_x - int(radius_x))
    x2 = min(width - 1, anchor_x + int(radius_x))
    y1 = max(0, anchor_y - int(radius_y))
    y2 = min(height - 1, anchor_y + int(radius_y))

    local = binary[y1 : y2 + 1, x1 : x2 + 1]
    ys, xs = np.where(local > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    xs = xs.astype(np.int32) + int(x1)
    ys = ys.astype(np.int32) + int(y1)
    return np.column_stack([xs, ys]).astype(np.int32)


def _draw_pair_bridge_with_local_hull(
    base_mask: np.ndarray,
    component_a_mask: Optional[np.ndarray],
    component_b_mask: Optional[np.ndarray],
    p1: Point,
    p2: Point,
    bridge_thickness_px: int,
    use_local_hull: bool = True,
) -> Tuple[np.ndarray, str, int]:
    """
    Construieste bridge-ul pentru o pereche.

    Varianta de baza este linia ingrosata. Pentru gap-uri simple, peste ea se
    aplica hull convex local. Pentru zone fragmentate, hull-ul este dezactivat
    si ramane doar banda controlata, ca sa nu apara pete mari intre fragmente.
    """
    pair_bridge = np.zeros_like(base_mask, dtype=np.uint8)
    cv2.line(
        pair_bridge,
        p1,
        p2,
        255,
        int(bridge_thickness_px),
        cv2.LINE_8,
    )

    if not BRIDGE_LOCAL_CONVEX_HULL_ENABLE:
        return pair_bridge, "thick_line_only_global_hull_disabled", 0

    if not use_local_hull:
        return pair_bridge, "fragmented_context_thick_line_no_hull", 0

    support_points: List[np.ndarray] = []

    ys_line, xs_line = np.where(pair_bridge > 0)
    if len(xs_line) > 0:
        support_points.append(
            np.column_stack([xs_line, ys_line]).astype(np.int32)
        )

    if component_a_mask is not None:
        points_a = _local_mask_points(
            component_a_mask,
            p1,
            BRIDGE_HULL_LOCAL_RADIUS_X,
            BRIDGE_HULL_LOCAL_RADIUS_Y,
        )
        if len(points_a) > 0:
            support_points.append(points_a)

    if component_b_mask is not None:
        points_b = _local_mask_points(
            component_b_mask,
            p2,
            BRIDGE_HULL_LOCAL_RADIUS_X,
            BRIDGE_HULL_LOCAL_RADIUS_Y,
        )
        if len(points_b) > 0:
            support_points.append(points_b)

    if len(support_points) == 0:
        return pair_bridge, "thick_line_fallback_no_support", 0

    all_points = np.vstack(support_points).astype(np.int32)
    if len(all_points) < BRIDGE_HULL_MIN_POINTS:
        return pair_bridge, "thick_line_fallback_too_few_hull_points", int(len(all_points))

    hull = cv2.convexHull(all_points.reshape(-1, 1, 2))
    hull_bridge = np.zeros_like(base_mask, dtype=np.uint8)
    cv2.fillConvexPoly(hull_bridge, hull, 255, lineType=cv2.LINE_8)

    return hull_bridge, "local_convex_hull", int(len(all_points))


def _split_sorted_values_into_runs(values: np.ndarray) -> List[Tuple[int, int]]:
    if len(values) == 0:
        return []

    values = np.sort(values.astype(np.int32))
    runs: List[Tuple[int, int]] = []
    start = int(values[0])
    previous = int(values[0])

    for value in values[1:]:
        value = int(value)
        if value == previous + 1:
            previous = value
            continue

        runs.append((start, previous))
        start = value
        previous = value

    runs.append((start, previous))
    return runs


def _run_distance_to_y(run: Tuple[int, int], y: int) -> int:
    top, bottom = run
    if top <= y <= bottom:
        return 0
    if y < top:
        return int(top - y)
    return int(y - bottom)


def _estimate_local_vertical_thickness(
    component_mask: np.ndarray,
    anchor: Point,
) -> Optional[float]:
    """
    Estimeaza grosimea locala a componentei langa un punct de legatura.

    Pentru fiecare coloana din jurul anchor-ului cauta run-ul vertical de pixeli
    al componentei care este cel mai aproape de y-ul anchor-ului. Pleura este de
    obicei o structura alungita pe orizontala, deci inaltimea locala a run-ului
    este o aproximare buna pentru grosimea pleurei.
    """
    binary = _as_binary_mask(component_mask)
    if int(np.count_nonzero(binary)) == 0:
        return None

    height, width = binary.shape[:2]
    anchor_x, anchor_y = anchor
    anchor_x = max(0, min(width - 1, int(anchor_x)))
    anchor_y = max(0, min(height - 1, int(anchor_y)))

    x_start = max(0, anchor_x - BRIDGE_LOCAL_THICKNESS_RADIUS_X)
    x_end = min(width - 1, anchor_x + BRIDGE_LOCAL_THICKNESS_RADIUS_X)

    thickness_samples: List[int] = []

    for x in range(x_start, x_end + 1):
        ys = np.where(binary[:, x] > 0)[0]
        if len(ys) == 0:
            continue

        runs = _split_sorted_values_into_runs(ys)
        if len(runs) == 0:
            continue

        best_run = min(runs, key=lambda run: _run_distance_to_y(run, anchor_y))
        distance_to_anchor = _run_distance_to_y(best_run, anchor_y)

        if distance_to_anchor > BRIDGE_LOCAL_THICKNESS_MAX_VERTICAL_DISTANCE:
            continue

        run_top, run_bottom = best_run
        thickness_samples.append(int(run_bottom - run_top + 1))

    if len(thickness_samples) == 0:
        return None

    return float(np.median(np.array(thickness_samples, dtype=np.float32)))


def _clamp_bridge_thickness(value: int) -> int:
    value = int(value)
    value = max(BRIDGE_LOCAL_THICKNESS_MIN_PX, value)
    value = min(BRIDGE_LOCAL_THICKNESS_MAX_PX, value)
    return value


def _estimate_bridge_thickness_for_pair(pair: Dict[str, object]) -> int:
    if not BRIDGE_DYNAMIC_THICKNESS_ENABLE:
        return int(BRIDGE_THICKNESS_PX)

    component_a = pair.get("component_a")
    component_b = pair.get("component_b")
    p1 = pair.get("anchor_a")
    p2 = pair.get("anchor_b")

    if component_a is None or component_b is None or p1 is None or p2 is None:
        return int(BRIDGE_THICKNESS_PX)

    p1 = tuple(int(v) for v in p1)
    p2 = tuple(int(v) for v in p2)

    thickness_values: List[float] = []

    mask_a = component_a.get("mask") if isinstance(component_a, dict) else None
    mask_b = component_b.get("mask") if isinstance(component_b, dict) else None

    if mask_a is not None:
        thickness_a = _estimate_local_vertical_thickness(mask_a, p1)
        if thickness_a is not None:
            thickness_values.append(thickness_a)
            pair["bridge_thickness_estimate_a"] = float(thickness_a)

    if mask_b is not None:
        thickness_b = _estimate_local_vertical_thickness(mask_b, p2)
        if thickness_b is not None:
            thickness_values.append(thickness_b)
            pair["bridge_thickness_estimate_b"] = float(thickness_b)

    if len(thickness_values) == 0:
        thickness_px = int(BRIDGE_THICKNESS_PX)
    else:
        thickness_px = int(round(float(np.median(thickness_values)) * BRIDGE_LOCAL_THICKNESS_SCALE))

    thickness_px = _clamp_bridge_thickness(thickness_px)

    if bool(pair.get("fragmented_context", False)):
        thickness_px = int(round(float(thickness_px) * FRAGMENTED_BRIDGE_THICKNESS_SCALE))
        thickness_px = max(BRIDGE_LOCAL_THICKNESS_MIN_PX, thickness_px)
        thickness_px = min(FRAGMENTED_BRIDGE_THICKNESS_MAX_PX, thickness_px)

    pair["bridge_thickness_px"] = int(thickness_px)
    return int(thickness_px)



def _component_intersects_roi(
    component: Dict[str, object],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> bool:
    left = int(component["left"])
    right = int(component["right"])
    top = int(component["top"])
    bottom = int(component["bottom"])

    if right < x1 or left > x2:
        return False
    if bottom < y1 or top > y2:
        return False
    return True


def _mark_fragmented_context_for_pairs(
    base_mask: np.ndarray,
    components: List[Dict[str, object]],
    pairs: List[Dict[str, object]],
) -> None:
    """
    Marcheaza perechile care sunt intr-o zona fragmentata.

    Ideea: hull-ul local e bun cand avem doua componente mari/clare. Cand imaginea
    are multe componente sau in jurul gap-ului apar mai multe fragmente mici,
    hull-ul tinde sa umple prea mult. Pentru acele perechi folosim doar banda
    ingrosata, fara hull.
    """
    if not FRAGMENTED_MODE_ENABLE:
        for pair in pairs:
            pair["fragmented_context"] = False
            pair["fragmented_reason"] = "fragmented_mode_disabled"
        return

    height, width = base_mask.shape[:2]
    global_fragmented = len(components) >= FRAGMENTED_GLOBAL_COMPONENT_COUNT_MIN

    for pair in pairs:
        pair["fragmented_context"] = False
        pair["fragmented_reason"] = "simple_gap_default"
        pair["fragmented_global_component_count"] = int(len(components))
        pair["fragmented_local_component_count"] = 0
        pair["fragmented_local_small_component_count"] = 0

        if not bool(pair.get("accepted", False)):
            continue

        p1 = pair.get("anchor_a")
        p2 = pair.get("anchor_b")
        if p1 is None or p2 is None:
            continue

        p1 = tuple(int(v) for v in p1)
        p2 = tuple(int(v) for v in p2)

        x1 = max(0, min(p1[0], p2[0]) - FRAGMENTED_LOCAL_RADIUS_X)
        x2 = min(width - 1, max(p1[0], p2[0]) + FRAGMENTED_LOCAL_RADIUS_X)
        y1 = max(0, min(p1[1], p2[1]) - FRAGMENTED_LOCAL_RADIUS_Y)
        y2 = min(height - 1, max(p1[1], p2[1]) + FRAGMENTED_LOCAL_RADIUS_Y)

        local_component_count = 0
        local_small_component_count = 0

        for component in components:
            if not _component_intersects_roi(component, x1, y1, x2, y2):
                continue

            local_component_count += 1
            if int(component["area"]) < FRAGMENTED_SMALL_COMPONENT_AREA_LT:
                local_small_component_count += 1

        pair["fragmented_local_roi"] = (int(x1), int(y1), int(x2), int(y2))
        pair["fragmented_local_component_count"] = int(local_component_count)
        pair["fragmented_local_small_component_count"] = int(local_small_component_count)

        area_a = int(pair.get("area_a", 0))
        area_b = int(pair.get("area_b", 0))
        min_area = min(area_a, area_b)
        max_area = max(area_a, area_b)

        local_fragmented = (
            local_component_count >= FRAGMENTED_LOCAL_COMPONENT_COUNT_MIN
            and local_small_component_count >= FRAGMENTED_LOCAL_SMALL_COMPONENT_COUNT_MIN
        )
        small_pair_fragmented = (
            min_area < FRAGMENTED_PAIR_SMALL_AREA_LT
            and max_area < FRAGMENTED_PAIR_BOTH_AREA_LT
        )

        if global_fragmented and (local_fragmented or small_pair_fragmented):
            pair["fragmented_context"] = True
            pair["fragmented_reason"] = "global_many_components_and_local_or_small_pair"
        elif local_fragmented:
            pair["fragmented_context"] = True
            pair["fragmented_reason"] = "local_many_components_near_gap"
        elif small_pair_fragmented:
            pair["fragmented_context"] = True
            pair["fragmented_reason"] = "small_pair_fragment"

def _build_bridge_mask_from_pairs(
    base_mask: np.ndarray,
    pairs: List[Dict[str, object]],
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    raw_bridge_mask = np.zeros_like(base_mask, dtype=np.uint8)
    height, width = base_mask.shape[:2]
    bridge_infos: List[Dict[str, object]] = []

    for pair in pairs:
        if not bool(pair.get("accepted", False)):
            continue

        p1 = pair.get("anchor_a")
        p2 = pair.get("anchor_b")
        if p1 is None or p2 is None:
            continue

        p1 = tuple(int(v) for v in p1)
        p2 = tuple(int(v) for v in p2)
        bridge_thickness_px = _estimate_bridge_thickness_for_pair(pair)
        p1, p2 = _extend_line_endpoints(p1, p2, BRIDGE_ENDPOINT_OVERLAP_PX)
        p1 = _clip_point(p1, width, height)
        p2 = _clip_point(p2, width, height)

        component_a = pair.get("component_a")
        component_b = pair.get("component_b")
        component_a_mask = None
        component_b_mask = None

        if isinstance(component_a, dict):
            component_a_mask = component_a.get("mask")

        if isinstance(component_b, dict):
            component_b_mask = component_b.get("mask")

        pair_bridge, bridge_method, hull_support_points = (
            _draw_pair_bridge_with_local_hull(
                base_mask=base_mask,
                component_a_mask=component_a_mask,
                component_b_mask=component_b_mask,
                p1=p1,
                p2=p2,
                bridge_thickness_px=bridge_thickness_px,
                use_local_hull=not bool(pair.get("fragmented_context", False)),
            )
        )

        pair["bridge_method"] = bridge_method
        pair["hull_support_points"] = int(hull_support_points)

        added_only = np.zeros_like(base_mask, dtype=np.uint8)
        added_only[(pair_bridge > 0) & (base_mask == 0)] = 255
        raw_bridge_mask[added_only > 0] = 255

        bridge_infos.append(
            {
                "component_a_index": int(pair["component_a_index"]),
                "component_b_index": int(pair["component_b_index"]),
                "p1": p1,
                "p2": p2,
                "added_pixels": int(np.count_nonzero(added_only)),
                "line_pixels_raw": int(np.count_nonzero(pair_bridge)),
                "bridge_thickness_px": int(bridge_thickness_px),
                "bridge_method": str(pair.get("bridge_method", "NA")),
                "hull_support_points": int(pair.get("hull_support_points", 0)),
                "closest_distance": float(pair.get("closest_distance", 0.0)),
                "empty_ratio": float(pair.get("empty_ratio", 0.0)),
                "fragmented_context": bool(pair.get("fragmented_context", False)),
                "fragmented_reason": str(pair.get("fragmented_reason", "NA")),
                "fragmented_local_component_count": int(pair.get("fragmented_local_component_count", 0)),
                "fragmented_local_small_component_count": int(pair.get("fragmented_local_small_component_count", 0)),
            }
        )

    return raw_bridge_mask, bridge_infos


def _extract_components(mask: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    binary = _as_binary_mask(mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    components: List[Dict[str, object]] = []

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA_PX:
            continue

        left = int(stats[label_id, cv2.CC_STAT_LEFT])
        top = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx = float(centroids[label_id][0])
        cy = float(centroids[label_id][1])
        label_mask = np.zeros_like(binary, dtype=np.uint8)
        label_mask[labels == label_id] = 255
        boundary_points = _component_boundary_points(label_mask)

        components.append(
            {
                "label_id": label_id,
                "area": area,
                "left": left,
                "top": top,
                "right": left + width - 1,
                "bottom": top + height - 1,
                "width": width,
                "height": height,
                "centroid_x": cx,
                "centroid_y": cy,
                "boundary_points": boundary_points,
                "mask": label_mask,
            }
        )

    components.sort(key=lambda item: (int(item["left"]), int(item["top"])))
    return labels, components


def _evaluate_pair(
    base_mask: np.ndarray,
    index_a: int,
    index_b: int,
    component_a: Dict[str, object],
    component_b: Dict[str, object],
) -> Dict[str, object]:
    bbox_data = _bbox_gap_and_overlap(component_a, component_b)
    p1, p2, closest_distance = _closest_points(
        component_a["boundary_points"],
        component_b["boundary_points"],
    )

    pair: Dict[str, object] = {
        "component_a_index": index_a,
        "component_b_index": index_b,
        "component_a": component_a,
        "component_b": component_b,
        "anchor_a": p1,
        "anchor_b": p2,
        "closest_distance": closest_distance,
        "accepted": False,
        "reason": "not_selected_for_chain",
    }
    pair.update(bbox_data)

    if p1 is None or p2 is None:
        pair["reason"] = "missing_boundary_points"
        return pair

    dx = int(p2[0] - p1[0])
    dy = int(p2[1] - p1[1])
    empty_ratio, empty_pixels, line_pixels = _line_empty_ratio(base_mask, p1, p2)

    pair.update(
        {
            "dx": dx,
            "dy": dy,
            "abs_dx": abs(dx),
            "abs_dy": abs(dy),
            "empty_ratio": empty_ratio,
            "empty_pixels": empty_pixels,
            "line_pixels": line_pixels,
            "area_a": int(component_a["area"]),
            "area_b": int(component_b["area"]),
        }
    )

    return pair


def _build_all_component_chain_pairs(
    base_mask: np.ndarray,
    components: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    if len(components) < 2:
        return []

    all_pairs: List[Dict[str, object]] = []

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            pair = _evaluate_pair(
                base_mask=base_mask,
                index_a=i + 1,
                index_b=j + 1,
                component_a=components[i],
                component_b=components[j],
            )
            all_pairs.append(pair)

    selectable_pairs = [
        pair
        for pair in all_pairs
        if pair.get("anchor_a") is not None and pair.get("anchor_b") is not None
    ]
    selectable_pairs.sort(
        key=lambda pair: (
            float(pair.get("closest_distance", 999999.0)),
            int(pair.get("component_a_index", 0)),
            int(pair.get("component_b_index", 0)),
        )
    )

    dsu = DisjointSet(len(components) + 1)
    selected_keys = set()

    for pair in selectable_pairs:
        a = int(pair["component_a_index"])
        b = int(pair["component_b_index"])

        if dsu.union(a, b):
            selected_keys.add((a, b))

        if len(selected_keys) >= len(components) - 1:
            break

    result_pairs: List[Dict[str, object]] = []

    for pair in all_pairs:
        a = int(pair["component_a_index"])
        b = int(pair["component_b_index"])
        pair = dict(pair)

        if (a, b) in selected_keys:
            pair["accepted"] = True
            pair["reason"] = "selected_for_all_components_minimal_chain_no_distance_limit"
        elif pair.get("anchor_a") is None or pair.get("anchor_b") is None:
            pair["accepted"] = False
            pair["reason"] = "missing_boundary_points"
        else:
            pair["accepted"] = False
            pair["reason"] = "not_needed_for_minimal_chain"

        result_pairs.append(pair)

    result_pairs.sort(
        key=lambda pair: (
            not bool(pair.get("accepted", False)),
            float(pair.get("closest_distance", 999999.0)),
            int(pair.get("component_a_index", 0)),
            int(pair.get("component_b_index", 0)),
        )
    )

    return result_pairs


def _draw_components_labeled(
    crop_bgr: np.ndarray,
    labels: np.ndarray,
    components: List[Dict[str, object]],
) -> np.ndarray:
    output = _ensure_bgr(crop_bgr)
    overlay = output.copy()

    for component_index, component in enumerate(components, start=1):
        label_id = int(component["label_id"])
        color = _component_color(component_index)
        overlay[labels == label_id] = color

    output = cv2.addWeighted(overlay, 0.65, output, 0.35, 0)

    for component_index, component in enumerate(components, start=1):
        left = int(component["left"])
        top = int(component["top"])
        right = int(component["right"])
        bottom = int(component["bottom"])
        cx = int(round(float(component["centroid_x"])))
        cy = int(round(float(component["centroid_y"])))
        color = _component_color(component_index)

        cv2.rectangle(output, (left, top), (right, bottom), color, 1)
        cv2.circle(output, (cx, cy), 3, color, -1)
        cv2.putText(
            output,
            f"{component_index} a={component['area']}",
            (left, max(12, top - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )

    return output


def _draw_pairs(
    crop_bgr: np.ndarray,
    pairs: List[Dict[str, object]],
    only_accepted: Optional[bool] = None,
) -> np.ndarray:
    output = _ensure_bgr(crop_bgr)

    for pair in pairs:
        accepted = bool(pair.get("accepted", False))

        if only_accepted is not None and accepted != only_accepted:
            continue

        p1 = pair.get("anchor_a")
        p2 = pair.get("anchor_b")
        if p1 is None or p2 is None:
            continue

        p1 = tuple(int(v) for v in p1)
        p2 = tuple(int(v) for v in p2)
        color = (0, 255, 255) if accepted else (0, 0, 255)
        thickness = 2 if accepted else 1

        cv2.line(output, p1, p2, color, thickness, cv2.LINE_AA)
        cv2.circle(output, p1, 3, color, -1)
        cv2.circle(output, p2, 3, color, -1)

        mid_x = int(round((p1[0] + p2[0]) / 2.0))
        mid_y = int(round((p1[1] + p2[1]) / 2.0))
        label = f"{pair['component_a_index']}-{pair['component_b_index']}"
        cv2.putText(
            output,
            label,
            (mid_x + 4, mid_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )

    return output



def _draw_bridge_lines(
    crop_bgr: np.ndarray,
    pairs: List[Dict[str, object]],
) -> np.ndarray:
    output = _ensure_bgr(crop_bgr)

    for pair in pairs:
        if not bool(pair.get("accepted", False)):
            continue

        p1 = pair.get("anchor_a")
        p2 = pair.get("anchor_b")
        if p1 is None or p2 is None:
            continue

        p1 = tuple(int(v) for v in p1)
        p2 = tuple(int(v) for v in p2)
        bridge_thickness_px = int(pair.get("bridge_thickness_px", BRIDGE_THICKNESS_PX))
        cv2.line(output, p1, p2, (255, 0, 255), bridge_thickness_px, cv2.LINE_AA)
        cv2.circle(output, p1, 4, (0, 255, 255), -1)
        cv2.circle(output, p2, 4, (0, 255, 255), -1)

    return output


def _draw_bridge_mask_only(bridge_mask: np.ndarray) -> np.ndarray:
    output = np.zeros((bridge_mask.shape[0], bridge_mask.shape[1], 3), dtype=np.uint8)
    output[bridge_mask > 0] = (255, 0, 255)
    return output


def _draw_endpoint_debug(
    crop_bgr: np.ndarray,
    components: List[Dict[str, object]],
    pairs: List[Dict[str, object]],
) -> np.ndarray:
    output = _ensure_bgr(crop_bgr)

    for component_index, component in enumerate(components, start=1):
        color = _component_color(component_index)
        boundary_points = component.get("boundary_points")
        if boundary_points is None:
            continue

        points = boundary_points
        if len(points) > 400:
            step = max(1, len(points) // 400)
            points = points[::step]

        for point in points:
            cv2.circle(output, (int(point[0]), int(point[1])), 1, color, -1)

    for pair in pairs:
        if not bool(pair.get("accepted", False)):
            continue

        p1 = pair.get("anchor_a")
        p2 = pair.get("anchor_b")
        if p1 is None or p2 is None:
            continue

        color = (0, 255, 255)
        p1 = tuple(int(v) for v in p1)
        p2 = tuple(int(v) for v in p2)
        cv2.line(output, p1, p2, color, 2, cv2.LINE_AA)
        cv2.circle(output, p1, 4, color, -1)
        cv2.circle(output, p2, 4, color, -1)

    return output



def _draw_fragmentation_decisions(
    crop_bgr: np.ndarray,
    pairs: List[Dict[str, object]],
) -> np.ndarray:
    output = _ensure_bgr(crop_bgr)

    for pair in pairs:
        if not bool(pair.get("accepted", False)):
            continue

        p1 = pair.get("anchor_a")
        p2 = pair.get("anchor_b")
        if p1 is None or p2 is None:
            continue

        p1 = tuple(int(v) for v in p1)
        p2 = tuple(int(v) for v in p2)
        fragmented = bool(pair.get("fragmented_context", False))
        color = (0, 165, 255) if fragmented else (0, 255, 255)
        label = "NO_HULL" if fragmented else "HULL"

        cv2.line(output, p1, p2, color, 2, cv2.LINE_AA)
        cv2.circle(output, p1, 4, color, -1)
        cv2.circle(output, p2, 4, color, -1)

        roi = pair.get("fragmented_local_roi")
        if fragmented and roi is not None:
            x1, y1, x2, y2 = tuple(int(v) for v in roi)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)

        mid_x = int(round((p1[0] + p2[0]) / 2.0))
        mid_y = int(round((p1[1] + p2[1]) / 2.0))
        cv2.putText(
            output,
            label,
            (mid_x + 4, mid_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )

    return output


def _make_report(
    original_base_mask: np.ndarray,
    base_mask: np.ndarray,
    bridge_mask: np.ndarray,
    unified_mask: np.ndarray,
    components: List[Dict[str, object]],
    pairs: List[Dict[str, object]],
    upper_envelope_decision: Optional[Dict[str, object]] = None,
) -> str:
    changed_pixels = int(np.count_nonzero(cv2.absdiff(base_mask, unified_mask)))
    changed_pixels_vs_original = int(np.count_nonzero(cv2.absdiff(original_base_mask, unified_mask)))
    bridge_pixels = int(np.count_nonzero(bridge_mask))
    original_base_pixels = int(np.count_nonzero(original_base_mask))
    base_pixels = int(np.count_nonzero(base_mask))
    accepted_pairs = [pair for pair in pairs if bool(pair.get("accepted", False))]
    rejected_pairs = [pair for pair in pairs if not bool(pair.get("accepted", False))]

    lines = [
        "TOP2 UNIFICATION - RESTART 5 UPPER ENVELOPE FRAGMENTED BRIDGE",
        f"original_base_pixels={original_base_pixels}",
        f"base_pixels_used_for_unification={base_pixels}",
        f"bridge_pixels={bridge_pixels}",
        f"changed_pixels_vs_base_used={changed_pixels}",
        f"changed_pixels_vs_original_base={changed_pixels_vs_original}",
        f"min_component_area_px={MIN_COMPONENT_AREA_PX}",
        f"component_count={len(components)}",
        f"pair_count={len(pairs)}",
        f"accepted_pair_count={len(accepted_pairs)}",
        f"rejected_pair_count={len(rejected_pairs)}",
        "distance_limits=none",
        "pair_selection=minimum_spanning_chain_all_components",
        f"bridge_default_thickness_px={BRIDGE_THICKNESS_PX}",
        f"bridge_dynamic_thickness_enable={BRIDGE_DYNAMIC_THICKNESS_ENABLE}",
        f"bridge_local_thickness_radius_x={BRIDGE_LOCAL_THICKNESS_RADIUS_X}",
        f"bridge_local_thickness_min_px={BRIDGE_LOCAL_THICKNESS_MIN_PX}",
        f"bridge_local_thickness_max_px={BRIDGE_LOCAL_THICKNESS_MAX_PX}",
        f"bridge_endpoint_overlap_px={BRIDGE_ENDPOINT_OVERLAP_PX}",
        f"bridge_local_convex_hull_enable={BRIDGE_LOCAL_CONVEX_HULL_ENABLE}",
        f"bridge_hull_local_radius_x={BRIDGE_HULL_LOCAL_RADIUS_X}",
        f"bridge_hull_local_radius_y={BRIDGE_HULL_LOCAL_RADIUS_Y}",
        f"bridge_hull_min_points={BRIDGE_HULL_MIN_POINTS}",
        f"fragmented_mode_enable={FRAGMENTED_MODE_ENABLE}",
        f"fragmented_global_component_count_min={FRAGMENTED_GLOBAL_COMPONENT_COUNT_MIN}",
        f"fragmented_local_component_count_min={FRAGMENTED_LOCAL_COMPONENT_COUNT_MIN}",
        f"fragmented_local_small_component_count_min={FRAGMENTED_LOCAL_SMALL_COMPONENT_COUNT_MIN}",
        f"fragmented_local_radius_x={FRAGMENTED_LOCAL_RADIUS_X}",
        f"fragmented_local_radius_y={FRAGMENTED_LOCAL_RADIUS_Y}",
        f"fragmented_small_component_area_lt={FRAGMENTED_SMALL_COMPONENT_AREA_LT}",
        f"fragmented_pair_small_area_lt={FRAGMENTED_PAIR_SMALL_AREA_LT}",
        f"fragmented_pair_both_area_lt={FRAGMENTED_PAIR_BOTH_AREA_LT}",
        f"fragmented_bridge_thickness_scale={FRAGMENTED_BRIDGE_THICKNESS_SCALE}",
        f"fragmented_bridge_thickness_max_px={FRAGMENTED_BRIDGE_THICKNESS_MAX_PX}",
        f"upper_envelope_mode_enable={UPPER_ENVELOPE_MODE_ENABLE}",
        f"upper_envelope_component_count_ge={UPPER_ENVELOPE_COMPONENT_COUNT_GE}",
        f"upper_envelope_bbox_height_min_px={UPPER_ENVELOPE_BBOX_HEIGHT_MIN_PX}",
        f"upper_envelope_bbox_width_min_px={UPPER_ENVELOPE_BBOX_WIDTH_MIN_PX}",
        f"upper_envelope_q90_thickness_min_px={UPPER_ENVELOPE_Q90_THICKNESS_MIN_PX}",
        f"upper_envelope_band_thickness_px={UPPER_ENVELOPE_BAND_THICKNESS_PX}",
        f"upper_envelope_max_interpolation_gap_x={UPPER_ENVELOPE_MAX_INTERPOLATION_GAP_X}",
        "expected: in upper-envelope mode, base_mask_used can be thinner than original_base_mask",
        "components:",
    ]

    if upper_envelope_decision is not None:
        lines.insert(9, f"upper_envelope_apply={upper_envelope_decision.get('apply')}")
        lines.insert(10, f"upper_envelope_reason={upper_envelope_decision.get('reason')}")
        lines.insert(11, f"upper_envelope_initial_component_count={upper_envelope_decision.get('component_count')}")
        lines.insert(12, f"upper_envelope_bbox_height={upper_envelope_decision.get('bbox_height')}")
        lines.insert(13, f"upper_envelope_q90_column_thickness={upper_envelope_decision.get('q90_column_thickness')}")

    for component_index, component in enumerate(components, start=1):
        lines.append(
            "  "
            f"#{component_index}: "
            f"area={component['area']}, "
            f"bbox=({component['left']},{component['top']})-({component['right']},{component['bottom']}), "
            f"size={component['width']}x{component['height']}, "
            f"centroid=({float(component['centroid_x']):.1f},{float(component['centroid_y']):.1f})"
        )

    lines.append("candidate_pairs:")

    for pair in pairs:
        accepted_text = "ACCEPT" if bool(pair.get("accepted", False)) else "REJECT"
        closest_distance = pair.get("closest_distance", "NA")
        bbox_distance = pair.get("bbox_distance", "NA")
        empty_ratio = pair.get("empty_ratio", "NA")

        closest_text = f"{closest_distance:.2f}" if isinstance(closest_distance, float) else str(closest_distance)
        bbox_text = f"{bbox_distance:.2f}" if isinstance(bbox_distance, float) else str(bbox_distance)
        empty_text = f"{empty_ratio:.2f}" if isinstance(empty_ratio, float) else str(empty_ratio)

        lines.append(
            "  "
            f"#{pair['component_a_index']}-#{pair['component_b_index']}: "
            f"{accepted_text}, "
            f"reason={pair.get('reason')}, "
            f"relation={pair.get('relation')}, "
            f"closest={closest_text}, bbox_distance={bbox_text}, "
            f"gap_x={pair.get('gap_x')}, gap_y={pair.get('gap_y')}, "
            f"x_overlap={pair.get('x_overlap')}, y_overlap={pair.get('y_overlap')}, "
            f"dx={pair.get('dx', 'NA')}, dy={pair.get('dy', 'NA')}, "
            f"empty_ratio={empty_text}, "
            f"bridge_thickness_px={pair.get('bridge_thickness_px', 'NA')}, "
            f"bridge_method={pair.get('bridge_method', 'NA')}, "
            f"hull_support_points={pair.get('hull_support_points', 'NA')}, "
            f"fragmented_context={pair.get('fragmented_context', 'NA')}, "
            f"fragmented_reason={pair.get('fragmented_reason', 'NA')}, "
            f"frag_local_count={pair.get('fragmented_local_component_count', 'NA')}, "
            f"frag_local_small_count={pair.get('fragmented_local_small_component_count', 'NA')}, "
            f"thickness_a={pair.get('bridge_thickness_estimate_a', 'NA')}, "
            f"thickness_b={pair.get('bridge_thickness_estimate_b', 'NA')}"
        )

    return "\n".join(lines)


def build_top2_unification_debug(
    crop_bgr: np.ndarray,
    top2_final_mask: np.ndarray,
) -> Dict[str, object]:
    """
    Restart 5 pentru unificare: daca masca top2 este foarte fragmentata sau
    ramificata vertical, nu mai unim bloburile intregi. Intai extragem o banda
    subtire dupa muchia superioara a mastii (upper envelope), apoi facem
    unificarea pe aceasta banda. Pentru imagini normale ramane comportamentul
    din restart 4.
    """
    original_base_mask = _as_binary_mask(top2_final_mask)

    initial_labels, initial_components = _extract_components(original_base_mask)
    upper_envelope_decision = _should_use_upper_envelope_mode(
        original_base_mask,
        initial_components,
    )

    upper_envelope_mask = _build_upper_envelope_mask(original_base_mask)
    if bool(upper_envelope_decision.get("apply", False)):
        base_mask = upper_envelope_mask
    else:
        base_mask = original_base_mask.copy()

    labels, components = _extract_components(base_mask)
    pairs = _build_all_component_chain_pairs(base_mask, components)
    _mark_fragmented_context_for_pairs(base_mask, components, pairs)
    bridge_mask, bridge_infos = _build_bridge_mask_from_pairs(base_mask, pairs)
    unified_mask = base_mask.copy()
    unified_mask[bridge_mask > 0] = 255

    images = {
        "00_original_top2_final_mask_on_crop": _draw_mask_on_crop(crop_bgr, original_base_mask),
        "00A_upper_envelope_base_on_crop": _draw_mask_on_crop(crop_bgr, base_mask),
        "00B_upper_envelope_decision": _draw_upper_envelope_decision(
            crop_bgr,
            original_base_mask,
            upper_envelope_mask,
            upper_envelope_decision,
        ),
        "01_all_components_labeled_no_distance_limit": _draw_components_labeled(
            crop_bgr,
            labels,
            components,
        ),
        "02_all_component_chain_preview": _draw_pairs(
            crop_bgr,
            pairs,
            only_accepted=None,
        ),
        "03_accepted_chain_pairs_only": _draw_pairs(
            crop_bgr,
            pairs,
            only_accepted=True,
        ),
        "04_minimal_bridges_added_only": _draw_bridge_mask_only(bridge_mask),
        "05_endpoint_chain_debug": _draw_endpoint_debug(
            crop_bgr,
            components,
            pairs,
        ),
        "06_unified_contour_on_crop": _draw_contour_on_crop(crop_bgr, unified_mask),
        "07_unified_mask_on_crop": _draw_mask_on_crop(crop_bgr, unified_mask),
        "08_original_plus_added_pixels": _draw_added_pixels_on_crop(
            crop_bgr,
            base_mask,
            bridge_mask,
        ),
        "09_bridge_lines_on_crop": _draw_bridge_lines(crop_bgr, pairs),
        "10_fragmentation_decisions_hull_vs_no_hull": _draw_fragmentation_decisions(
            crop_bgr,
            pairs,
        ),
    }

    report_text = _make_report(
        original_base_mask=original_base_mask,
        base_mask=base_mask,
        bridge_mask=bridge_mask,
        unified_mask=unified_mask,
        components=components,
        pairs=pairs,
        upper_envelope_decision=upper_envelope_decision,
    )

    return {
        "images": images,
        "report_text": report_text,
        "bridge_mask": bridge_mask,
        "unified_mask": unified_mask,
        "base_mask_used_for_unification": base_mask,
        "original_base_mask": original_base_mask,
        "upper_envelope_decision": upper_envelope_decision,
        "components": components,
        "candidate_pairs": pairs,
        "bridge_infos": bridge_infos,
    }
