from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

MIN_COMPONENT_AREA_PX = 20
MIN_COMPONENT_WIDTH_PX = 3
MIN_COMPONENT_HEIGHT_PX = 2

LOCAL_SMOOTH_WINDOW_PX = 7

FINAL_MASK_MIN_THICKNESS_PX = 2
FINAL_MASK_MAX_THICKNESS_PX = 8
FINAL_MASK_DEFAULT_THICKNESS_PX = 4

MIDDLE_BRIDGE_EXTRA_THICKNESS_PX = 3
MIDDLE_BRIDGE_MAX_THICKNESS_PX = 12

MIDDLE_BRIDGE_ENDPOINT_OVERLAP_PX = 18
MIDDLE_BRIDGE_CLOSE_KERNEL_PX = 3
MIDDLE_BRIDGE_CLOSE_ITERATIONS = 1

MIDDLE_BRIDGE_BEZIER_POINTS = 36
MIDDLE_BRIDGE_TANGENT_SCALE = 0.42
MIDDLE_BRIDGE_TANGENT_MIN_PX = 12
MIDDLE_BRIDGE_TANGENT_MAX_PX = 55
MIDDLE_BRIDGE_EDGE_SMOOTH_KERNEL_PX = 1

MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_MIN_PX = 1
MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_MAX_PX = 2
MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_HARD_CAP_PX = 2
MIDDLE_BRIDGE_RANDOM_PERIOD_MIN_PX = 6
MIDDLE_BRIDGE_RANDOM_PERIOD_MAX_PX = 15
MIDDLE_BRIDGE_RANDOM_HARMONICS = 4


@dataclass
class ComponentInfo:
    label: int
    area: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    width: int
    height: int
    center_x: float
    center_y: float
    median_thickness: float
    mask: np.ndarray


@dataclass
class LocalPolylineInfo:
    component_order: int
    component_label: int
    point_count: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    points: np.ndarray


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        raise ValueError("Masca de intrare este goala sau None.")

    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask.copy()

    return np.where(gray > 0, 255, 0).astype(np.uint8)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Imaginea de intrare este goala sau None.")

    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGRA2BGR)

    return image.astype(np.uint8).copy()


def _component_median_thickness(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return 0.0

    thicknesses: List[int] = []

    for x in np.unique(xs):
        col_ys = ys[xs == x]
        thicknesses.append(int(col_ys.max()) - int(col_ys.min()) + 1)

    return float(np.median(thicknesses)) if len(thicknesses) > 0 else 0.0


def _extract_components(mask: np.ndarray) -> List[ComponentInfo]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    components: List[ComponentInfo] = []

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)

        if (
            area < MIN_COMPONENT_AREA_PX
            or width < MIN_COMPONENT_WIDTH_PX
            or height < MIN_COMPONENT_HEIGHT_PX
        ):
            continue

        components.append(
            ComponentInfo(
                label=label,
                area=area,
                x_min=x,
                x_max=x + width - 1,
                y_min=y,
                y_max=y + height - 1,
                width=width,
                height=height,
                center_x=float(centroids[label][0]),
                center_y=float(centroids[label][1]),
                median_thickness=_component_median_thickness(component_mask),
                mask=component_mask,
            )
        )

    components.sort(key=lambda c: (c.x_min, c.center_y, -c.area))

    return components


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values.copy()

    if window % 2 == 0:
        window += 1

    if len(values) < window:
        return values.copy()

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    result = np.zeros_like(values)

    for index in range(len(values)):
        result[index] = int(np.median(padded[index : index + window]))

    return result


def _component_to_middle_polyline(
    component: ComponentInfo,
    component_order: int,
) -> Optional[LocalPolylineInfo]:
    xs_all: List[int] = []
    ys_all: List[int] = []

    for x in range(component.x_min, component.x_max + 1):
        ys = np.where(component.mask[:, x] > 0)[0]

        if len(ys) == 0:
            continue

        y_middle = int(round((int(ys.min()) + int(ys.max())) / 2.0))

        xs_all.append(int(x))
        ys_all.append(y_middle)

    if len(xs_all) < 2:
        return None

    xs = np.array(xs_all, dtype=np.int32)
    ys = np.array(ys_all, dtype=np.int32)

    ys = _rolling_median(ys, LOCAL_SMOOTH_WINDOW_PX)
    ys = np.clip(ys, 0, component.mask.shape[0] - 1).astype(np.int32)

    points = np.stack([xs, ys], axis=1).astype(np.int32)

    return LocalPolylineInfo(
        component_order=component_order,
        component_label=component.label,
        point_count=len(points),
        x_min=int(points[:, 0].min()),
        x_max=int(points[:, 0].max()),
        y_min=int(points[:, 1].min()),
        y_max=int(points[:, 1].max()),
        start_point=(int(points[0, 0]), int(points[0, 1])),
        end_point=(int(points[-1, 0]), int(points[-1, 1])),
        points=points,
    )


def _build_middle_polylines(components: List[ComponentInfo]) -> List[LocalPolylineInfo]:
    polylines: List[LocalPolylineInfo] = []

    for index, component in enumerate(components, start=1):
        polyline = _component_to_middle_polyline(component, index)

        if polyline is not None:
            polylines.append(polyline)

    polylines.sort(key=lambda p: (p.x_min, p.start_point[1]))

    return polylines


def _estimate_final_thickness(components: List[ComponentInfo]) -> int:
    values = [
        float(component.median_thickness)
        for component in components
        if component.median_thickness > 0
    ]

    if len(values) == 0:
        return FINAL_MASK_DEFAULT_THICKNESS_PX

    thickness = int(round(float(np.median(values))))
    thickness = max(FINAL_MASK_MIN_THICKNESS_PX, thickness)
    thickness = min(FINAL_MASK_MAX_THICKNESS_PX, thickness)

    return int(thickness)


def _sample_polyline_point_near_x(
    points: np.ndarray,
    target_x: int,
) -> Tuple[int, int]:
    if points is None or len(points) == 0:
        return (int(target_x), 0)

    xs = points[:, 0].astype(np.int32)
    index = int(np.argmin(np.abs(xs - int(target_x))))

    return (int(points[index, 0]), int(points[index, 1]))


def _polyline_local_tangent(
    points: np.ndarray,
    near_start: bool,
    sample_px: int = 18,
) -> np.ndarray:
    if points is None or len(points) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)

    pts = points[np.argsort(points[:, 0])].astype(np.float32)

    if near_start:
        p0 = pts[0]
        target_x = p0[0] + sample_px
        index = int(np.argmin(np.abs(pts[:, 0] - target_x)))
        p1 = pts[max(1, index)]
        vector = p1 - p0
    else:
        p0 = pts[-1]
        target_x = p0[0] - sample_px
        index = int(np.argmin(np.abs(pts[:, 0] - target_x)))
        p1 = pts[min(len(pts) - 2, index)]
        vector = p0 - p1

    norm = float(np.hypot(vector[0], vector[1]))

    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)

    return (vector / norm).astype(np.float32)


def _sample_cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    count: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(max(4, count)), dtype=np.float32)
    one_minus_t = 1.0 - t

    curve = (
        (one_minus_t**3)[:, None] * p0[None, :]
        + (3.0 * one_minus_t**2 * t)[:, None] * p1[None, :]
        + (3.0 * one_minus_t * t**2)[:, None] * p2[None, :]
        + (t**3)[:, None] * p3[None, :]
    )

    return curve.astype(np.float32)


def _stable_bridge_seed(
    bridge_index: int,
    p0: np.ndarray,
    p3: np.ndarray,
    distance: float,
) -> int:
    values = [
        int(bridge_index + 1) * 73856093,
        int(round(float(p0[0]))) * 19349663,
        int(round(float(p0[1]))) * 83492791,
        int(round(float(p3[0]))) * 2654435761,
        int(round(float(p3[1]))) * 97531,
        int(round(float(distance))) * 433494437,
    ]

    seed = 0

    for value in values:
        seed ^= int(value) & 0xFFFFFFFF

    return int(seed & 0xFFFFFFFF)


def _random_jagged_profile(
    cumulative: np.ndarray,
    rng: np.random.Generator,
    roughness: float,
    period: float,
) -> np.ndarray:
    if len(cumulative) == 0:
        return np.zeros(0, dtype=np.float32)

    profile = np.zeros(len(cumulative), dtype=np.float32)

    for harmonic_index in range(int(max(1, MIDDLE_BRIDGE_RANDOM_HARMONICS))):
        period_factor = float(rng.uniform(0.65, 1.75))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        weight = float(rng.uniform(0.25, 1.0)) / float(harmonic_index + 1)
        local_period = max(3.0, period * period_factor)

        profile += (
            weight * np.sin(2.0 * np.pi * cumulative / local_period + phase)
        ).astype(np.float32)

    max_abs = float(np.max(np.abs(profile))) if len(profile) > 0 else 0.0

    if max_abs > 1e-6:
        profile = profile / max_abs

    return (profile * float(roughness)).astype(np.float32)


def _ribbon_polygon_from_centerline(
    centerline: np.ndarray,
    thickness_px: int,
    shape: Tuple[int, int],
    rng: np.random.Generator,
    distance_px: float,
) -> np.ndarray:
    h, w = shape[:2]

    if centerline is None or len(centerline) < 2:
        return np.empty((0, 2), dtype=np.int32)

    half = float(max(1.0, thickness_px / 2.0))
    length_factor = float(np.clip(distance_px / 90.0, 0.65, 1.35))

    roughness_min = float(MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_MIN_PX)
    roughness_max = float(MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_MAX_PX)

    roughness = float(rng.uniform(roughness_min, roughness_max))
    roughness = float(np.clip(roughness * length_factor, roughness_min, roughness_max))
    roughness = float(
        min(
            roughness,
            float(MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_HARD_CAP_PX),
            max(1.0, half * 0.45),
        )
    )

    period = float(
        rng.uniform(
            float(MIDDLE_BRIDGE_RANDOM_PERIOD_MIN_PX),
            float(MIDDLE_BRIDGE_RANDOM_PERIOD_MAX_PX),
        )
    )

    cumulative = np.zeros(len(centerline), dtype=np.float32)

    for index in range(1, len(centerline)):
        delta = centerline[index] - centerline[index - 1]
        cumulative[index] = cumulative[index - 1] + float(np.hypot(delta[0], delta[1]))

    left_profile = _random_jagged_profile(cumulative, rng, roughness, period)
    right_profile = _random_jagged_profile(
        cumulative,
        rng,
        roughness * float(rng.uniform(0.75, 1.15)),
        period * float(rng.uniform(0.75, 1.35)),
    )

    left_side: List[List[int]] = []
    right_side: List[List[int]] = []

    for index in range(len(centerline)):
        if index == 0:
            direction = centerline[1] - centerline[0]
        elif index == len(centerline) - 1:
            direction = centerline[-1] - centerline[-2]
        else:
            direction = centerline[index + 1] - centerline[index - 1]

        norm = float(np.hypot(direction[0], direction[1]))

        if norm < 1e-6:
            normal = np.array([0.0, 1.0], dtype=np.float32)
        else:
            direction = direction / norm
            normal = np.array([-direction[1], direction[0]], dtype=np.float32)

        edge_fade = min(index, len(centerline) - 1 - index)
        fade = float(np.clip(edge_fade / 5.0, 0.0, 1.0))

        p = centerline[index]

        p_left = p + normal * max(1.0, half + float(left_profile[index]) * fade)
        p_right = p - normal * max(1.0, half + float(right_profile[index]) * fade)

        left_side.append(
            [
                int(round(float(np.clip(p_left[0], 0, w - 1)))),
                int(round(float(np.clip(p_left[1], 0, h - 1)))),
            ]
        )

        right_side.append(
            [
                int(round(float(np.clip(p_right[0], 0, w - 1)))),
                int(round(float(np.clip(p_right[1], 0, h - 1)))),
            ]
        )

    return np.array(left_side + right_side[::-1], dtype=np.int32)


def _draw_bridge_between_components(
    bridge_mask: np.ndarray,
    left: LocalPolylineInfo,
    right: LocalPolylineInfo,
    bridge_thickness: int,
    bridge_index: int,
) -> None:
    h, w = bridge_mask.shape[:2]

    left_target_x = int(
        np.clip(
            int(left.end_point[0] - MIDDLE_BRIDGE_ENDPOINT_OVERLAP_PX),
            left.x_min,
            left.x_max,
        )
    )

    right_target_x = int(
        np.clip(
            int(right.start_point[0] + MIDDLE_BRIDGE_ENDPOINT_OVERLAP_PX),
            right.x_min,
            right.x_max,
        )
    )

    p0_tuple = _sample_polyline_point_near_x(left.points, left_target_x)
    p3_tuple = _sample_polyline_point_near_x(right.points, right_target_x)

    p0 = np.array(
        [
            float(np.clip(p0_tuple[0], 0, w - 1)),
            float(np.clip(p0_tuple[1], 0, h - 1)),
        ],
        dtype=np.float32,
    )

    p3 = np.array(
        [
            float(np.clip(p3_tuple[0], 0, w - 1)),
            float(np.clip(p3_tuple[1], 0, h - 1)),
        ],
        dtype=np.float32,
    )

    distance = float(np.hypot(*(p3 - p0)))

    if distance < 1.0:
        return

    seed = _stable_bridge_seed(bridge_index, p0, p3, distance)
    rng = np.random.default_rng(seed)

    tangent_length = int(round(distance * MIDDLE_BRIDGE_TANGENT_SCALE))
    tangent_length = int(
        np.clip(
            tangent_length,
            MIDDLE_BRIDGE_TANGENT_MIN_PX,
            MIDDLE_BRIDGE_TANGENT_MAX_PX,
        )
    )
    tangent_length = int(round(float(tangent_length) * float(rng.uniform(0.82, 1.18))))

    left_tangent = _polyline_local_tangent(
        left.points,
        near_start=False,
        sample_px=max(8, MIDDLE_BRIDGE_ENDPOINT_OVERLAP_PX),
    )
    right_tangent = _polyline_local_tangent(
        right.points,
        near_start=True,
        sample_px=max(8, MIDDLE_BRIDGE_ENDPOINT_OVERLAP_PX),
    )

    p1 = p0 + left_tangent * float(tangent_length)
    p2 = p3 - right_tangent * float(tangent_length)

    direction = p3 - p0
    norm = float(np.hypot(direction[0], direction[1]))

    if norm > 1e-6:
        direction = direction / norm
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        offset = float(
            rng.uniform(
                -MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_MAX_PX,
                MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_MAX_PX,
            )
        )
        p1 = p1 + normal * offset
        p2 = p2 - normal * offset * float(rng.uniform(0.5, 1.0))

    p1[0] = float(np.clip(p1[0], 0, w - 1))
    p1[1] = float(np.clip(p1[1], 0, h - 1))
    p2[0] = float(np.clip(p2[0], 0, w - 1))
    p2[1] = float(np.clip(p2[1], 0, h - 1))

    centerline = _sample_cubic_bezier(
        p0,
        p1,
        p2,
        p3,
        MIDDLE_BRIDGE_BEZIER_POINTS,
    )

    polygon = _ribbon_polygon_from_centerline(
        centerline,
        bridge_thickness,
        bridge_mask.shape,
        rng,
        distance,
    )

    if len(polygon) >= 3:
        cv2.fillPoly(
            bridge_mask,
            [polygon.reshape((-1, 1, 2))],
            255,
            cv2.LINE_AA,
        )


def _build_bridge_mask(
    shape: Tuple[int, int],
    middle_polylines: List[LocalPolylineInfo],
    final_thickness_px: int,
) -> np.ndarray:
    bridge_mask = np.zeros(shape[:2], dtype=np.uint8)

    bridge_thickness = int(
        max(
            final_thickness_px,
            FINAL_MASK_DEFAULT_THICKNESS_PX,
        )
    )
    bridge_thickness = int(bridge_thickness + MIDDLE_BRIDGE_EXTRA_THICKNESS_PX)
    bridge_thickness = int(
        min(
            max(bridge_thickness, FINAL_MASK_MIN_THICKNESS_PX),
            MIDDLE_BRIDGE_MAX_THICKNESS_PX,
        )
    )

    for bridge_index, (left, right) in enumerate(
        zip(middle_polylines[:-1], middle_polylines[1:]),
        start=1,
    ):
        _draw_bridge_between_components(
            bridge_mask,
            left,
            right,
            bridge_thickness,
            bridge_index,
        )

    if MIDDLE_BRIDGE_CLOSE_KERNEL_PX > 1:
        close_size = int(MIDDLE_BRIDGE_CLOSE_KERNEL_PX)

        if close_size % 2 == 0:
            close_size += 1

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size),
        )

        bridge_mask = cv2.morphologyEx(
            bridge_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=int(MIDDLE_BRIDGE_CLOSE_ITERATIONS),
        )

    if MIDDLE_BRIDGE_EDGE_SMOOTH_KERNEL_PX > 1:
        smooth_size = int(MIDDLE_BRIDGE_EDGE_SMOOTH_KERNEL_PX)

        if smooth_size % 2 == 0:
            smooth_size += 1

        smooth_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (smooth_size, smooth_size),
        )

        bridge_mask = cv2.morphologyEx(
            bridge_mask,
            cv2.MORPH_CLOSE,
            smooth_kernel,
            iterations=1,
        )

    return np.where(bridge_mask > 0, 255, 0).astype(np.uint8)


def _build_final_mask(
    original_mask: np.ndarray,
    bridge_mask: np.ndarray,
) -> np.ndarray:
    final_mask = original_mask.copy()
    final_mask[bridge_mask > 0] = 255

    return np.where(final_mask > 0, 255, 0).astype(np.uint8)


def _draw_final_contour(
    base_bgr: np.ndarray,
    final_mask: np.ndarray,
) -> np.ndarray:
    out = base_bgr.copy()

    contours, _hierarchy = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    cv2.drawContours(
        out,
        contours,
        -1,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def build_top2_unification(
    crop_bgr: np.ndarray,
    top2_final_mask: np.ndarray,
    support_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    _ = support_mask

    base_bgr = _to_bgr(crop_bgr)
    original_mask = _to_binary_mask(top2_final_mask)

    components = _extract_components(original_mask)
    middle_polylines = _build_middle_polylines(components)
    final_thickness_px = _estimate_final_thickness(components)

    bridge_mask = _build_bridge_mask(
        original_mask.shape,
        middle_polylines,
        final_thickness_px,
    )

    final_mask = _build_final_mask(original_mask, bridge_mask)
    image = _draw_final_contour(base_bgr, final_mask)

    return {
        "image": image,
        "unified_mask": final_mask,
        "bridge_mask": bridge_mask,
        "components": components,
        "middle_polylines": middle_polylines,
        "final_thickness_px": final_thickness_px,
    }
