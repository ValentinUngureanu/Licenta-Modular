from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ============================================================
# UNIFICATION POLYLINE - PASUL 9
# ============================================================
# Scop:
#   masca finala curatata -> output final simplu
#
# Important:
#   - pastram conectarea si smoothing-ul global din pasul 4;
#   - detectam coborarile locale de tip down-up;
#   - ridicam doar acele zone scurte care par redundante;
#   - NU folosim inca support_mask;
#   - NU refacem inca grosimea finala a mastii.
# ============================================================


MIN_COMPONENT_AREA_PX = 20
MIN_COMPONENT_WIDTH_PX = 3
MIN_COMPONENT_HEIGHT_PX = 2

LOCAL_SMOOTH_WINDOW_PX = 7
DEBUG_LOCAL_POLYLINE_THICKNESS_PX = 1
DEBUG_FINAL_POLYLINE_THICKNESS_PX = 2
GLOBAL_SMOOTH_WINDOW_PX = 15

DIP_BASELINE_WINDOW_PX = 55
DIP_DEPTH_MIN_PX = 8
DIP_MAX_WIDTH_PX = 85
DIP_SHOULDER_DY_MAX_PX = 28
DIP_EDGE_MARGIN_POINTS = 3
DIP_FINAL_SMOOTH_WINDOW_PX = 5

FINAL_MASK_MIN_THICKNESS_PX = 2
FINAL_MASK_MAX_THICKNESS_PX = 8
FINAL_MASK_DEFAULT_THICKNESS_PX = 4
FINAL_MASK_DILATE_ITERATIONS = 1

# Reglaj doar pentru bridge-ul construit din middle polyline.
# Componentele originale NU sunt ingrosate.
MIDDLE_BRIDGE_EXTRA_THICKNESS_PX = 3
MIDDLE_BRIDGE_MAX_THICKNESS_PX = 12

# Eliminare sugrumari bridge, fara terminatii rotunjite:
# bridge-ul intra putin peste componente, dar capetele sunt plate.
MIDDLE_BRIDGE_ENDPOINT_OVERLAP_PX = 18
MIDDLE_BRIDGE_CLOSE_KERNEL_PX = 3
MIDDLE_BRIDGE_CLOSE_ITERATIONS = 1

# Naturalete bridge:
# bridge-ul nu mai este un dreptunghi drept intre componente,
# ci o banda pe o curba Bezier care urmareste directia pleurei.
MIDDLE_BRIDGE_BEZIER_POINTS = 36
MIDDLE_BRIDGE_TANGENT_SCALE = 0.42
MIDDLE_BRIDGE_TANGENT_MIN_PX = 12
MIDDLE_BRIDGE_TANGENT_MAX_PX = 55
MIDDLE_BRIDGE_EDGE_SMOOTH_KERNEL_PX = 1

# Crestare naturala pe marginea bridge-ului.
# Afecteaza DOAR bridge-ul, nu componentele originale.
MIDDLE_BRIDGE_EDGE_JAGGEDNESS_PX = 2
MIDDLE_BRIDGE_EDGE_JAGGED_PERIOD_PX = 9
MIDDLE_BRIDGE_EDGE_JAGGED_PHASE_SHIFT = 1.7

# Variatie pseudo-random pentru crestarea bridge-ului.
# Este determinista: aceeasi imagine produce acelasi rezultat,
# dar fiecare bridge primeste amplitudine/perioada/faza diferita.
MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_MIN_PX = 1
MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_MAX_PX = 2
MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_HARD_CAP_PX = 2
MIDDLE_BRIDGE_RANDOM_PERIOD_MIN_PX = 6
MIDDLE_BRIDGE_RANDOM_PERIOD_MAX_PX = 15
MIDDLE_BRIDGE_RANDOM_HARMONICS = 4

SUPPORT_BAND_EXTRA_PX = 6
SUPPORT_CLOSE_KERNEL_PX = 3

FINAL_CLEAN_MIN_COMPONENT_AREA_PX = 8
FINAL_CLEAN_CLOSE_KERNEL_PX = 3

SMALL_GAP_MAX_PX = 35
MEDIUM_GAP_MAX_PX = 120
SMALL_DY_MAX_PX = 14
MEDIUM_DY_MAX_PX = 42


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

    median_top: float
    median_bottom: float
    median_center_y: float
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


@dataclass
class PolylineGapInfo:
    index: int

    left_component_order: int
    right_component_order: int

    left_component_label: int
    right_component_label: int

    gap_px: int
    dx_px: int
    dy_px: int

    left_endpoint: Tuple[int, int]
    right_endpoint: Tuple[int, int]

    classification: str
    accepted: bool


@dataclass
class ConnectedSegmentInfo:
    segment_index: int
    point_count: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    points: np.ndarray


@dataclass
class DipCleanupInfo:
    segment_index: int
    dip_index: int

    x_start: int
    x_end: int
    width_px: int

    max_raise_px: int

    left_point: Tuple[int, int]
    right_point: Tuple[int, int]


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        raise ValueError("top2_final_mask este gol sau None")

    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask.copy()

    return np.where(gray > 0, 255, 0).astype(np.uint8)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("crop_bgr este gol sau None")

    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGRA2BGR)

    return image.astype(np.uint8).copy()


def _component_profile(mask: np.ndarray) -> Tuple[float, float, float, float]:
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0

    unique_xs = np.unique(xs)

    tops: List[int] = []
    bottoms: List[int] = []
    centers: List[float] = []
    thicknesses: List[int] = []

    for x in unique_xs:
        col_ys = ys[xs == x]

        y_top = int(col_ys.min())
        y_bottom = int(col_ys.max())

        thickness = y_bottom - y_top + 1
        center = (y_top + y_bottom) / 2.0

        tops.append(y_top)
        bottoms.append(y_bottom)
        centers.append(center)
        thicknesses.append(thickness)

    return (
        float(np.median(tops)),
        float(np.median(bottoms)),
        float(np.median(centers)),
        float(np.median(thicknesses)),
    )


def _extract_components(
    mask: np.ndarray,
) -> Tuple[List[ComponentInfo], List[ComponentInfo]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    valid_components: List[ComponentInfo] = []
    rejected_components: List[ComponentInfo] = []

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)

        median_top, median_bottom, median_center_y, median_thickness = (
            _component_profile(component_mask)
        )

        info = ComponentInfo(
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
            median_top=median_top,
            median_bottom=median_bottom,
            median_center_y=median_center_y,
            median_thickness=median_thickness,
            mask=component_mask,
        )

        if (
            area >= MIN_COMPONENT_AREA_PX
            and width >= MIN_COMPONENT_WIDTH_PX
            and height >= MIN_COMPONENT_HEIGHT_PX
        ):
            valid_components.append(info)
        else:
            rejected_components.append(info)

    valid_components.sort(key=lambda c: (c.x_min, c.center_y, -c.area))
    rejected_components.sort(key=lambda c: (c.x_min, c.center_y, -c.area))

    return valid_components, rejected_components


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values

    if window <= 1:
        return values.copy()

    if window % 2 == 0:
        window += 1

    if len(values) < window:
        return values.copy()

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")

    result = np.zeros_like(values)

    for i in range(len(values)):
        result[i] = int(np.median(padded[i : i + window]))

    return result


def _component_to_local_polyline(
    component: ComponentInfo,
    component_order: int,
) -> Optional[LocalPolylineInfo]:
    xs_all: List[int] = []
    ys_all: List[int] = []

    for x in range(component.x_min, component.x_max + 1):
        ys = np.where(component.mask[:, x] > 0)[0]

        if len(ys) == 0:
            continue

        xs_all.append(int(x))
        ys_all.append(int(ys.min()))

    if len(xs_all) < 2:
        return None

    xs = np.array(xs_all, dtype=np.int32)
    ys = np.array(ys_all, dtype=np.int32)

    ys = _rolling_median(ys, LOCAL_SMOOTH_WINDOW_PX)
    ys = np.clip(ys, 0, component.mask.shape[0] - 1).astype(np.int32)

    points = np.stack([xs, ys], axis=1).astype(np.int32)

    start_point = (int(points[0, 0]), int(points[0, 1]))
    end_point = (int(points[-1, 0]), int(points[-1, 1]))

    return LocalPolylineInfo(
        component_order=component_order,
        component_label=component.label,
        point_count=len(points),
        x_min=int(points[:, 0].min()),
        x_max=int(points[:, 0].max()),
        y_min=int(points[:, 1].min()),
        y_max=int(points[:, 1].max()),
        start_point=start_point,
        end_point=end_point,
        points=points,
    )


def _build_local_polylines(
    components: List[ComponentInfo],
) -> List[LocalPolylineInfo]:
    polylines: List[LocalPolylineInfo] = []

    for idx, component in enumerate(components, start=1):
        polyline = _component_to_local_polyline(
            component=component,
            component_order=idx,
        )

        if polyline is not None:
            polylines.append(polyline)

    polylines.sort(key=lambda p: (p.x_min, p.start_point[1]))

    return polylines


def _classify_polyline_gap(gap_px: int, dy_px: int) -> Tuple[str, bool]:
    if gap_px <= 0:
        return "overlap_or_touch", True

    if gap_px <= SMALL_GAP_MAX_PX and dy_px <= SMALL_DY_MAX_PX:
        return "small_gap_connected", True

    if gap_px <= MEDIUM_GAP_MAX_PX and dy_px <= MEDIUM_DY_MAX_PX:
        return "medium_gap_connected_raw", True

    if gap_px > MEDIUM_GAP_MAX_PX and dy_px <= MEDIUM_DY_MAX_PX:
        return "large_gap_rejected", False

    if gap_px <= MEDIUM_GAP_MAX_PX and dy_px > MEDIUM_DY_MAX_PX:
        return "vertical_jump_rejected", False

    return "large_gap_and_vertical_jump_rejected", False


def _identify_polyline_gaps(
    polylines: List[LocalPolylineInfo],
) -> List[PolylineGapInfo]:
    gaps: List[PolylineGapInfo] = []

    if len(polylines) < 2:
        return gaps

    for idx, (left, right) in enumerate(zip(polylines[:-1], polylines[1:]), start=1):
        left_endpoint = left.end_point
        right_endpoint = right.start_point

        gap_px = int(right_endpoint[0] - left_endpoint[0] - 1)
        dx_px = int(right_endpoint[0] - left_endpoint[0])
        dy_px = int(abs(right_endpoint[1] - left_endpoint[1]))

        classification, accepted = _classify_polyline_gap(gap_px, dy_px)

        gaps.append(
            PolylineGapInfo(
                index=idx,
                left_component_order=left.component_order,
                right_component_order=right.component_order,
                left_component_label=left.component_label,
                right_component_label=right.component_label,
                gap_px=gap_px,
                dx_px=dx_px,
                dy_px=dy_px,
                left_endpoint=left_endpoint,
                right_endpoint=right_endpoint,
                classification=classification,
                accepted=accepted,
            )
        )

    return gaps


def _build_linear_connector(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
) -> np.ndarray:
    x1, y1 = p1
    x2, y2 = p2

    if x2 <= x1:
        return np.empty((0, 2), dtype=np.int32)

    xs = np.arange(x1 + 1, x2, dtype=np.int32)

    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    ys = np.interp(xs, [x1, x2], [y1, y2]).astype(np.int32)

    return np.stack([xs, ys], axis=1).astype(np.int32)


def _build_connected_segments(
    polylines: List[LocalPolylineInfo],
    gaps: List[PolylineGapInfo],
) -> List[ConnectedSegmentInfo]:
    if len(polylines) == 0:
        return []

    segments_raw: List[List[np.ndarray]] = [[polylines[0].points]]

    for idx, gap in enumerate(gaps):
        next_polyline = polylines[idx + 1]

        if gap.accepted:
            connector = _build_linear_connector(
                gap.left_endpoint,
                gap.right_endpoint,
            )

            if len(connector) > 0:
                segments_raw[-1].append(connector)

            segments_raw[-1].append(next_polyline.points)
        else:
            segments_raw.append([next_polyline.points])

    segments: List[ConnectedSegmentInfo] = []

    for idx, chunks in enumerate(segments_raw, start=1):
        if len(chunks) == 0:
            continue

        points = np.vstack(chunks).astype(np.int32)

        if len(points) < 2:
            continue

        # siguranta: ordonam dupa x ca sa ramana polyline stanga-dreapta.
        points = points[np.argsort(points[:, 0])]

        segments.append(
            ConnectedSegmentInfo(
                segment_index=idx,
                point_count=len(points),
                x_min=int(points[:, 0].min()),
                x_max=int(points[:, 0].max()),
                y_min=int(points[:, 1].min()),
                y_max=int(points[:, 1].max()),
                points=points,
            )
        )

    return segments


def _smooth_connected_segments(
    connected_segments: List[ConnectedSegmentInfo],
) -> List[ConnectedSegmentInfo]:
    """
    Netezeste global fiecare segment conectat.

    Regula:
        - pastram x-urile;
        - aplicam rolling median pe y;
        - nu modificam ordinea punctelor;
        - nu unim segmente respinse anterior.
    """

    smoothed_segments: List[ConnectedSegmentInfo] = []

    for segment in connected_segments:
        points = segment.points.copy()

        if len(points) < GLOBAL_SMOOTH_WINDOW_PX:
            smoothed_points = points
        else:
            xs = points[:, 0].astype(np.int32)
            ys = points[:, 1].astype(np.int32)

            ys_smoothed = _rolling_median(ys, GLOBAL_SMOOTH_WINDOW_PX)
            ys_smoothed = np.clip(
                ys_smoothed,
                0,
                int(np.max(points[:, 1]) + 1000),
            ).astype(np.int32)

            smoothed_points = np.stack([xs, ys_smoothed], axis=1).astype(np.int32)

        smoothed_segments.append(
            ConnectedSegmentInfo(
                segment_index=segment.segment_index,
                point_count=len(smoothed_points),
                x_min=int(smoothed_points[:, 0].min()),
                x_max=int(smoothed_points[:, 0].max()),
                y_min=int(smoothed_points[:, 1].min()),
                y_max=int(smoothed_points[:, 1].max()),
                points=smoothed_points,
            )
        )

    return smoothed_segments


def _draw_connected_mask(
    shape: Tuple[int, int],
    connected_segments: List[ConnectedSegmentInfo],
) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)

    for segment in connected_segments:
        cv2.polylines(
            mask,
            [segment.points.reshape((-1, 1, 2))],
            False,
            255,
            DEBUG_FINAL_POLYLINE_THICKNESS_PX,
            cv2.LINE_AA,
        )

    return mask


def _find_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []

    start: Optional[int] = None

    for idx, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None

    if start is not None:
        runs.append((start, len(mask) - 1))

    return runs


def _clean_redundant_dips(
    smoothed_segments: List[ConnectedSegmentInfo],
) -> Tuple[List[ConnectedSegmentInfo], List[DipCleanupInfo]]:
    """
    Curata coborarile locale redundante din polyline.

    In coordonate imagine:
        y mai mare inseamna mai jos.

    O coborare redundanta este tratata ca o zona scurta unde y-ul coboara
    vizibil sub nivelul local, apoi revine. Zona este ridicata printr-o
    interpolare intre umerii din stanga si dreapta.
    """

    cleaned_segments: List[ConnectedSegmentInfo] = []
    cleanup_infos: List[DipCleanupInfo] = []

    for segment in smoothed_segments:
        points = segment.points.copy()

        if len(points) < max(DIP_BASELINE_WINDOW_PX // 2, 8):
            cleaned_segments.append(segment)
            continue

        xs = points[:, 0].astype(np.int32)
        ys_original = points[:, 1].astype(np.int32)
        ys_clean = ys_original.copy()

        baseline = _rolling_median(ys_original, DIP_BASELINE_WINDOW_PX)
        candidate_mask = ys_original > baseline + DIP_DEPTH_MIN_PX
        runs = _find_true_runs(candidate_mask)

        local_dip_index = 0

        for start_idx, end_idx in runs:
            if end_idx <= start_idx:
                continue

            x_start = int(xs[start_idx])
            x_end = int(xs[end_idx])
            width_px = int(x_end - x_start + 1)

            if width_px > DIP_MAX_WIDTH_PX:
                continue

            left_idx = max(0, start_idx - DIP_EDGE_MARGIN_POINTS)
            right_idx = min(len(xs) - 1, end_idx + DIP_EDGE_MARGIN_POINTS)

            if left_idx >= start_idx or right_idx <= end_idx:
                continue

            if (
                abs(int(ys_original[left_idx]) - int(ys_original[right_idx]))
                > DIP_SHOULDER_DY_MAX_PX
            ):
                continue

            interp_y = np.interp(
                xs[start_idx : end_idx + 1],
                [xs[left_idx], xs[right_idx]],
                [ys_original[left_idx], ys_original[right_idx]],
            )

            raise_values = ys_original[start_idx : end_idx + 1] - interp_y
            max_raise = int(round(float(np.max(raise_values))))

            if max_raise < DIP_DEPTH_MIN_PX:
                continue

            replacement = np.minimum(
                ys_clean[start_idx : end_idx + 1],
                np.rint(interp_y).astype(np.int32),
            )
            ys_clean[start_idx : end_idx + 1] = replacement

            local_dip_index += 1
            cleanup_infos.append(
                DipCleanupInfo(
                    segment_index=segment.segment_index,
                    dip_index=local_dip_index,
                    x_start=x_start,
                    x_end=x_end,
                    width_px=width_px,
                    max_raise_px=max_raise,
                    left_point=(int(xs[left_idx]), int(ys_original[left_idx])),
                    right_point=(int(xs[right_idx]), int(ys_original[right_idx])),
                )
            )

        if (
            DIP_FINAL_SMOOTH_WINDOW_PX > 1
            and len(ys_clean) >= DIP_FINAL_SMOOTH_WINDOW_PX
        ):
            ys_clean = _rolling_median(ys_clean, DIP_FINAL_SMOOTH_WINDOW_PX)

        cleaned_points = np.stack([xs, ys_clean], axis=1).astype(np.int32)

        cleaned_segments.append(
            ConnectedSegmentInfo(
                segment_index=segment.segment_index,
                point_count=len(cleaned_points),
                x_min=int(cleaned_points[:, 0].min()),
                x_max=int(cleaned_points[:, 0].max()),
                y_min=int(cleaned_points[:, 1].min()),
                y_max=int(cleaned_points[:, 1].max()),
                points=cleaned_points,
            )
        )

    return cleaned_segments, cleanup_infos


def _draw_step5_debug(
    base_bgr: np.ndarray,
    components: List[ComponentInfo],
    rejected_components: List[ComponentInfo],
    polylines: List[LocalPolylineInfo],
    gaps: List[PolylineGapInfo],
    connected_segments: List[ConnectedSegmentInfo],
    smoothed_segments: List[ConnectedSegmentInfo],
    cleaned_segments: List[ConnectedSegmentInfo],
    dip_cleanups: List[DipCleanupInfo],
) -> np.ndarray:
    out = base_bgr.copy()

    for idx, component in enumerate(components, start=1):
        cv2.rectangle(
            out,
            (component.x_min, component.y_min),
            (component.x_max, component.y_max),
            (0, 150, 0),
            1,
        )

        cv2.putText(
            out,
            f"C{idx}",
            (component.x_min, max(0, component.y_min - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 150, 0),
            1,
            cv2.LINE_AA,
        )

    for rejected in rejected_components:
        cv2.rectangle(
            out,
            (rejected.x_min, rejected.y_min),
            (rejected.x_max, rejected.y_max),
            (0, 0, 255),
            1,
        )

    # polyline-uri locale: gri
    for polyline in polylines:
        cv2.polylines(
            out,
            [polyline.points.reshape((-1, 1, 2))],
            False,
            (180, 180, 180),
            DEBUG_LOCAL_POLYLINE_THICKNESS_PX,
            cv2.LINE_AA,
        )

    # gap-uri acceptate/respinse
    for gap in gaps:
        color = (0, 255, 255) if gap.accepted else (0, 0, 255)
        cv2.line(out, gap.left_endpoint, gap.right_endpoint, color, 1, cv2.LINE_AA)

    # polyline bruta: albastru foarte subtire
    for segment in connected_segments:
        cv2.polylines(
            out,
            [segment.points.reshape((-1, 1, 2))],
            False,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # polyline netezita pasul 4: portocaliu
    for segment in smoothed_segments:
        cv2.polylines(
            out,
            [segment.points.reshape((-1, 1, 2))],
            False,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

    # zone curatate: verde
    for cleanup in dip_cleanups:
        cv2.line(
            out,
            cleanup.left_point,
            cleanup.right_point,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        mid_x = int(round((cleanup.left_point[0] + cleanup.right_point[0]) / 2.0))
        mid_y = int(round((cleanup.left_point[1] + cleanup.right_point[1]) / 2.0))

        cv2.putText(
            out,
            f"D{cleanup.segment_index}.{cleanup.dip_index}",
            (mid_x, max(0, mid_y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # polyline finala curatata: rosu gros
    for segment in cleaned_segments:
        cv2.polylines(
            out,
            [segment.points.reshape((-1, 1, 2))],
            False,
            (0, 0, 255),
            DEBUG_FINAL_POLYLINE_THICKNESS_PX,
            cv2.LINE_AA,
        )

        start = tuple(segment.points[0])
        end = tuple(segment.points[-1])

        cv2.circle(out, start, 4, (255, 255, 255), -1)
        cv2.circle(out, end, 4, (0, 0, 0), -1)

        cv2.putText(
            out,
            f"S{segment.segment_index}",
            (start[0], max(0, start[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def _build_report_step5(
    final_mask: np.ndarray,
    components: List[ComponentInfo],
    rejected_components: List[ComponentInfo],
    polylines: List[LocalPolylineInfo],
    gaps: List[PolylineGapInfo],
    connected_segments: List[ConnectedSegmentInfo],
    smoothed_segments: List[ConnectedSegmentInfo],
    cleaned_segments: List[ConnectedSegmentInfo],
    dip_cleanups: List[DipCleanupInfo],
) -> str:
    lines: List[str] = []

    lines.append("UNIFICATION POLYLINE - PASUL 5")
    lines.append("CURATARE COBORARI REDUNDANTE")
    lines.append("")
    lines.append("IMPORTANT:")
    lines.append("Se pastreaza polyline-ul netezit din pasul 4.")
    lines.append("Se ridica doar coborarile locale scurte de tip down-up.")
    lines.append("Nu se foloseste inca support_mask.")
    lines.append("Nu se refac inca grosimea si masca finala extinsa.")
    lines.append("")
    lines.append("CONFIG:")
    lines.append(f"LOCAL_SMOOTH_WINDOW_PX={LOCAL_SMOOTH_WINDOW_PX}")
    lines.append(f"GLOBAL_SMOOTH_WINDOW_PX={GLOBAL_SMOOTH_WINDOW_PX}")
    lines.append(f"DIP_BASELINE_WINDOW_PX={DIP_BASELINE_WINDOW_PX}")
    lines.append(f"DIP_DEPTH_MIN_PX={DIP_DEPTH_MIN_PX}")
    lines.append(f"DIP_MAX_WIDTH_PX={DIP_MAX_WIDTH_PX}")
    lines.append(f"DIP_SHOULDER_DY_MAX_PX={DIP_SHOULDER_DY_MAX_PX}")
    lines.append("")
    lines.append("SUMMARY:")
    lines.append(f"mask_pixels={int(cv2.countNonZero(final_mask))}")
    lines.append(f"valid_components={len(components)}")
    lines.append(f"rejected_components={len(rejected_components)}")
    lines.append(f"local_polylines={len(polylines)}")
    lines.append(f"gaps={len(gaps)}")
    lines.append(f"accepted_gaps={len([g for g in gaps if g.accepted])}")
    lines.append(f"rejected_gaps={len([g for g in gaps if not g.accepted])}")
    lines.append(f"raw_connected_segments={len(connected_segments)}")
    lines.append(f"smoothed_segments={len(smoothed_segments)}")
    lines.append(f"cleaned_segments={len(cleaned_segments)}")
    lines.append(f"dip_cleanups={len(dip_cleanups)}")
    lines.append("")

    lines.append("DIP CLEANUPS:")
    if len(dip_cleanups) == 0:
        lines.append("none")

    for cleanup in dip_cleanups:
        lines.append(
            f"D{cleanup.segment_index}.{cleanup.dip_index}: "
            f"x=[{cleanup.x_start},{cleanup.x_end}] "
            f"width_px={cleanup.width_px} "
            f"max_raise_px={cleanup.max_raise_px} "
            f"left={cleanup.left_point} "
            f"right={cleanup.right_point}"
        )

    lines.append("")
    lines.append("GAPS:")
    if len(gaps) == 0:
        lines.append("none")

    for gap in gaps:
        lines.append(
            f"G{gap.index}: "
            f"P{gap.left_component_order}->P{gap.right_component_order} "
            f"gap_px={gap.gap_px} "
            f"dx_px={gap.dx_px} "
            f"dy_px={gap.dy_px} "
            f"accepted={gap.accepted} "
            f"classification={gap.classification} "
            f"left={gap.left_endpoint} "
            f"right={gap.right_endpoint}"
        )

    lines.append("")
    lines.append("CLEANED SEGMENTS:")
    if len(cleaned_segments) == 0:
        lines.append("none")

    for segment in cleaned_segments:
        lines.append(
            f"S{segment.segment_index}: "
            f"points={segment.point_count} "
            f"x=[{segment.x_min},{segment.x_max}] "
            f"y=[{segment.y_min},{segment.y_max}] "
            f"start={tuple(segment.points[0])} "
            f"end={tuple(segment.points[-1])}"
        )

    return "\n".join(lines)


def _estimate_final_thickness(components: List[ComponentInfo]) -> int:
    """
    Estimeaza grosimea finala a mastii din grosimea componentelor detectate.

    Motiv:
        polyline-ul este doar o axa/margine.
        Pentru a avea o masca finala vizibila si utila, desenam polyline-ul
        cu o grosime apropiata de grosimea pleurei extrase anterior.
    """

    thickness_values: List[float] = []

    for component in components:
        if component.median_thickness > 0:
            thickness_values.append(float(component.median_thickness))

    if len(thickness_values) == 0:
        return FINAL_MASK_DEFAULT_THICKNESS_PX

    estimated = int(round(float(np.median(thickness_values))))

    estimated = max(FINAL_MASK_MIN_THICKNESS_PX, estimated)
    estimated = min(FINAL_MASK_MAX_THICKNESS_PX, estimated)

    return int(estimated)


def _draw_final_polyline_mask(
    shape: Tuple[int, int],
    cleaned_segments: List[ConnectedSegmentInfo],
    thickness_px: int,
) -> np.ndarray:
    """
    Transforma segmentele polyline finale intr-o masca binara.
    """

    mask = np.zeros(shape[:2], dtype=np.uint8)

    for segment in cleaned_segments:
        if len(segment.points) < 2:
            continue

        cv2.polylines(
            mask,
            [segment.points.reshape((-1, 1, 2))],
            False,
            255,
            int(thickness_px),
            cv2.LINE_AA,
        )

    if FINAL_MASK_DILATE_ITERATIONS > 0:
        kernel_size = max(3, int(thickness_px))
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=FINAL_MASK_DILATE_ITERATIONS,
        )

    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    return mask


def _overlay_binary_mask(
    base_bgr: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    out = base_bgr.copy()

    overlay = out.copy()
    overlay[mask > 0] = color

    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)

    return out


def _draw_step6_debug(
    base_bgr: np.ndarray,
    components: List[ComponentInfo],
    rejected_components: List[ComponentInfo],
    polylines: List[LocalPolylineInfo],
    gaps: List[PolylineGapInfo],
    connected_segments: List[ConnectedSegmentInfo],
    smoothed_segments: List[ConnectedSegmentInfo],
    cleaned_segments: List[ConnectedSegmentInfo],
    dip_cleanups: List[DipCleanupInfo],
    final_mask: np.ndarray,
    final_thickness_px: int,
) -> np.ndarray:
    out = base_bgr.copy()

    # masca finala ca overlay rosu
    out = _overlay_binary_mask(
        out,
        final_mask,
        (0, 0, 255),
        0.35,
    )

    # componente valide
    for idx, component in enumerate(components, start=1):
        cv2.rectangle(
            out,
            (component.x_min, component.y_min),
            (component.x_max, component.y_max),
            (0, 150, 0),
            1,
        )

        cv2.putText(
            out,
            f"C{idx}",
            (component.x_min, max(0, component.y_min - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 150, 0),
            1,
            cv2.LINE_AA,
        )

    # componente respinse
    for rejected in rejected_components:
        cv2.rectangle(
            out,
            (rejected.x_min, rejected.y_min),
            (rejected.x_max, rejected.y_max),
            (0, 0, 255),
            1,
        )

    # polyline-uri locale subtiri gri
    for polyline in polylines:
        cv2.polylines(
            out,
            [polyline.points.reshape((-1, 1, 2))],
            False,
            (180, 180, 180),
            DEBUG_LOCAL_POLYLINE_THICKNESS_PX,
            cv2.LINE_AA,
        )

    # gap-uri acceptate/respinse
    for gap in gaps:
        color = (0, 255, 255) if gap.accepted else (0, 0, 255)

        cv2.line(
            out,
            gap.left_endpoint,
            gap.right_endpoint,
            color,
            1,
            cv2.LINE_AA,
        )

    # polyline netezit pasul 4: portocaliu subtire
    for segment in smoothed_segments:
        cv2.polylines(
            out,
            [segment.points.reshape((-1, 1, 2))],
            False,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

    # zone unde au fost curatate coborari: verde
    for cleanup in dip_cleanups:
        cv2.line(
            out,
            cleanup.left_point,
            cleanup.right_point,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        mid_x = int(round((cleanup.left_point[0] + cleanup.right_point[0]) / 2.0))
        mid_y = int(round((cleanup.left_point[1] + cleanup.right_point[1]) / 2.0))

        cv2.putText(
            out,
            f"D{cleanup.segment_index}.{cleanup.dip_index}",
            (mid_x, max(0, mid_y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # polyline final curatat: rosu gros, peste masca
    for segment in cleaned_segments:
        cv2.polylines(
            out,
            [segment.points.reshape((-1, 1, 2))],
            False,
            (0, 0, 255),
            DEBUG_FINAL_POLYLINE_THICKNESS_PX,
            cv2.LINE_AA,
        )

        start = tuple(segment.points[0])
        end = tuple(segment.points[-1])

        cv2.circle(out, start, 4, (255, 255, 255), -1)
        cv2.circle(out, end, 4, (0, 0, 0), -1)

        cv2.putText(
            out,
            f"S{segment.segment_index}",
            (start[0], max(0, start[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        f"final_thickness={final_thickness_px}px",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def _build_report_step6(
    original_final_mask: np.ndarray,
    final_polyline_mask: np.ndarray,
    components: List[ComponentInfo],
    rejected_components: List[ComponentInfo],
    polylines: List[LocalPolylineInfo],
    gaps: List[PolylineGapInfo],
    connected_segments: List[ConnectedSegmentInfo],
    smoothed_segments: List[ConnectedSegmentInfo],
    cleaned_segments: List[ConnectedSegmentInfo],
    dip_cleanups: List[DipCleanupInfo],
    final_thickness_px: int,
) -> str:
    lines: List[str] = []

    lines.append("UNIFICATION POLYLINE - PASUL 6")
    lines.append("MASCA FINALA CU GROSIME CONTROLATA")
    lines.append("")
    lines.append("IMPORTANT:")
    lines.append("Se transforma polyline-ul curatat din pasul 5 intr-o masca finala.")
    lines.append("Grosimea mastii este estimata din grosimea mediana a componentelor.")
    lines.append("Nu se foloseste inca support_mask pentru corectii suplimentare.")
    lines.append("")
    lines.append("CONFIG:")
    lines.append(f"LOCAL_SMOOTH_WINDOW_PX={LOCAL_SMOOTH_WINDOW_PX}")
    lines.append(f"GLOBAL_SMOOTH_WINDOW_PX={GLOBAL_SMOOTH_WINDOW_PX}")
    lines.append(f"DIP_BASELINE_WINDOW_PX={DIP_BASELINE_WINDOW_PX}")
    lines.append(f"DIP_DEPTH_MIN_PX={DIP_DEPTH_MIN_PX}")
    lines.append(f"DIP_MAX_WIDTH_PX={DIP_MAX_WIDTH_PX}")
    lines.append(f"FINAL_MASK_MIN_THICKNESS_PX={FINAL_MASK_MIN_THICKNESS_PX}")
    lines.append(f"FINAL_MASK_MAX_THICKNESS_PX={FINAL_MASK_MAX_THICKNESS_PX}")
    lines.append(f"FINAL_MASK_DEFAULT_THICKNESS_PX={FINAL_MASK_DEFAULT_THICKNESS_PX}")
    lines.append(f"FINAL_MASK_DILATE_ITERATIONS={FINAL_MASK_DILATE_ITERATIONS}")
    lines.append("")
    lines.append("SUMMARY:")
    lines.append(f"original_mask_pixels={int(cv2.countNonZero(original_final_mask))}")
    lines.append(
        f"final_polyline_mask_pixels={int(cv2.countNonZero(final_polyline_mask))}"
    )
    lines.append(f"final_thickness_px={final_thickness_px}")
    lines.append(f"valid_components={len(components)}")
    lines.append(f"rejected_components={len(rejected_components)}")
    lines.append(f"local_polylines={len(polylines)}")
    lines.append(f"gaps={len(gaps)}")
    lines.append(f"accepted_gaps={len([g for g in gaps if g.accepted])}")
    lines.append(f"rejected_gaps={len([g for g in gaps if not g.accepted])}")
    lines.append(f"raw_connected_segments={len(connected_segments)}")
    lines.append(f"smoothed_segments={len(smoothed_segments)}")
    lines.append(f"cleaned_segments={len(cleaned_segments)}")
    lines.append(f"dip_cleanups={len(dip_cleanups)}")
    lines.append("")

    lines.append("GAPS:")
    if len(gaps) == 0:
        lines.append("none")

    for gap in gaps:
        lines.append(
            f"G{gap.index}: "
            f"P{gap.left_component_order}->P{gap.right_component_order} "
            f"gap_px={gap.gap_px} "
            f"dx_px={gap.dx_px} "
            f"dy_px={gap.dy_px} "
            f"accepted={gap.accepted} "
            f"classification={gap.classification} "
            f"left={gap.left_endpoint} "
            f"right={gap.right_endpoint}"
        )

    lines.append("")
    lines.append("DIP CLEANUPS:")
    if len(dip_cleanups) == 0:
        lines.append("none")

    for cleanup in dip_cleanups:
        lines.append(
            f"D{cleanup.segment_index}.{cleanup.dip_index}: "
            f"segment={cleanup.segment_index} "
            f"x=[{cleanup.x_start},{cleanup.x_end}] "
            f"width={cleanup.width_px} "
            f"max_raise_px={cleanup.max_raise_px} "
            f"left_point={cleanup.left_point} "
            f"right_point={cleanup.right_point}"
        )

    lines.append("")
    lines.append("FINAL CLEANED SEGMENTS:")
    if len(cleaned_segments) == 0:
        lines.append("none")

    for segment in cleaned_segments:
        lines.append(
            f"S{segment.segment_index}: "
            f"points={segment.point_count} "
            f"x=[{segment.x_min},{segment.x_max}] "
            f"y=[{segment.y_min},{segment.y_max}] "
            f"start={tuple(segment.points[0])} "
            f"end={tuple(segment.points[-1])}"
        )

    return "\n".join(lines)


def _support_guided_refine_mask(
    original_final_mask: np.ndarray,
    support_mask: Optional[np.ndarray],
    final_polyline_mask: np.ndarray,
    final_thickness_px: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rafineaza masca finala folosind support_mask intr-o banda locala.

    Regula:
        - pornim de la masca finala din polyline;
        - construim o banda ingusta in jurul ei;
        - din support_mask luam doar pixelii aflati in acea banda;
        - pastram si pixelii din top2_final_mask care sunt in acea banda;
        - nu luam nimic din support_mask aflat departe de polyline.
    """

    if support_mask is None:
        support_binary = np.zeros_like(original_final_mask)
    else:
        support_binary = _to_binary_mask(support_mask)

    band_kernel_size = int(max(3, final_thickness_px * 2 + SUPPORT_BAND_EXTRA_PX))

    if band_kernel_size % 2 == 0:
        band_kernel_size += 1

    band_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (band_kernel_size, band_kernel_size),
    )

    local_band = cv2.dilate(
        final_polyline_mask,
        band_kernel,
        iterations=1,
    )

    local_band = np.where(local_band > 0, 255, 0).astype(np.uint8)

    support_near_polyline = cv2.bitwise_and(
        support_binary,
        local_band,
    )

    original_near_polyline = cv2.bitwise_and(
        original_final_mask,
        local_band,
    )

    refined_mask = np.zeros_like(original_final_mask)
    refined_mask[final_polyline_mask > 0] = 255
    refined_mask[support_near_polyline > 0] = 255
    refined_mask[original_near_polyline > 0] = 255

    if SUPPORT_CLOSE_KERNEL_PX > 1:
        close_size = int(SUPPORT_CLOSE_KERNEL_PX)

        if close_size % 2 == 0:
            close_size += 1

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size),
        )

        refined_mask = cv2.morphologyEx(
            refined_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=1,
        )

    refined_mask = np.where(refined_mask > 0, 255, 0).astype(np.uint8)

    support_added_mask = np.zeros_like(original_final_mask)
    support_added_mask[refined_mask > 0] = 255
    support_added_mask[final_polyline_mask > 0] = 0
    support_added_mask = np.where(support_added_mask > 0, 255, 0).astype(np.uint8)

    return refined_mask, support_added_mask, local_band


def _draw_step7_debug(
    base_bgr: np.ndarray,
    components: List[ComponentInfo],
    rejected_components: List[ComponentInfo],
    polylines: List[LocalPolylineInfo],
    gaps: List[PolylineGapInfo],
    smoothed_segments: List[ConnectedSegmentInfo],
    cleaned_segments: List[ConnectedSegmentInfo],
    dip_cleanups: List[DipCleanupInfo],
    step6_polyline_mask: np.ndarray,
    support_guided_mask: np.ndarray,
    support_added_mask: np.ndarray,
    support_band_mask: np.ndarray,
    final_thickness_px: int,
) -> np.ndarray:
    out = base_bgr.copy()

    # banda locala folosita pentru support_mask: albastru foarte transparent
    out = _overlay_binary_mask(
        out,
        support_band_mask,
        (255, 0, 0),
        0.12,
    )

    # masca finala support-guided: rosu transparent
    out = _overlay_binary_mask(
        out,
        support_guided_mask,
        (0, 0, 255),
        0.32,
    )

    # pixeli adaugati de support/original in banda: verde
    out = _overlay_binary_mask(
        out,
        support_added_mask,
        (0, 255, 0),
        0.55,
    )

    # componente valide
    for idx, component in enumerate(components, start=1):
        cv2.rectangle(
            out,
            (component.x_min, component.y_min),
            (component.x_max, component.y_max),
            (0, 150, 0),
            1,
        )

        cv2.putText(
            out,
            f"C{idx}",
            (component.x_min, max(0, component.y_min - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 150, 0),
            1,
            cv2.LINE_AA,
        )

    # componente respinse
    for rejected in rejected_components:
        cv2.rectangle(
            out,
            (rejected.x_min, rejected.y_min),
            (rejected.x_max, rejected.y_max),
            (0, 0, 255),
            1,
        )

    # polyline-uri locale subtiri gri
    for polyline in polylines:
        cv2.polylines(
            out,
            [polyline.points.reshape((-1, 1, 2))],
            False,
            (180, 180, 180),
            DEBUG_LOCAL_POLYLINE_THICKNESS_PX,
            cv2.LINE_AA,
        )

    # gap-uri acceptate/respinse
    for gap in gaps:
        color = (0, 255, 255) if gap.accepted else (0, 0, 255)

        cv2.line(
            out,
            gap.left_endpoint,
            gap.right_endpoint,
            color,
            1,
            cv2.LINE_AA,
        )

    # polyline netezit pasul 4: portocaliu subtire
    for segment in smoothed_segments:
        cv2.polylines(
            out,
            [segment.points.reshape((-1, 1, 2))],
            False,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

    # zone unde au fost curatate coborari: verde
    for cleanup in dip_cleanups:
        cv2.line(
            out,
            cleanup.left_point,
            cleanup.right_point,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # polyline final curatat: rosu gros
    for segment in cleaned_segments:
        cv2.polylines(
            out,
            [segment.points.reshape((-1, 1, 2))],
            False,
            (0, 0, 255),
            DEBUG_FINAL_POLYLINE_THICKNESS_PX,
            cv2.LINE_AA,
        )

        start = tuple(segment.points[0])
        end = tuple(segment.points[-1])

        cv2.circle(out, start, 4, (255, 255, 255), -1)
        cv2.circle(out, end, 4, (0, 0, 0), -1)

    cv2.putText(
        out,
        f"step7 support-guided | thickness={final_thickness_px}px",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        "blue=band  red=final  green=added support",
        (10, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def _build_report_step7(
    original_final_mask: np.ndarray,
    step6_polyline_mask: np.ndarray,
    support_guided_mask: np.ndarray,
    support_added_mask: np.ndarray,
    support_band_mask: np.ndarray,
    components: List[ComponentInfo],
    rejected_components: List[ComponentInfo],
    polylines: List[LocalPolylineInfo],
    gaps: List[PolylineGapInfo],
    connected_segments: List[ConnectedSegmentInfo],
    smoothed_segments: List[ConnectedSegmentInfo],
    cleaned_segments: List[ConnectedSegmentInfo],
    dip_cleanups: List[DipCleanupInfo],
    final_thickness_px: int,
) -> str:
    lines: List[str] = []

    lines.append("UNIFICATION POLYLINE - PASUL 7")
    lines.append("RAFINARE GHIDATA DE SUPPORT_MASK")
    lines.append("")
    lines.append("IMPORTANT:")
    lines.append("Se pastreaza masca finala din pasul 6.")
    lines.append(
        "support_mask este folosit doar intr-o banda locala in jurul polyline-ului."
    )
    lines.append("Pixelii de suport din afara benzii sunt ignorati.")
    lines.append("")
    lines.append("CONFIG:")
    lines.append(f"LOCAL_SMOOTH_WINDOW_PX={LOCAL_SMOOTH_WINDOW_PX}")
    lines.append(f"GLOBAL_SMOOTH_WINDOW_PX={GLOBAL_SMOOTH_WINDOW_PX}")
    lines.append(f"DIP_BASELINE_WINDOW_PX={DIP_BASELINE_WINDOW_PX}")
    lines.append(f"DIP_DEPTH_MIN_PX={DIP_DEPTH_MIN_PX}")
    lines.append(f"DIP_MAX_WIDTH_PX={DIP_MAX_WIDTH_PX}")
    lines.append(f"FINAL_MASK_MIN_THICKNESS_PX={FINAL_MASK_MIN_THICKNESS_PX}")
    lines.append(f"FINAL_MASK_MAX_THICKNESS_PX={FINAL_MASK_MAX_THICKNESS_PX}")
    lines.append(f"FINAL_MASK_DEFAULT_THICKNESS_PX={FINAL_MASK_DEFAULT_THICKNESS_PX}")
    lines.append(f"FINAL_MASK_DILATE_ITERATIONS={FINAL_MASK_DILATE_ITERATIONS}")
    lines.append(f"SUPPORT_BAND_EXTRA_PX={SUPPORT_BAND_EXTRA_PX}")
    lines.append(f"SUPPORT_CLOSE_KERNEL_PX={SUPPORT_CLOSE_KERNEL_PX}")
    lines.append("")
    lines.append("SUMMARY:")
    lines.append(f"original_mask_pixels={int(cv2.countNonZero(original_final_mask))}")
    lines.append(
        f"step6_polyline_mask_pixels={int(cv2.countNonZero(step6_polyline_mask))}"
    )
    lines.append(
        f"support_guided_mask_pixels={int(cv2.countNonZero(support_guided_mask))}"
    )
    lines.append(f"support_added_pixels={int(cv2.countNonZero(support_added_mask))}")
    lines.append(f"support_band_pixels={int(cv2.countNonZero(support_band_mask))}")
    lines.append(f"final_thickness_px={final_thickness_px}")
    lines.append(f"valid_components={len(components)}")
    lines.append(f"rejected_components={len(rejected_components)}")
    lines.append(f"local_polylines={len(polylines)}")
    lines.append(f"gaps={len(gaps)}")
    lines.append(f"accepted_gaps={len([g for g in gaps if g.accepted])}")
    lines.append(f"rejected_gaps={len([g for g in gaps if not g.accepted])}")
    lines.append(f"raw_connected_segments={len(connected_segments)}")
    lines.append(f"smoothed_segments={len(smoothed_segments)}")
    lines.append(f"cleaned_segments={len(cleaned_segments)}")
    lines.append(f"dip_cleanups={len(dip_cleanups)}")
    lines.append("")

    lines.append("GAPS:")
    if len(gaps) == 0:
        lines.append("none")

    for gap in gaps:
        lines.append(
            f"G{gap.index}: "
            f"P{gap.left_component_order}->P{gap.right_component_order} "
            f"gap_px={gap.gap_px} "
            f"dx_px={gap.dx_px} "
            f"dy_px={gap.dy_px} "
            f"accepted={gap.accepted} "
            f"classification={gap.classification} "
            f"left={gap.left_endpoint} "
            f"right={gap.right_endpoint}"
        )

    lines.append("")
    lines.append("DIP CLEANUPS:")
    if len(dip_cleanups) == 0:
        lines.append("none")

    for cleanup in dip_cleanups:
        lines.append(
            f"D{cleanup.segment_index}.{cleanup.dip_index}: "
            f"segment={cleanup.segment_index} "
            f"x=[{cleanup.x_start},{cleanup.x_end}] "
            f"width={cleanup.width_px} "
            f"max_raise_px={cleanup.max_raise_px} "
            f"left_point={cleanup.left_point} "
            f"right_point={cleanup.right_point}"
        )

    return "\n".join(lines)


def _final_clean_keep_core_connected(
    step6_polyline_mask: np.ndarray,
    support_guided_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Curata masca rafinata din pasul 7.

    Regula:
        - impartim support_guided_mask in componente conexe;
        - pastram doar componentele care ating masca de baza a polyline-ului;
        - eliminam componentele care au aparut doar din support_mask si sunt izolate;
        - astfel nu schimbam traseul liniei, doar stergem fragmente fara suport real.
    """

    guided = _to_binary_mask(support_guided_mask)
    core = _to_binary_mask(step6_polyline_mask)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        guided,
        8,
    )

    cleaned = np.zeros_like(guided)
    removed = np.zeros_like(guided)

    kept_components = 0
    removed_components = 0
    kept_pixels = 0
    removed_pixels = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)

        overlap_with_core = cv2.countNonZero(cv2.bitwise_and(component_mask, core))

        if overlap_with_core > 0:
            cleaned[component_mask > 0] = 255
            kept_components += 1
            kept_pixels += area
        else:
            if area >= FINAL_CLEAN_MIN_COMPONENT_AREA_PX:
                removed[component_mask > 0] = 255
                removed_components += 1
                removed_pixels += area
            else:
                removed[component_mask > 0] = 255
                removed_components += 1
                removed_pixels += area

    if FINAL_CLEAN_CLOSE_KERNEL_PX > 1:
        close_size = int(FINAL_CLEAN_CLOSE_KERNEL_PX)

        if close_size % 2 == 0:
            close_size += 1

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size),
        )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=1,
        )

    cleaned = np.where(cleaned > 0, 255, 0).astype(np.uint8)
    removed = np.where(removed > 0, 255, 0).astype(np.uint8)

    info = {
        "total_components": int(num_labels - 1),
        "kept_components": int(kept_components),
        "removed_components": int(removed_components),
        "kept_pixels": int(kept_pixels),
        "removed_pixels": int(removed_pixels),
    }

    return cleaned, removed, info


def _draw_step8_debug(
    base_bgr: np.ndarray,
    components: List[ComponentInfo],
    rejected_components: List[ComponentInfo],
    polylines: List[LocalPolylineInfo],
    gaps: List[PolylineGapInfo],
    smoothed_segments: List[ConnectedSegmentInfo],
    cleaned_segments: List[ConnectedSegmentInfo],
    dip_cleanups: List[DipCleanupInfo],
    step7_support_guided_mask: np.ndarray,
    final_clean_mask: np.ndarray,
    removed_fragments_mask: np.ndarray,
    final_thickness_px: int,
) -> np.ndarray:
    out = base_bgr.copy()

    # masca pasului 7: albastru transparent, pentru comparatie
    out = _overlay_binary_mask(
        out,
        step7_support_guided_mask,
        (255, 0, 0),
        0.16,
    )

    # fragmente eliminate: magenta/roz
    out = _overlay_binary_mask(
        out,
        removed_fragments_mask,
        (255, 0, 255),
        0.65,
    )

    # masca finala curatata: rosu
    out = _overlay_binary_mask(
        out,
        final_clean_mask,
        (0, 0, 255),
        0.36,
    )

    # componente valide
    for idx, component in enumerate(components, start=1):
        cv2.rectangle(
            out,
            (component.x_min, component.y_min),
            (component.x_max, component.y_max),
            (0, 150, 0),
            1,
        )

        cv2.putText(
            out,
            f"C{idx}",
            (component.x_min, max(0, component.y_min - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 150, 0),
            1,
            cv2.LINE_AA,
        )

    # componente respinse
    for rejected in rejected_components:
        cv2.rectangle(
            out,
            (rejected.x_min, rejected.y_min),
            (rejected.x_max, rejected.y_max),
            (0, 0, 255),
            1,
        )

    # polyline-uri locale subtiri gri
    for polyline in polylines:
        cv2.polylines(
            out,
            [polyline.points.reshape((-1, 1, 2))],
            False,
            (180, 180, 180),
            DEBUG_LOCAL_POLYLINE_THICKNESS_PX,
            cv2.LINE_AA,
        )

    # gap-uri acceptate/respinse
    for gap in gaps:
        color = (0, 255, 255) if gap.accepted else (0, 0, 255)

        cv2.line(
            out,
            gap.left_endpoint,
            gap.right_endpoint,
            color,
            1,
            cv2.LINE_AA,
        )

    # polyline netezit pasul 4: portocaliu subtire
    for segment in smoothed_segments:
        cv2.polylines(
            out,
            [segment.points.reshape((-1, 1, 2))],
            False,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

    # zone unde au fost curatate coborari: verde
    for cleanup in dip_cleanups:
        cv2.line(
            out,
            cleanup.left_point,
            cleanup.right_point,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # polyline final curatat: rosu gros
    for segment in cleaned_segments:
        cv2.polylines(
            out,
            [segment.points.reshape((-1, 1, 2))],
            False,
            (0, 0, 255),
            DEBUG_FINAL_POLYLINE_THICKNESS_PX,
            cv2.LINE_AA,
        )

        start = tuple(segment.points[0])
        end = tuple(segment.points[-1])

        cv2.circle(out, start, 4, (255, 255, 255), -1)
        cv2.circle(out, end, 4, (0, 0, 0), -1)

    cv2.putText(
        out,
        f"step8 final clean | thickness={final_thickness_px}px",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        "red=final  blue=step7  magenta=removed",
        (10, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def _build_report_step8(
    original_final_mask: np.ndarray,
    step6_polyline_mask: np.ndarray,
    step7_support_guided_mask: np.ndarray,
    final_clean_mask: np.ndarray,
    removed_fragments_mask: np.ndarray,
    support_added_mask: np.ndarray,
    clean_info: Dict[str, int],
    components: List[ComponentInfo],
    rejected_components: List[ComponentInfo],
    polylines: List[LocalPolylineInfo],
    gaps: List[PolylineGapInfo],
    connected_segments: List[ConnectedSegmentInfo],
    smoothed_segments: List[ConnectedSegmentInfo],
    cleaned_segments: List[ConnectedSegmentInfo],
    dip_cleanups: List[DipCleanupInfo],
    final_thickness_px: int,
) -> str:
    lines: List[str] = []

    lines.append("UNIFICATION POLYLINE - PASUL 8")
    lines.append("CURATARE FINALA FRAGMENTE IZOLATE")
    lines.append("")
    lines.append("IMPORTANT:")
    lines.append(
        "Se pastreaza numai componentele din masca pasului 7 care ating masca polyline-ului de baza."
    )
    lines.append(
        "Fragmentele din support_mask care nu au legatura cu polyline-ul sunt eliminate."
    )
    lines.append("Traseul polyline-ului nu este recalculat.")
    lines.append("")
    lines.append("CONFIG:")
    lines.append(f"LOCAL_SMOOTH_WINDOW_PX={LOCAL_SMOOTH_WINDOW_PX}")
    lines.append(f"GLOBAL_SMOOTH_WINDOW_PX={GLOBAL_SMOOTH_WINDOW_PX}")
    lines.append(f"DIP_BASELINE_WINDOW_PX={DIP_BASELINE_WINDOW_PX}")
    lines.append(f"DIP_DEPTH_MIN_PX={DIP_DEPTH_MIN_PX}")
    lines.append(f"DIP_MAX_WIDTH_PX={DIP_MAX_WIDTH_PX}")
    lines.append(f"FINAL_MASK_MIN_THICKNESS_PX={FINAL_MASK_MIN_THICKNESS_PX}")
    lines.append(f"FINAL_MASK_MAX_THICKNESS_PX={FINAL_MASK_MAX_THICKNESS_PX}")
    lines.append(f"FINAL_MASK_DEFAULT_THICKNESS_PX={FINAL_MASK_DEFAULT_THICKNESS_PX}")
    lines.append(f"FINAL_MASK_DILATE_ITERATIONS={FINAL_MASK_DILATE_ITERATIONS}")
    lines.append(f"SUPPORT_BAND_EXTRA_PX={SUPPORT_BAND_EXTRA_PX}")
    lines.append(f"SUPPORT_CLOSE_KERNEL_PX={SUPPORT_CLOSE_KERNEL_PX}")
    lines.append(
        f"FINAL_CLEAN_MIN_COMPONENT_AREA_PX={FINAL_CLEAN_MIN_COMPONENT_AREA_PX}"
    )
    lines.append(f"FINAL_CLEAN_CLOSE_KERNEL_PX={FINAL_CLEAN_CLOSE_KERNEL_PX}")
    lines.append("")
    lines.append("SUMMARY:")
    lines.append(f"original_mask_pixels={int(cv2.countNonZero(original_final_mask))}")
    lines.append(
        f"step6_polyline_mask_pixels={int(cv2.countNonZero(step6_polyline_mask))}"
    )
    lines.append(
        f"step7_support_guided_mask_pixels={int(cv2.countNonZero(step7_support_guided_mask))}"
    )
    lines.append(
        f"support_added_pixels_step7={int(cv2.countNonZero(support_added_mask))}"
    )
    lines.append(f"final_clean_mask_pixels={int(cv2.countNonZero(final_clean_mask))}")
    lines.append(
        f"removed_fragments_pixels={int(cv2.countNonZero(removed_fragments_mask))}"
    )
    lines.append(f"final_thickness_px={final_thickness_px}")
    lines.append(f"valid_components={len(components)}")
    lines.append(f"rejected_components={len(rejected_components)}")
    lines.append(f"local_polylines={len(polylines)}")
    lines.append(f"gaps={len(gaps)}")
    lines.append(f"accepted_gaps={len([g for g in gaps if g.accepted])}")
    lines.append(f"rejected_gaps={len([g for g in gaps if not g.accepted])}")
    lines.append(f"raw_connected_segments={len(connected_segments)}")
    lines.append(f"smoothed_segments={len(smoothed_segments)}")
    lines.append(f"cleaned_segments={len(cleaned_segments)}")
    lines.append(f"dip_cleanups={len(dip_cleanups)}")
    lines.append(f"clean_total_components={clean_info['total_components']}")
    lines.append(f"clean_kept_components={clean_info['kept_components']}")
    lines.append(f"clean_removed_components={clean_info['removed_components']}")
    lines.append(f"clean_kept_pixels={clean_info['kept_pixels']}")
    lines.append(f"clean_removed_pixels={clean_info['removed_pixels']}")
    lines.append("")

    lines.append("GAPS:")
    if len(gaps) == 0:
        lines.append("none")

    for gap in gaps:
        lines.append(
            f"G{gap.index}: "
            f"P{gap.left_component_order}->P{gap.right_component_order} "
            f"gap_px={gap.gap_px} "
            f"dx_px={gap.dx_px} "
            f"dy_px={gap.dy_px} "
            f"accepted={gap.accepted} "
            f"classification={gap.classification} "
            f"left={gap.left_endpoint} "
            f"right={gap.right_endpoint}"
        )

    lines.append("")
    lines.append("DIP CLEANUPS:")
    if len(dip_cleanups) == 0:
        lines.append("none")

    for cleanup in dip_cleanups:
        lines.append(
            f"D{cleanup.segment_index}.{cleanup.dip_index}: "
            f"segment={cleanup.segment_index} "
            f"x=[{cleanup.x_start},{cleanup.x_end}] "
            f"width={cleanup.width_px} "
            f"max_raise_px={cleanup.max_raise_px} "
            f"left_point={cleanup.left_point} "
            f"right_point={cleanup.right_point}"
        )

    return "\n".join(lines)


def _draw_final_polyline_only_mask(
    shape: Tuple[int, int],
    cleaned_segments: List[ConnectedSegmentInfo],
) -> np.ndarray:
    """
    Creeaza o masca subtire doar cu traseul final al polyline-ului.
    """

    mask = np.zeros(shape[:2], dtype=np.uint8)

    for segment in cleaned_segments:
        if len(segment.points) < 2:
            continue

        cv2.polylines(
            mask,
            [segment.points.reshape((-1, 1, 2))],
            False,
            255,
            DEBUG_FINAL_POLYLINE_THICKNESS_PX,
            cv2.LINE_AA,
        )

    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    return mask


def _draw_step9_final_output(
    base_bgr: np.ndarray,
    final_clean_mask: np.ndarray,
    final_polyline_only: np.ndarray,
) -> np.ndarray:
    """
    Imagine finala simpla:
        - masca finala transparenta;
        - polyline final rosu clar.
    """

    out = base_bgr.copy()

    out = _overlay_binary_mask(
        out,
        final_clean_mask,
        (0, 0, 255),
        0.34,
    )

    contours, _hierarchy = cv2.findContours(
        final_clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        out,
        contours,
        -1,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    out[final_polyline_only > 0] = (0, 0, 255)

    return out


def _draw_before_after_overlay(
    base_bgr: np.ndarray,
    original_final_mask: np.ndarray,
    final_clean_mask: np.ndarray,
    final_polyline_only: np.ndarray,
) -> np.ndarray:
    """
    Comparatie:
        - albastru = top2_final_mask primit ca input la unification;
        - rosu = masca finala dupa polyline unification;
        - verde = zona comuna;
        - linie rosie = traseu polyline final.
    """

    original = _to_binary_mask(original_final_mask)
    final = _to_binary_mask(final_clean_mask)

    overlap = cv2.bitwise_and(original, final)

    original_only = original.copy()
    original_only[overlap > 0] = 0

    final_only = final.copy()
    final_only[overlap > 0] = 0

    out = base_bgr.copy()

    out = _overlay_binary_mask(
        out,
        original_only,
        (255, 0, 0),
        0.35,
    )

    out = _overlay_binary_mask(
        out,
        final_only,
        (0, 0, 255),
        0.35,
    )

    out = _overlay_binary_mask(
        out,
        overlap,
        (0, 255, 0),
        0.32,
    )

    out[final_polyline_only > 0] = (0, 0, 255)

    cv2.putText(
        out,
        "blue=original top2  red=final only  green=overlap",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def _build_report_step9(
    original_final_mask: np.ndarray,
    step6_polyline_mask: np.ndarray,
    step7_support_guided_mask: np.ndarray,
    final_clean_mask: np.ndarray,
    final_polyline_only: np.ndarray,
    removed_fragments_mask: np.ndarray,
    support_added_mask: np.ndarray,
    clean_info: Dict[str, int],
    components: List[ComponentInfo],
    rejected_components: List[ComponentInfo],
    polylines: List[LocalPolylineInfo],
    gaps: List[PolylineGapInfo],
    connected_segments: List[ConnectedSegmentInfo],
    smoothed_segments: List[ConnectedSegmentInfo],
    cleaned_segments: List[ConnectedSegmentInfo],
    dip_cleanups: List[DipCleanupInfo],
    final_thickness_px: int,
) -> str:
    original_pixels = int(cv2.countNonZero(original_final_mask))
    final_pixels = int(cv2.countNonZero(final_clean_mask))
    overlap_pixels = int(
        cv2.countNonZero(cv2.bitwise_and(original_final_mask, final_clean_mask))
    )

    final_only = final_clean_mask.copy()
    final_only[original_final_mask > 0] = 0

    original_only = original_final_mask.copy()
    original_only[final_clean_mask > 0] = 0

    lines: List[str] = []

    lines.append("UNIFICATION POLYLINE - PASUL 9")
    lines.append("OUTPUT FINAL SIMPLU")
    lines.append("")
    lines.append("IMPORTANT:")
    lines.append("Nu se mai modifica algoritmic masca.")
    lines.append("Acest pas genereaza imagini curate pentru verificare finala.")
    lines.append("")
    lines.append("SUMMARY:")
    lines.append(f"original_top2_pixels={original_pixels}")
    lines.append(f"final_clean_mask_pixels={final_pixels}")
    lines.append(f"overlap_pixels={overlap_pixels}")
    lines.append(f"final_only_pixels={int(cv2.countNonZero(final_only))}")
    lines.append(f"original_only_pixels={int(cv2.countNonZero(original_only))}")
    lines.append(
        f"final_polyline_only_pixels={int(cv2.countNonZero(final_polyline_only))}"
    )
    lines.append(
        f"step6_polyline_mask_pixels={int(cv2.countNonZero(step6_polyline_mask))}"
    )
    lines.append(
        f"step7_support_guided_mask_pixels={int(cv2.countNonZero(step7_support_guided_mask))}"
    )
    lines.append(
        f"support_added_pixels_step7={int(cv2.countNonZero(support_added_mask))}"
    )
    lines.append(
        f"removed_fragments_pixels_step8={int(cv2.countNonZero(removed_fragments_mask))}"
    )
    lines.append(f"final_thickness_px={final_thickness_px}")
    lines.append(f"valid_components={len(components)}")
    lines.append(f"rejected_components={len(rejected_components)}")
    lines.append(f"local_polylines={len(polylines)}")
    lines.append(f"gaps={len(gaps)}")
    lines.append(f"accepted_gaps={len([g for g in gaps if g.accepted])}")
    lines.append(f"rejected_gaps={len([g for g in gaps if not g.accepted])}")
    lines.append(f"raw_connected_segments={len(connected_segments)}")
    lines.append(f"smoothed_segments={len(smoothed_segments)}")
    lines.append(f"cleaned_segments={len(cleaned_segments)}")
    lines.append(f"dip_cleanups={len(dip_cleanups)}")
    lines.append(f"clean_total_components={clean_info['total_components']}")
    lines.append(f"clean_kept_components={clean_info['kept_components']}")
    lines.append(f"clean_removed_components={clean_info['removed_components']}")
    lines.append(f"clean_kept_pixels={clean_info['kept_pixels']}")
    lines.append(f"clean_removed_pixels={clean_info['removed_pixels']}")
    lines.append("")

    lines.append("GAPS:")
    if len(gaps) == 0:
        lines.append("none")

    for gap in gaps:
        lines.append(
            f"G{gap.index}: "
            f"P{gap.left_component_order}->P{gap.right_component_order} "
            f"gap_px={gap.gap_px} "
            f"dx_px={gap.dx_px} "
            f"dy_px={gap.dy_px} "
            f"accepted={gap.accepted} "
            f"classification={gap.classification} "
            f"left={gap.left_endpoint} "
            f"right={gap.right_endpoint}"
        )

    lines.append("")
    lines.append("FINAL CLEANED SEGMENTS:")
    if len(cleaned_segments) == 0:
        lines.append("none")

    for segment in cleaned_segments:
        lines.append(
            f"S{segment.segment_index}: "
            f"points={segment.point_count} "
            f"x=[{segment.x_min},{segment.x_max}] "
            f"y=[{segment.y_min},{segment.y_max}] "
            f"start={tuple(segment.points[0])} "
            f"end={tuple(segment.points[-1])}"
        )

    return "\n".join(lines)


def _component_to_local_bottom_polyline(
    component: ComponentInfo,
    component_order: int,
) -> Optional[LocalPolylineInfo]:
    """
    Polyline locala de JOS:
        pentru fiecare coloana x din componenta,
        luam pixelul alb cel mai de jos.
    """

    xs_all: List[int] = []
    ys_all: List[int] = []

    for x in range(component.x_min, component.x_max + 1):
        ys = np.where(component.mask[:, x] > 0)[0]

        if len(ys) == 0:
            continue

        xs_all.append(int(x))
        ys_all.append(int(ys.max()))

    if len(xs_all) < 2:
        return None

    xs = np.array(xs_all, dtype=np.int32)
    ys = np.array(ys_all, dtype=np.int32)

    ys = _rolling_median(ys, LOCAL_SMOOTH_WINDOW_PX)
    ys = np.clip(ys, 0, component.mask.shape[0] - 1).astype(np.int32)

    points = np.stack([xs, ys], axis=1).astype(np.int32)

    start_point = (int(points[0, 0]), int(points[0, 1]))
    end_point = (int(points[-1, 0]), int(points[-1, 1]))

    return LocalPolylineInfo(
        component_order=component_order,
        component_label=component.label,
        point_count=len(points),
        x_min=int(points[:, 0].min()),
        x_max=int(points[:, 0].max()),
        y_min=int(points[:, 1].min()),
        y_max=int(points[:, 1].max()),
        start_point=start_point,
        end_point=end_point,
        points=points,
    )


def _build_local_bottom_polylines(
    components: List[ComponentInfo],
) -> List[LocalPolylineInfo]:
    bottom_polylines: List[LocalPolylineInfo] = []

    for idx, component in enumerate(components, start=1):
        polyline = _component_to_local_bottom_polyline(
            component=component,
            component_order=idx,
        )

        if polyline is not None:
            bottom_polylines.append(polyline)

    bottom_polylines.sort(key=lambda p: (p.x_min, p.start_point[1]))

    return bottom_polylines


def _identify_bottom_polyline_gaps(
    bottom_polylines: List[LocalPolylineInfo],
    top_gaps: List[PolylineGapInfo],
) -> List[PolylineGapInfo]:
    """
    Creeaza gap-uri pentru polyline-ul de jos.

    Acceptarea ramane cea calculata pe polyline-ul de sus,
    ca sa pastram aceeasi logica de unificare.
    """

    bottom_gaps: List[PolylineGapInfo] = []

    if len(bottom_polylines) < 2:
        return bottom_gaps

    for idx, (left, right) in enumerate(
        zip(bottom_polylines[:-1], bottom_polylines[1:]),
        start=1,
    ):
        left_endpoint = left.end_point
        right_endpoint = right.start_point

        gap_px = int(right_endpoint[0] - left_endpoint[0] - 1)
        dx_px = int(right_endpoint[0] - left_endpoint[0])
        dy_px = int(abs(right_endpoint[1] - left_endpoint[1]))

        accepted = False
        classification = "bottom_gap_without_top_pair"

        if idx - 1 < len(top_gaps):
            accepted = bool(top_gaps[idx - 1].accepted)
            classification = top_gaps[idx - 1].classification

        bottom_gaps.append(
            PolylineGapInfo(
                index=idx,
                left_component_order=left.component_order,
                right_component_order=right.component_order,
                left_component_label=left.component_label,
                right_component_label=right.component_label,
                gap_px=gap_px,
                dx_px=dx_px,
                dy_px=dy_px,
                left_endpoint=left_endpoint,
                right_endpoint=right_endpoint,
                classification=classification,
                accepted=accepted,
            )
        )

    return bottom_gaps


def _draw_polylines_only_on_crop(
    base_bgr: np.ndarray,
    top_segments: List[ConnectedSegmentInfo],
    bottom_segments: List[ConnectedSegmentInfo],
) -> np.ndarray:
    """
    Imagine curata:
        - crop original;
        - doar polyline sus si polyline jos;
        - fara contururi, fara text, fara puncte, fara bounding boxes.
    """

    out = base_bgr.copy()

    # Polyline sus: rosu.
    for segment in top_segments:
        if len(segment.points) < 2:
            continue

        cv2.polylines(
            out,
            [segment.points.reshape((-1, 1, 2))],
            False,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # Polyline jos: verde.
    for segment in bottom_segments:
        if len(segment.points) < 2:
            continue

        cv2.polylines(
            out,
            [segment.points.reshape((-1, 1, 2))],
            False,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return out


def _draw_special_final_thin_contour_on_crop(
    base_bgr: np.ndarray,
    cleaned_segments: List[ConnectedSegmentInfo],
) -> np.ndarray:
    """
    Imagine curata:
        - crop original;
        - doar conturul final unificat;
        - fara text, fara etichete, fara bounding boxes, fara masca groasa.
    """

    out = base_bgr.copy()

    for segment in cleaned_segments:
        if len(segment.points) < 2:
            continue

        cv2.polylines(
            out,
            [segment.points.reshape((-1, 1, 2))],
            False,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def _draw_special_components_before_unification_on_crop(
    base_bgr: np.ndarray,
    components: List[ComponentInfo],
    polylines: List[LocalPolylineInfo],
) -> np.ndarray:
    """
    Imagine curata inainte de unificare:
        - crop original;
        - doar contururile/linile componentelor separate;
        - fara text, fara etichete, fara bounding boxes, fara umplere.
    """

    out = base_bgr.copy()

    component_colors = [
        (0, 0, 255),
        (0, 180, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
        (255, 255, 0),
        (180, 255, 0),
        (255, 180, 0),
    ]

    # Doar contururile componentelor, fara scris si fara dreptunghiuri.
    for idx, component in enumerate(components, start=1):
        color = component_colors[(idx - 1) % len(component_colors)]

        contours, _hierarchy = cv2.findContours(
            component.mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            out,
            contours,
            -1,
            color,
            1,
            cv2.LINE_AA,
        )

    # Doar linia locala a fiecarei componente, fara conexiuni intre componente.
    for idx, polyline in enumerate(polylines, start=1):
        color = component_colors[(idx - 1) % len(component_colors)]

        cv2.polylines(
            out,
            [polyline.points.reshape((-1, 1, 2))],
            False,
            color,
            1,
            cv2.LINE_AA,
        )

    return out


def _component_to_local_middle_polyline(
    component: ComponentInfo,
    component_order: int,
) -> Optional[LocalPolylineInfo]:
    """
    Polyline locala pe MIJLOCUL pleurei.

    Pentru fiecare coloana x:
        top = cel mai de sus pixel alb din componenta;
        bottom = cel mai de jos pixel alb din componenta;
        middle = (top + bottom) / 2.

    Rezultatul este axa mediana a pleurei.
    """

    xs_all: List[int] = []
    ys_all: List[int] = []

    for x in range(component.x_min, component.x_max + 1):
        ys = np.where(component.mask[:, x] > 0)[0]

        if len(ys) == 0:
            continue

        y_top = int(ys.min())
        y_bottom = int(ys.max())
        y_middle = int(round((y_top + y_bottom) / 2.0))

        xs_all.append(int(x))
        ys_all.append(y_middle)

    if len(xs_all) < 2:
        return None

    xs = np.array(xs_all, dtype=np.int32)
    ys = np.array(ys_all, dtype=np.int32)

    ys = _rolling_median(ys, LOCAL_SMOOTH_WINDOW_PX)
    ys = np.clip(ys, 0, component.mask.shape[0] - 1).astype(np.int32)

    points = np.stack([xs, ys], axis=1).astype(np.int32)

    start_point = (int(points[0, 0]), int(points[0, 1]))
    end_point = (int(points[-1, 0]), int(points[-1, 1]))

    return LocalPolylineInfo(
        component_order=component_order,
        component_label=component.label,
        point_count=len(points),
        x_min=int(points[:, 0].min()),
        x_max=int(points[:, 0].max()),
        y_min=int(points[:, 1].min()),
        y_max=int(points[:, 1].max()),
        start_point=start_point,
        end_point=end_point,
        points=points,
    )


def _build_local_middle_polylines(
    components: List[ComponentInfo],
) -> List[LocalPolylineInfo]:
    middle_polylines: List[LocalPolylineInfo] = []

    for idx, component in enumerate(components, start=1):
        polyline = _component_to_local_middle_polyline(
            component=component,
            component_order=idx,
        )

        if polyline is not None:
            middle_polylines.append(polyline)

    middle_polylines.sort(key=lambda p: (p.x_min, p.start_point[1]))

    return middle_polylines


def _draw_middle_polyline_on_crop(
    base_bgr: np.ndarray,
    middle_segments: List[ConnectedSegmentInfo],
) -> np.ndarray:
    """
    Imagine curata:
        - crop original;
        - doar polyline-ul median al pleurei;
        - fara contur complet, fara text, fara puncte, fara bounding boxes.
    """

    out = base_bgr.copy()

    for segment in middle_segments:
        if len(segment.points) < 2:
            continue

        cv2.polylines(
            out,
            [segment.points.reshape((-1, 1, 2))],
            False,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def _draw_contour_from_middle_polyline_on_crop(
    base_bgr: np.ndarray,
    middle_segments: List[ConnectedSegmentInfo],
    final_thickness_px: int,
) -> np.ndarray:
    """
    Creeaza contur complet pe baza polyline-ului median.

    Regula:
        - middle polyline este axa centrala;
        - construim o masca subtire/groasa in jurul ei;
        - grosimea mastii este estimata din grosimea componentelor;
        - apoi extragem conturul complet al acelei masti.

    Rezultat:
        - doar conturul exterior complet;
        - fara text, fara puncte, fara bounding boxes.
    """

    out = base_bgr.copy()

    h, w = base_bgr.shape[:2]
    middle_mask = np.zeros((h, w), dtype=np.uint8)

    contour_thickness = int(
        max(
            final_thickness_px,
            FINAL_MASK_DEFAULT_THICKNESS_PX,
        )
    )

    contour_thickness = int(
        min(
            max(contour_thickness, FINAL_MASK_MIN_THICKNESS_PX),
            FINAL_MASK_MAX_THICKNESS_PX,
        )
    )

    for segment in middle_segments:
        if len(segment.points) < 2:
            continue

        cv2.polylines(
            middle_mask,
            [segment.points.reshape((-1, 1, 2))],
            False,
            255,
            contour_thickness,
            cv2.LINE_AA,
        )

    # O mica dilatare face conturul sa fie inchis si stabil,
    # dar ramane centrat pe polyline-ul median.
    if FINAL_MASK_DILATE_ITERATIONS > 0:
        kernel_size = max(3, contour_thickness)

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        middle_mask = cv2.dilate(
            middle_mask,
            kernel,
            iterations=FINAL_MASK_DILATE_ITERATIONS,
        )

    middle_mask = np.where(middle_mask > 0, 255, 0).astype(np.uint8)

    contours, _hierarchy = cv2.findContours(
        middle_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
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


def _identify_middle_gaps_connect_all_components(
    middle_polylines: List[LocalPolylineInfo],
) -> List[PolylineGapInfo]:
    """
    Creeaza gap-uri pentru polyline-ul median si le accepta pe TOATE.

    Folosim asta doar pentru imaginea finala:
        final_mask = original_final_mask + bridge_median_in_gapuri

    Ideea:
        - componentele existente raman cu conturul original;
        - toate componentele consecutive sunt unite prin bridge;
        - nu mai respingem gap-ul dupa dy/gap_px, pentru ca vrem unificare completa.
    """

    gaps: List[PolylineGapInfo] = []

    if len(middle_polylines) < 2:
        return gaps

    for idx, (left, right) in enumerate(
        zip(middle_polylines[:-1], middle_polylines[1:]),
        start=1,
    ):
        left_endpoint = left.end_point
        right_endpoint = right.start_point

        gap_px = int(right_endpoint[0] - left_endpoint[0] - 1)
        dx_px = int(right_endpoint[0] - left_endpoint[0])
        dy_px = int(abs(right_endpoint[1] - left_endpoint[1]))

        gaps.append(
            PolylineGapInfo(
                index=idx,
                left_component_order=left.component_order,
                right_component_order=right.component_order,
                left_component_label=left.component_label,
                right_component_label=right.component_label,
                gap_px=gap_px,
                dx_px=dx_px,
                dy_px=dy_px,
                left_endpoint=left_endpoint,
                right_endpoint=right_endpoint,
                classification="forced_connect_all_components",
                accepted=True,
            )
        )

    return gaps


def _find_exact_middle_polyline_points_for_gap(
    middle_segments: List[ConnectedSegmentInfo],
    gap: PolylineGapInfo,
) -> np.ndarray:
    """
    Intoarce punctele din middle polyline strict pentru zona gap-ului.

    Important:
        - nu reconstruim toata pleura;
        - nu inlocuim conturul componentelor existente;
        - middle polyline este folosita doar unde lipseste pleura, adica intre
          endpoint-urile gap-ului acceptat.
    """

    x1, y1 = gap.left_endpoint
    x2, y2 = gap.right_endpoint

    if x2 <= x1:
        return np.empty((0, 2), dtype=np.int32)

    best_points = np.empty((0, 2), dtype=np.int32)

    for segment in middle_segments:
        points = segment.points

        if len(points) < 2:
            continue

        inside = points[(points[:, 0] >= x1) & (points[:, 0] <= x2)]

        if len(inside) > len(best_points):
            best_points = inside.copy()

    if len(best_points) >= 2:
        best_points = best_points[np.argsort(best_points[:, 0])]

        if best_points[0, 0] > x1:
            best_points = np.vstack(
                [
                    np.array([[x1, y1]], dtype=np.int32),
                    best_points,
                ]
            )

        if best_points[-1, 0] < x2:
            best_points = np.vstack(
                [
                    best_points,
                    np.array([[x2, y2]], dtype=np.int32),
                ]
            )

        return best_points.astype(np.int32)

    # fallback: doar pentru gap, nu pentru componentele existente
    xs = np.arange(x1, x2 + 1, dtype=np.int32)

    if len(xs) < 2:
        return np.empty((0, 2), dtype=np.int32)

    ys = np.interp(xs, [x1, x2], [y1, y2]).astype(np.int32)

    return np.stack([xs, ys], axis=1).astype(np.int32)


def _build_middle_bridge_mask_only_in_gaps(
    shape: Tuple[int, int],
    middle_segments: List[ConnectedSegmentInfo],
    middle_gaps: List[PolylineGapInfo],
    final_thickness_px: int,
) -> np.ndarray:
    """
    Creeaza doar bridge-ul din zonele lipsa.

    Regula finala:
        final_mask = original_final_mask + bridge_mask

    Asta inseamna:
        - unde exista pleura, ramane conturul original;
        - unde lipseste pleura, completam cu o zona construita pe middle polyline;
        - nu umflam si nu redesenam componentele vechi.
    """

    h, w = shape[:2]
    bridge_mask = np.zeros((h, w), dtype=np.uint8)

    bridge_thickness = int(
        max(
            final_thickness_px,
            FINAL_MASK_DEFAULT_THICKNESS_PX,
        )
    )

    # Ingrosam DOAR bridge-ul, nu componentele originale.
    bridge_thickness = int(bridge_thickness + MIDDLE_BRIDGE_EXTRA_THICKNESS_PX)

    bridge_thickness = int(
        min(
            max(bridge_thickness, FINAL_MASK_MIN_THICKNESS_PX),
            MIDDLE_BRIDGE_MAX_THICKNESS_PX,
        )
    )

    for gap in middle_gaps:
        if not gap.accepted:
            continue

        middle_points = _find_exact_middle_polyline_points_for_gap(
            middle_segments,
            gap,
        )

        if len(middle_points) < 2:
            continue

        middle_points = middle_points[np.argsort(middle_points[:, 0])]
        middle_points[:, 0] = np.clip(middle_points[:, 0], 0, w - 1)
        middle_points[:, 1] = np.clip(middle_points[:, 1], 0, h - 1)

        cv2.polylines(
            bridge_mask,
            [middle_points.reshape((-1, 1, 2))],
            False,
            255,
            bridge_thickness,
            cv2.LINE_AA,
        )

    bridge_mask = np.where(bridge_mask > 0, 255, 0).astype(np.uint8)

    return bridge_mask


def _draw_original_contour_with_middle_bridge_on_crop(
    base_bgr: np.ndarray,
    original_final_mask: np.ndarray,
    middle_segments: List[ConnectedSegmentInfo],
    middle_gaps: List[PolylineGapInfo],
    final_thickness_px: int,
) -> np.ndarray:
    """
    Imagine finala ceruta:
        - foloseste conturul original unde exista pleura;
        - foloseste middle polyline in toate zonele lipsa dintre componentele consecutive;
        - extrage conturul complet al mastii:
              original_final_mask + bridge_middle_polyline
        - fara text, fara report, fara bounding boxes.
    """

    out = base_bgr.copy()

    original_mask = _to_binary_mask(original_final_mask)

    bridge_mask = _build_middle_bridge_mask_only_in_gaps(
        shape=original_mask.shape,
        middle_segments=middle_segments,
        middle_gaps=middle_gaps,
        final_thickness_px=final_thickness_px,
    )

    final_mask = original_mask.copy()
    final_mask[bridge_mask > 0] = 255

    contours, _hierarchy = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
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


def _draw_forced_middle_polyline_only_on_crop(
    base_bgr: np.ndarray,
    middle_segments: List[ConnectedSegmentInfo],
) -> np.ndarray:
    """
    Imagine curata:
        - crop original;
        - DOAR polyline-ul median;
        - polyline-ul este fortat conectat intre toate componentele consecutive;
        - fara contururi, fara masca, fara text, fara puncte, fara bounding boxes.
    """

    out = base_bgr.copy()

    for segment in middle_segments:
        if len(segment.points) < 2:
            continue

        cv2.polylines(
            out,
            [segment.points.reshape((-1, 1, 2))],
            False,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def _build_connector_any_direction(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
) -> np.ndarray:
    """
    Connector fara conditii:
        - merge si daca x2 <= x1;
        - merge si daca gap-ul e mare;
        - merge si daca saltul vertical e mare;
        - include ambele capete.
    """

    x1, y1 = p1
    x2, y2 = p2

    steps = int(max(abs(x2 - x1), abs(y2 - y1))) + 1

    if steps < 2:
        return np.array([[x1, y1], [x2, y2]], dtype=np.int32)

    xs = np.rint(np.linspace(x1, x2, steps)).astype(np.int32)
    ys = np.rint(np.linspace(y1, y2, steps)).astype(np.int32)

    return np.stack([xs, ys], axis=1).astype(np.int32)


def _build_forced_middle_chain_segments(
    middle_polylines: List[LocalPolylineInfo],
) -> List[ConnectedSegmentInfo]:
    """
    Construieste UN SINGUR lant median care uneste toate componentele consecutive.

    Nu exista conditii:
        - nu verificam gap_px;
        - nu verificam dy;
        - nu respingem nimic;
        - conectam fiecare componenta cu urmatoarea.
    """

    if len(middle_polylines) == 0:
        return []

    chunks: List[np.ndarray] = []

    first = middle_polylines[0].points

    if len(first) > 0:
        chunks.append(first.astype(np.int32))

    for left, right in zip(middle_polylines[:-1], middle_polylines[1:]):
        connector = _build_connector_any_direction(
            left.end_point,
            right.start_point,
        )

        if len(connector) > 0:
            chunks.append(connector.astype(np.int32))

        if len(right.points) > 0:
            chunks.append(right.points.astype(np.int32))

    if len(chunks) == 0:
        return []

    points = np.vstack(chunks).astype(np.int32)

    if len(points) < 2:
        return []

    return [
        ConnectedSegmentInfo(
            segment_index=1,
            point_count=len(points),
            x_min=int(points[:, 0].min()),
            x_max=int(points[:, 0].max()),
            y_min=int(points[:, 1].min()),
            y_max=int(points[:, 1].max()),
            points=points,
        )
    ]


def _sample_polyline_point_near_x(
    points: np.ndarray,
    target_x: int,
) -> Tuple[int, int]:
    """
    Ia un punct de pe polyline cat mai aproape de target_x.
    Folosit ca bridge-ul sa intre putin peste componenta,
    nu sa atinga componenta doar intr-un punct subtire.
    """

    if points is None or len(points) == 0:
        return (int(target_x), 0)

    xs = points[:, 0].astype(np.int32)
    idx = int(np.argmin(np.abs(xs - int(target_x))))

    return (int(points[idx, 0]), int(points[idx, 1]))


def _polyline_local_tangent(
    points: np.ndarray,
    near_start: bool,
    sample_px: int = 18,
) -> np.ndarray:
    """
    Estimeaza directia locala a unei componente.

    Pentru capatul din stanga folosim directia primelor puncte.
    Pentru capatul din dreapta folosim directia ultimelor puncte.
    """

    if points is None or len(points) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)

    pts = points[np.argsort(points[:, 0])].astype(np.float32)

    if near_start:
        p0 = pts[0]
        target_x = p0[0] + sample_px
        idx = int(np.argmin(np.abs(pts[:, 0] - target_x)))
        p1 = pts[max(1, idx)]
        vec = p1 - p0
    else:
        p0 = pts[-1]
        target_x = p0[0] - sample_px
        idx = int(np.argmin(np.abs(pts[:, 0] - target_x)))
        p1 = pts[min(len(pts) - 2, idx)]
        vec = p0 - p1

    norm = float(np.hypot(vec[0], vec[1]))

    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)

    return (vec / norm).astype(np.float32)


def _sample_cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    count: int,
) -> np.ndarray:
    """
    Eșantioneaza o curba Bezier cubica.
    """

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
    """
    Seed determinist pentru bridge.

    Nu folosim random global, ca rezultatul sa fie reproductibil.
    Seed-ul depinde de:
        - ordinea bridge-ului;
        - pozitia capetelor;
        - lungimea bridge-ului.
    """

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
    base_roughness: float,
    base_period: float,
    side_shift: float,
) -> np.ndarray:
    """
    Genereaza o crestare neregulata, nu perfect periodica.

    Combina mai multe unde cu amplitudini/faze diferite.
    """

    if len(cumulative) == 0:
        return np.zeros(0, dtype=np.float32)

    profile = np.zeros(len(cumulative), dtype=np.float32)

    harmonic_count = int(max(1, MIDDLE_BRIDGE_RANDOM_HARMONICS))

    for harmonic_idx in range(harmonic_count):
        period_factor = float(rng.uniform(0.65, 1.75))
        phase = float(rng.uniform(0.0, 2.0 * np.pi)) + float(side_shift)
        weight = float(rng.uniform(0.25, 1.0)) / float(harmonic_idx + 1)

        local_period = max(3.0, base_period * period_factor)

        profile += (
            weight * np.sin(2.0 * np.pi * cumulative / local_period + phase)
        ).astype(np.float32)

    max_abs = float(np.max(np.abs(profile))) if len(profile) > 0 else 0.0

    if max_abs > 1e-6:
        profile = profile / max_abs

    profile = profile * float(base_roughness)

    return profile.astype(np.float32)


def _ribbon_polygon_from_centerline(
    centerline: np.ndarray,
    thickness_px: int,
    shape: Tuple[int, int],
    rng: np.random.Generator,
    bridge_index: int,
    distance_px: float,
) -> np.ndarray:
    """
    Transforma linia centrala a bridge-ului intr-o banda cu margini crestate variabil.

    Diferenta fata de varianta sinusoidala fixa:
        - fiecare bridge are crestaturi diferite;
        - amplitudinea depinde de lungime si de grosime;
        - perioada este aleasa pseudo-random;
        - rezultatul ramane determinist pentru aceeasi imagine.
    """

    h, w = shape[:2]

    if centerline is None or len(centerline) < 2:
        return np.empty((0, 2), dtype=np.int32)

    half = float(max(1.0, thickness_px / 2.0))

    length_factor = float(np.clip(distance_px / 90.0, 0.65, 1.35))

    roughness_min = float(MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_MIN_PX)
    roughness_max = float(MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_MAX_PX)

    roughness = float(rng.uniform(roughness_min, roughness_max))
    roughness *= length_factor
    roughness = float(np.clip(roughness, roughness_min, roughness_max))

    # Crestarea nu trebuie sa fie prea mare, altfel bridge-ul devine zimtat artificial.
    # Limitare mai agresiva pentru pozele unde zig-zag-ul era prea vizibil.
    roughness = float(
        min(
            roughness,
            float(MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_HARD_CAP_PX),
            max(1.0, half * 0.45),
        )
    )

    period_min = float(MIDDLE_BRIDGE_RANDOM_PERIOD_MIN_PX)
    period_max = float(MIDDLE_BRIDGE_RANDOM_PERIOD_MAX_PX)

    base_period = float(rng.uniform(period_min, period_max))
    base_period *= float(rng.uniform(0.85, 1.25))

    left_side: List[List[int]] = []
    right_side: List[List[int]] = []

    cumulative = np.zeros(len(centerline), dtype=np.float32)

    for idx in range(1, len(centerline)):
        delta = centerline[idx] - centerline[idx - 1]
        cumulative[idx] = cumulative[idx - 1] + float(np.hypot(delta[0], delta[1]))

    left_profile = _random_jagged_profile(
        cumulative,
        rng,
        roughness,
        base_period,
        side_shift=float(rng.uniform(0.0, 2.0 * np.pi)),
    )

    right_profile = _random_jagged_profile(
        cumulative,
        rng,
        roughness * float(rng.uniform(0.75, 1.15)),
        base_period * float(rng.uniform(0.75, 1.35)),
        side_shift=float(rng.uniform(0.0, 2.0 * np.pi)),
    )

    for idx in range(len(centerline)):
        if idx == 0:
            direction = centerline[1] - centerline[0]
        elif idx == len(centerline) - 1:
            direction = centerline[-1] - centerline[-2]
        else:
            direction = centerline[idx + 1] - centerline[idx - 1]

        norm = float(np.hypot(direction[0], direction[1]))

        if norm < 1e-6:
            normal = np.array([0.0, 1.0], dtype=np.float32)
        else:
            direction = direction / norm
            normal = np.array([-direction[1], direction[0]], dtype=np.float32)

        # La capete reducem crestarea ca sa se lipeasca mai natural de componenta.
        edge_fade = min(idx, len(centerline) - 1 - idx)
        fade = float(np.clip(edge_fade / 5.0, 0.0, 1.0))

        jag_left = float(left_profile[idx]) * fade
        jag_right = float(right_profile[idx]) * fade

        p = centerline[idx]

        left_half = max(1.0, half + jag_left)
        right_half = max(1.0, half + jag_right)

        p_left = p + normal * left_half
        p_right = p - normal * right_half

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

    polygon = np.array(left_side + right_side[::-1], dtype=np.int32)

    return polygon


def _draw_natural_bridge_between_components(
    bridge_mask: np.ndarray,
    left: LocalPolylineInfo,
    right: LocalPolylineInfo,
    bridge_thickness: int,
    bridge_index: int,
) -> None:
    """
    Bridge mai natural:
        - intra putin peste componente;
        - urmeaza o curba Bezier intre componente;
        - tangentele sunt luate din directia locala a componentelor;
        - crestarea marginilor este pseudo-random si diferita pentru fiecare bridge;
        - capetele raman plate, fara cercuri/bulbi.
    """

    h, w = bridge_mask.shape[:2]

    left_target_x = int(left.end_point[0] - MIDDLE_BRIDGE_ENDPOINT_OVERLAP_PX)
    right_target_x = int(right.start_point[0] + MIDDLE_BRIDGE_ENDPOINT_OVERLAP_PX)

    left_target_x = int(np.clip(left_target_x, left.x_min, left.x_max))
    right_target_x = int(np.clip(right_target_x, right.x_min, right.x_max))

    p0_tuple = _sample_polyline_point_near_x(
        left.points,
        left_target_x,
    )
    p3_tuple = _sample_polyline_point_near_x(
        right.points,
        right_target_x,
    )

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

    seed = _stable_bridge_seed(
        bridge_index,
        p0,
        p3,
        distance,
    )

    rng = np.random.default_rng(seed)

    tangent_length = int(round(distance * MIDDLE_BRIDGE_TANGENT_SCALE))
    tangent_length = int(
        np.clip(
            tangent_length,
            MIDDLE_BRIDGE_TANGENT_MIN_PX,
            MIDDLE_BRIDGE_TANGENT_MAX_PX,
        )
    )

    # Mica variatie si pe tangenta, ca bridge-urile sa nu fie toate identice.
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

    # O mica deplasare perpendiculara, diferita pe fiecare bridge,
    # ca sa nu fie toate curbele prea uniforme.
    global_direction = p3 - p0
    global_norm = float(np.hypot(global_direction[0], global_direction[1]))

    if global_norm > 1e-6:
        global_direction = global_direction / global_norm
        global_normal = np.array(
            [-global_direction[1], global_direction[0]],
            dtype=np.float32,
        )

        bend_offset = float(
            rng.uniform(
                -MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_MAX_PX,
                MIDDLE_BRIDGE_RANDOM_JAGGEDNESS_MAX_PX,
            )
        )

        p1 = p1 + global_normal * bend_offset
        p2 = p2 - global_normal * bend_offset * float(rng.uniform(0.5, 1.0))

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
        bridge_index,
        distance,
    )

    if len(polygon) < 3:
        return

    cv2.fillPoly(
        bridge_mask,
        [polygon.reshape((-1, 1, 2))],
        255,
        cv2.LINE_AA,
    )


def _draw_wide_bridge_between_components(
    bridge_mask: np.ndarray,
    left: LocalPolylineInfo,
    right: LocalPolylineInfo,
    bridge_thickness: int,
) -> None:
    """
    Deseneaza bridge-ul dintre doua componente fara sugrumare si fara terminatii rotunjite.

    Diferenta fata de varianta cu linie groasa si cercuri:
        - bridge-ul este un poligon cu capete plate;
        - intra putin peste componente;
        - nu mai apar bulbii/capetele rotunjite.
    """

    h, w = bridge_mask.shape[:2]

    left_target_x = int(left.end_point[0] - MIDDLE_BRIDGE_ENDPOINT_OVERLAP_PX)
    right_target_x = int(right.start_point[0] + MIDDLE_BRIDGE_ENDPOINT_OVERLAP_PX)

    left_target_x = int(np.clip(left_target_x, left.x_min, left.x_max))
    right_target_x = int(np.clip(right_target_x, right.x_min, right.x_max))

    p1 = _sample_polyline_point_near_x(
        left.points,
        left_target_x,
    )
    p2 = _sample_polyline_point_near_x(
        right.points,
        right_target_x,
    )

    x1 = float(np.clip(p1[0], 0, w - 1))
    y1 = float(np.clip(p1[1], 0, h - 1))
    x2 = float(np.clip(p2[0], 0, w - 1))
    y2 = float(np.clip(p2[1], 0, h - 1))

    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))

    if length < 1.0:
        return

    half_thickness = float(max(1.0, bridge_thickness / 2.0))

    # Vector perpendicular pe directia bridge-ului.
    nx = -dy / length
    ny = dx / length

    polygon = np.array(
        [
            [
                int(round(x1 + nx * half_thickness)),
                int(round(y1 + ny * half_thickness)),
            ],
            [
                int(round(x2 + nx * half_thickness)),
                int(round(y2 + ny * half_thickness)),
            ],
            [
                int(round(x2 - nx * half_thickness)),
                int(round(y2 - ny * half_thickness)),
            ],
            [
                int(round(x1 - nx * half_thickness)),
                int(round(y1 - ny * half_thickness)),
            ],
        ],
        dtype=np.int32,
    )

    polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

    cv2.fillConvexPoly(
        bridge_mask,
        polygon.reshape((-1, 1, 2)),
        255,
        cv2.LINE_AA,
    )


def _build_middle_bridge_mask_connect_everything(
    shape: Tuple[int, int],
    middle_polylines: List[LocalPolylineInfo],
    final_thickness_px: int,
) -> np.ndarray:
    """
    Bridge fara conditii, mai natural, fara sugrumari si cu margini pseudo-random.

    Regula:
        - conectam fiecare componenta cu urmatoarea;
        - nu verificam gap/dy/distanta;
        - bridge-ul intra putin peste componente;
        - capetele sunt plate;
        - close local minim pe bridge ca sa dispara micile gauri.
    """

    h, w = shape[:2]
    bridge_mask = np.zeros((h, w), dtype=np.uint8)

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
        _draw_natural_bridge_between_components(
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

    bridge_mask = np.where(bridge_mask > 0, 255, 0).astype(np.uint8)

    return bridge_mask


def _draw_contour_with_middle_polyline_on_crop(
    base_bgr: np.ndarray,
    original_final_mask: np.ndarray,
    middle_polylines: List[LocalPolylineInfo],
    forced_middle_segments: List[ConnectedSegmentInfo],
    final_thickness_px: int,
) -> np.ndarray:
    """
    Imagine curata:
        - crop original;
        - DOAR contur final;
        - contur final = contur original + bridge intre TOATE componentele;
        - fara polyline, fara text, fara report, fara bounding boxes.
    """

    out = base_bgr.copy()

    original_mask = _to_binary_mask(original_final_mask)

    bridge_mask = _build_middle_bridge_mask_connect_everything(
        shape=original_mask.shape,
        middle_polylines=middle_polylines,
        final_thickness_px=final_thickness_px,
    )

    final_mask = original_mask.copy()
    final_mask[bridge_mask > 0] = 255

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


def build_top2_unification_debug(
    crop_bgr: np.ndarray,
    top2_final_mask: np.ndarray,
    support_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Functia publica ramane compatibila cu main.py.

    Pasul 9:
        masca finala curatata -> output final simplu.

    support_mask:
        acceptat pentru compatibilitate;
        este folosit local in pasul 7, rezultatul este curatat in pasul 8, iar in pasul 9 se genereaza output-ul final.
    """

    base_bgr = _to_bgr(crop_bgr)
    original_final_mask = _to_binary_mask(top2_final_mask)

    components, rejected_components = _extract_components(original_final_mask)
    polylines = _build_local_polylines(components)
    gaps = _identify_polyline_gaps(polylines)

    connected_segments = _build_connected_segments(polylines, gaps)
    smoothed_segments = _smooth_connected_segments(connected_segments)
    cleaned_segments, dip_cleanups = _clean_redundant_dips(smoothed_segments)

    # Polyline median: contur/axa pe mijlocul pleurei.
    middle_polylines = _build_local_middle_polylines(components)
    middle_gaps = _identify_middle_gaps_connect_all_components(middle_polylines)
    middle_connected_segments = _build_connected_segments(middle_polylines, middle_gaps)
    middle_smoothed_segments = _smooth_connected_segments(middle_connected_segments)
    middle_cleaned_segments, middle_dip_cleanups = _clean_redundant_dips(
        middle_smoothed_segments
    )

    # Lant median fortat: uneste toate componentele fara nicio conditie.
    forced_middle_segments = _build_forced_middle_chain_segments(middle_polylines)

    bottom_polylines = _build_local_bottom_polylines(components)
    bottom_gaps = _identify_bottom_polyline_gaps(bottom_polylines, gaps)
    bottom_connected_segments = _build_connected_segments(bottom_polylines, bottom_gaps)
    bottom_smoothed_segments = _smooth_connected_segments(bottom_connected_segments)

    final_thickness_px = _estimate_final_thickness(components)

    step6_polyline_mask = _draw_final_polyline_mask(
        original_final_mask.shape,
        cleaned_segments,
        final_thickness_px,
    )

    step7_support_guided_mask, support_added_mask, support_band_mask = (
        _support_guided_refine_mask(
            original_final_mask=original_final_mask,
            support_mask=support_mask,
            final_polyline_mask=step6_polyline_mask,
            final_thickness_px=final_thickness_px,
        )
    )

    final_clean_mask, removed_fragments_mask, clean_info = (
        _final_clean_keep_core_connected(
            step6_polyline_mask=step6_polyline_mask,
            support_guided_mask=step7_support_guided_mask,
        )
    )

    final_polyline_only = _draw_final_polyline_only_mask(
        original_final_mask.shape,
        cleaned_segments,
    )

    final_output_on_crop = _draw_step9_final_output(
        base_bgr=base_bgr,
        final_clean_mask=final_clean_mask,
        final_polyline_only=final_polyline_only,
    )

    before_after_overlay = _draw_before_after_overlay(
        base_bgr=base_bgr,
        original_final_mask=original_final_mask,
        final_clean_mask=final_clean_mask,
        final_polyline_only=final_polyline_only,
    )

    report_text = _build_report_step9(
        original_final_mask=original_final_mask,
        step6_polyline_mask=step6_polyline_mask,
        step7_support_guided_mask=step7_support_guided_mask,
        final_clean_mask=final_clean_mask,
        final_polyline_only=final_polyline_only,
        removed_fragments_mask=removed_fragments_mask,
        support_added_mask=support_added_mask,
        clean_info=clean_info,
        components=components,
        rejected_components=rejected_components,
        polylines=polylines,
        gaps=gaps,
        connected_segments=connected_segments,
        smoothed_segments=smoothed_segments,
        cleaned_segments=cleaned_segments,
        dip_cleanups=dip_cleanups,
        final_thickness_px=final_thickness_px,
    )

    return {
        "images": {
            "special_contour_with_middle_polyline_on_crop": _draw_contour_with_middle_polyline_on_crop(
                base_bgr,
                original_final_mask,
                middle_polylines,
                forced_middle_segments,
                final_thickness_px,
            ),
            "special_forced_middle_polyline_only_on_crop": _draw_forced_middle_polyline_only_on_crop(
                base_bgr,
                middle_cleaned_segments,
            ),
            "special_original_contour_middle_bridge_on_crop": _draw_original_contour_with_middle_bridge_on_crop(
                base_bgr,
                original_final_mask,
                middle_cleaned_segments,
                middle_gaps,
                final_thickness_px,
            ),
            "special_contour_from_middle_polyline_on_crop": _draw_contour_from_middle_polyline_on_crop(
                base_bgr,
                middle_cleaned_segments,
                final_thickness_px,
            ),
            "special_middle_polyline_on_crop": _draw_middle_polyline_on_crop(
                base_bgr,
                middle_cleaned_segments,
            ),
            "step9_final_output_on_crop": final_output_on_crop,
            "step9_final_mask_only": final_clean_mask,
            "step9_final_polyline_only": final_polyline_only,
            "step9_before_after_overlay": before_after_overlay,
            "special_final_thin_contour_on_crop": _draw_special_final_thin_contour_on_crop(
                base_bgr,
                cleaned_segments,
            ),
            "special_components_before_unification_on_crop": _draw_special_components_before_unification_on_crop(
                base_bgr,
                components,
                polylines,
            ),
            "special_polylines_only_on_crop": _draw_polylines_only_on_crop(
                base_bgr,
                cleaned_segments,
                bottom_smoothed_segments,
            ),
        },
        "report_text": report_text,
        # In pasul 9 unified_mask ramane masca finala curatata.
        "unified_mask": final_clean_mask,
        "step6_polyline_mask": step6_polyline_mask,
        "step7_support_guided_mask": step7_support_guided_mask,
        "support_added_mask": support_added_mask,
        "support_band_mask": support_band_mask,
        "final_clean_mask": final_clean_mask,
        "final_polyline_only": final_polyline_only,
        "removed_fragments_mask": removed_fragments_mask,
        "clean_info": clean_info,
        "final_thickness_px": final_thickness_px,
        "components": components,
        "rejected_components": rejected_components,
        "local_polylines": polylines,
        "middle_polylines": middle_polylines,
        "middle_polyline_gaps": middle_gaps,
        "middle_connected_segments": middle_connected_segments,
        "middle_smoothed_segments": middle_smoothed_segments,
        "middle_cleaned_segments": middle_cleaned_segments,
        "forced_middle_segments": forced_middle_segments,
        "middle_dip_cleanups": middle_dip_cleanups,
        "bottom_polylines": bottom_polylines,
        "polyline_gaps": gaps,
        "bottom_polyline_gaps": bottom_gaps,
        "gaps": gaps,
        "connected_segments": connected_segments,
        "bottom_connected_segments": bottom_connected_segments,
        "bottom_smoothed_segments": bottom_smoothed_segments,
        "smoothed_segments": smoothed_segments,
        "cleaned_segments": cleaned_segments,
        "dip_cleanups": dip_cleanups,
        # Compatibilitate cu codul vechi.
        "bridge_mask": final_clean_mask,
        "chain_bridge_mask": final_clean_mask,
        "terminal_mask": np.zeros_like(original_final_mask),
        "added_bridge_mask": final_clean_mask,
        "added_terminal_mask": np.zeros_like(original_final_mask),
        "bridge_infos": [],
        "terminal_infos": [],
    }
