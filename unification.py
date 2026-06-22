from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

MIN_COMPONENT_AREA_PX = 20
MIN_COMPONENT_WIDTH_PX = 3
MIN_COMPONENT_HEIGHT_PX = 2

ANCHOR_WINDOW_PX = 18
BRIDGE_OVERLAP_PX = 8

THIN_BRIDGE_THICKNESS_PX = 4
MIN_THIN_BRIDGE_THICKNESS_PX = 3
MAX_THIN_BRIDGE_THICKNESS_PX = 6

TOP_OFFSET_RATIO = 0.30
MIN_TOP_OFFSET_PX = 1.0
MAX_TOP_OFFSET_PX = 4.0

MAX_CENTER_Y_JUMP_PX = 120
MAX_COMPONENT_GAP_PX = 10000

CONTOUR_THICKNESS_PX = 1

RIGHT_TERMINAL_ENABLE = True
RIGHT_TERMINAL_SEARCH_PX = 90
RIGHT_TERMINAL_MIN_TARGET_AREA_PX = 2
RIGHT_TERMINAL_SOURCE_MIN_AREA_PX = 20
RIGHT_TERMINAL_SOURCE_WINDOW_PX = 32
RIGHT_TERMINAL_TARGET_WINDOW_PX = 28
RIGHT_TERMINAL_MAX_DISTANCE_PX = 105.0
RIGHT_TERMINAL_MAX_DY_PX = 48.0
RIGHT_TERMINAL_MAX_BACKTRACK_PX = 14
RIGHT_TERMINAL_THICKNESS_PX = 4
RIGHT_TERMINAL_MIN_THICKNESS_PX = 3
RIGHT_TERMINAL_MAX_THICKNESS_PX = 5
RIGHT_TERMINAL_MAX_TARGETS = 4
RIGHT_TERMINAL_REJECT_BORDER_WIDTH_PX = 3
RIGHT_TERMINAL_REJECT_BORDER_HEIGHT_PX = 70


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
class AnchorProfile:
    xs: np.ndarray
    tops: np.ndarray
    bottoms: np.ndarray
    centers: np.ndarray
    thicknesses: np.ndarray
    bridge_ys: np.ndarray

    @property
    def valid(self) -> bool:
        return len(self.xs) > 0

    @property
    def x_median(self) -> float:
        return float(np.median(self.xs))

    @property
    def top_median(self) -> float:
        return float(np.median(self.tops))

    @property
    def bottom_median(self) -> float:
        return float(np.median(self.bottoms))

    @property
    def center_median(self) -> float:
        return float(np.median(self.centers))

    @property
    def thickness_median(self) -> float:
        return float(np.median(self.thicknesses))

    @property
    def bridge_y_median(self) -> float:
        return float(np.median(self.bridge_ys))


@dataclass
class BridgeInfo:
    left_label: int
    right_label: int
    accepted: bool
    reason: str
    gap_px: int
    bridge_y_diff: float
    bridge_area: int
    start_point: Optional[Tuple[int, int]]
    end_point: Optional[Tuple[int, int]]
    thickness: int


@dataclass
class TerminalInfo:
    source_label: int
    target_label: int
    accepted: bool
    reason: str
    distance_px: float
    dx_px: int
    dy_px: int
    bridge_area: int
    start_point: Optional[Tuple[int, int]]
    end_point: Optional[Tuple[int, int]]
    thickness: int


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


def _safe_int(value: float, low: int, high: int) -> int:
    return int(np.clip(int(round(value)), low, high))


def _top_run_thickness(mask: np.ndarray, x: int, y_top: int) -> int:
    h = mask.shape[0]
    y = int(y_top)

    while y + 1 < h and mask[y + 1, x] > 0:
        y += 1

    return max(1, y - int(y_top) + 1)


def _column_profile(mask: np.ndarray, x_start: int, x_end: int) -> AnchorProfile:
    h, w = mask.shape[:2]
    x_start = int(np.clip(x_start, 0, w - 1))
    x_end = int(np.clip(x_end, 0, w - 1))

    if x_end < x_start:
        x_start, x_end = x_end, x_start

    raw_rows: List[Tuple[int, int, int, float, int]] = []

    for x in range(x_start, x_end + 1):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) == 0:
            continue

        y_top = int(ys.min())
        y_bottom = int(ys.max())
        thickness = y_bottom - y_top + 1
        top_run = _top_run_thickness(mask, x, y_top)
        center = (y_top + y_bottom) / 2.0
        raw_rows.append((x, y_top, y_bottom, center, max(1, min(thickness, top_run + 6))))

    if len(raw_rows) == 0:
        return AnchorProfile(
            xs=np.array([], dtype=np.int32),
            tops=np.array([], dtype=np.float32),
            bottoms=np.array([], dtype=np.float32),
            centers=np.array([], dtype=np.float32),
            thicknesses=np.array([], dtype=np.float32),
            bridge_ys=np.array([], dtype=np.float32),
        )

    thickness_values = np.array([row[4] for row in raw_rows], dtype=np.float32)
    robust_thickness = float(np.percentile(thickness_values, 45))
    top_offset = float(np.clip(robust_thickness * TOP_OFFSET_RATIO, MIN_TOP_OFFSET_PX, MAX_TOP_OFFSET_PX))

    xs: List[int] = []
    tops: List[int] = []
    bottoms: List[int] = []
    centers: List[float] = []
    thicknesses: List[int] = []
    bridge_ys: List[float] = []

    for x, y_top, y_bottom, center, thickness in raw_rows:
        xs.append(x)
        tops.append(y_top)
        bottoms.append(y_bottom)
        centers.append(center)
        thicknesses.append(thickness)
        bridge_ys.append(float(np.clip(y_top + top_offset, 0, h - 1)))

    return AnchorProfile(
        xs=np.array(xs, dtype=np.int32),
        tops=np.array(tops, dtype=np.float32),
        bottoms=np.array(bottoms, dtype=np.float32),
        centers=np.array(centers, dtype=np.float32),
        thicknesses=np.array(thicknesses, dtype=np.float32),
        bridge_ys=np.array(bridge_ys, dtype=np.float32),
    )


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
        tops.append(y_top)
        bottoms.append(y_bottom)
        centers.append((y_top + y_bottom) / 2.0)
        thicknesses.append(thickness)

    return (
        float(np.median(tops)),
        float(np.median(bottoms)),
        float(np.median(centers)),
        float(np.median(thicknesses)),
    )


def _extract_components(mask: np.ndarray) -> Tuple[List[ComponentInfo], List[ComponentInfo]]:
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
        median_top, median_bottom, median_center_y, median_thickness = _component_profile(component_mask)

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


def _get_side_anchor(component: ComponentInfo, side: str) -> AnchorProfile:
    if side == "right":
        x_start = max(component.x_min, component.x_max - ANCHOR_WINDOW_PX + 1)
        x_end = component.x_max
    elif side == "left":
        x_start = component.x_min
        x_end = min(component.x_max, component.x_min + ANCHOR_WINDOW_PX - 1)
    else:
        raise ValueError(f"side invalid: {side}")

    profile = _column_profile(component.mask, x_start, x_end)

    if profile.valid:
        return profile

    return _column_profile(component.mask, component.x_min, component.x_max)


def _line_thickness(left_anchor: AnchorProfile, right_anchor: AnchorProfile) -> int:
    local_thickness = float(np.percentile(
        np.concatenate([left_anchor.thicknesses, right_anchor.thicknesses]),
        35,
    ))
    thickness = int(round(max(MIN_THIN_BRIDGE_THICKNESS_PX, min(THIN_BRIDGE_THICKNESS_PX, local_thickness))))
    return int(np.clip(thickness, MIN_THIN_BRIDGE_THICKNESS_PX, MAX_THIN_BRIDGE_THICKNESS_PX))


def _build_thin_centerline_bridge(
    shape: Tuple[int, int],
    left_component: ComponentInfo,
    right_component: ComponentInfo,
) -> Tuple[np.ndarray, BridgeInfo]:
    h, w = shape[:2]
    bridge_mask = np.zeros((h, w), dtype=np.uint8)

    left_anchor = _get_side_anchor(left_component, "right")
    right_anchor = _get_side_anchor(right_component, "left")

    gap_px = max(0, right_component.x_min - left_component.x_max - 1)

    if not left_anchor.valid or not right_anchor.valid:
        info = BridgeInfo(
            left_label=left_component.label,
            right_label=right_component.label,
            accepted=False,
            reason="anchor_invalid",
            gap_px=gap_px,
            bridge_y_diff=0.0,
            bridge_area=0,
            start_point=None,
            end_point=None,
            thickness=0,
        )
        return bridge_mask, info

    left_y = left_anchor.bridge_y_median
    right_y = right_anchor.bridge_y_median
    bridge_y_diff = float(abs(left_y - right_y))

    if gap_px > MAX_COMPONENT_GAP_PX:
        info = BridgeInfo(
            left_label=left_component.label,
            right_label=right_component.label,
            accepted=False,
            reason="gap_too_large",
            gap_px=gap_px,
            bridge_y_diff=bridge_y_diff,
            bridge_area=0,
            start_point=None,
            end_point=None,
            thickness=0,
        )
        return bridge_mask, info

    if bridge_y_diff > MAX_CENTER_Y_JUMP_PX:
        info = BridgeInfo(
            left_label=left_component.label,
            right_label=right_component.label,
            accepted=False,
            reason="bridge_y_jump_too_large",
            gap_px=gap_px,
            bridge_y_diff=bridge_y_diff,
            bridge_area=0,
            start_point=None,
            end_point=None,
            thickness=0,
        )
        return bridge_mask, info

    x_left = max(left_component.x_min, left_component.x_max - BRIDGE_OVERLAP_PX + 1)
    x_right = min(right_component.x_max, right_component.x_min + BRIDGE_OVERLAP_PX - 1)

    y_left = _safe_int(left_y, 0, h - 1)
    y_right = _safe_int(right_y, 0, h - 1)

    thickness = _line_thickness(left_anchor, right_anchor)

    cv2.line(
        bridge_mask,
        (int(x_left), int(y_left)),
        (int(x_right), int(y_right)),
        255,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    radius = max(1, thickness // 2)
    cv2.circle(bridge_mask, (int(x_left), int(y_left)), radius, 255, -1, lineType=cv2.LINE_AA)
    cv2.circle(bridge_mask, (int(x_right), int(y_right)), radius, 255, -1, lineType=cv2.LINE_AA)

    bridge_area = int(cv2.countNonZero(bridge_mask))
    info = BridgeInfo(
        left_label=left_component.label,
        right_label=right_component.label,
        accepted=True,
        reason="accepted",
        gap_px=gap_px,
        bridge_y_diff=bridge_y_diff,
        bridge_area=bridge_area,
        start_point=(int(x_left), int(y_left)),
        end_point=(int(x_right), int(y_right)),
        thickness=thickness,
    )
    return bridge_mask, info




def _all_components_for_terminal(components: List[ComponentInfo], rejected_components: List[ComponentInfo]) -> List[ComponentInfo]:
    all_components = list(components) + [c for c in rejected_components if c.area >= RIGHT_TERMINAL_MIN_TARGET_AREA_PX]
    all_components.sort(key=lambda c: (c.x_min, c.center_y, -c.area))
    return all_components


def _is_border_artifact(component: ComponentInfo, image_width: int) -> bool:
    touches_right_border = component.x_max >= image_width - 1
    too_vertical = (
        component.width <= RIGHT_TERMINAL_REJECT_BORDER_WIDTH_PX
        and component.height >= RIGHT_TERMINAL_REJECT_BORDER_HEIGHT_PX
    )
    return bool(touches_right_border and too_vertical)


def _endpoint_pixels(component: ComponentInfo, side: str, window_px: int) -> np.ndarray:
    ys, xs = np.where(component.mask > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    if side == "right":
        keep = xs >= max(component.x_min, component.x_max - window_px + 1)
    elif side == "left":
        keep = xs <= min(component.x_max, component.x_min + window_px - 1)
    else:
        raise ValueError(f"side invalid: {side}")

    xs = xs[keep]
    ys = ys[keep]
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    pts = np.column_stack([xs, ys]).astype(np.int32)

    if len(pts) > 900:
        step = int(np.ceil(len(pts) / 900.0))
        pts = pts[::step]

    return pts


def _find_nearest_terminal_pair(
    source_component: ComponentInfo,
    target_component: ComponentInfo,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], float, int, int]:
    source_pts = _endpoint_pixels(source_component, "right", RIGHT_TERMINAL_SOURCE_WINDOW_PX)
    target_pts = _endpoint_pixels(target_component, "left", RIGHT_TERMINAL_TARGET_WINDOW_PX)

    if len(source_pts) == 0 or len(target_pts) == 0:
        return None, None, 0.0, 0, 0

    best_source: Optional[Tuple[int, int]] = None
    best_target: Optional[Tuple[int, int]] = None
    best_score = float("inf")
    best_dist = float("inf")
    best_dx = 0
    best_dy = 0

    target_x = target_pts[:, 0].astype(np.float32)
    target_y = target_pts[:, 1].astype(np.float32)

    for sx, sy in source_pts:
        dx = target_x - float(sx)
        dy = target_y - float(sy)

        valid = (
            (dx >= -RIGHT_TERMINAL_MAX_BACKTRACK_PX)
            & (np.abs(dy) <= RIGHT_TERMINAL_MAX_DY_PX)
        )

        if not np.any(valid):
            continue

        d2 = dx * dx + dy * dy
        valid_indices = np.where(valid)[0]
        local_idx = valid_indices[int(np.argmin(d2[valid]))]

        local_dx = int(round(float(dx[local_idx])))
        local_dy = int(round(float(dy[local_idx])))
        local_dist = float(np.sqrt(float(d2[local_idx])))

        if local_dist > RIGHT_TERMINAL_MAX_DISTANCE_PX:
            continue

        score = local_dist + 0.22 * abs(local_dy) + 0.04 * max(0, local_dx)

        if score < best_score:
            best_score = score
            best_dist = local_dist
            best_dx = local_dx
            best_dy = local_dy
            best_source = (int(sx), int(sy))
            best_target = (int(target_pts[local_idx, 0]), int(target_pts[local_idx, 1]))

    if best_source is None or best_target is None:
        return None, None, 0.0, 0, 0

    return best_source, best_target, best_dist, best_dx, best_dy


def _terminal_line_thickness(source_component: ComponentInfo, target_component: ComponentInfo) -> int:
    values = [source_component.median_thickness, target_component.median_thickness]
    values = [v for v in values if v > 0]
    if len(values) == 0:
        return RIGHT_TERMINAL_THICKNESS_PX

    robust = int(round(float(np.percentile(np.array(values, dtype=np.float32), 35))))
    thickness = min(RIGHT_TERMINAL_THICKNESS_PX, robust)
    return int(np.clip(thickness, RIGHT_TERMINAL_MIN_THICKNESS_PX, RIGHT_TERMINAL_MAX_THICKNESS_PX))


def _build_single_terminal_bridge(
    shape: Tuple[int, int],
    source_component: ComponentInfo,
    target_component: ComponentInfo,
) -> Tuple[np.ndarray, TerminalInfo]:
    h, w = shape[:2]
    bridge_mask = np.zeros((h, w), dtype=np.uint8)

    start, end, distance_px, dx_px, dy_px = _find_nearest_terminal_pair(source_component, target_component)

    if start is None or end is None:
        info = TerminalInfo(
            source_label=source_component.label,
            target_label=target_component.label,
            accepted=False,
            reason="no_valid_endpoint_pair",
            distance_px=0.0,
            dx_px=0,
            dy_px=0,
            bridge_area=0,
            start_point=None,
            end_point=None,
            thickness=0,
        )
        return bridge_mask, info

    thickness = _terminal_line_thickness(source_component, target_component)

    cv2.line(bridge_mask, start, end, 255, thickness=thickness, lineType=cv2.LINE_AA)
    radius = max(1, thickness // 2)
    cv2.circle(bridge_mask, start, radius, 255, -1, lineType=cv2.LINE_AA)
    cv2.circle(bridge_mask, end, radius, 255, -1, lineType=cv2.LINE_AA)

    bridge_area = int(cv2.countNonZero(bridge_mask))
    info = TerminalInfo(
        source_label=source_component.label,
        target_label=target_component.label,
        accepted=True,
        reason="accepted_terminal_endpoint_rescue",
        distance_px=distance_px,
        dx_px=dx_px,
        dy_px=dy_px,
        bridge_area=bridge_area,
        start_point=start,
        end_point=end,
        thickness=thickness,
    )
    return bridge_mask, info


def _build_right_terminal_rescue(
    shape: Tuple[int, int],
    components: List[ComponentInfo],
    rejected_components: List[ComponentInfo],
) -> Tuple[np.ndarray, List[TerminalInfo]]:
    h, w = shape[:2]
    terminal_mask = np.zeros((h, w), dtype=np.uint8)
    terminal_infos: List[TerminalInfo] = []

    if not RIGHT_TERMINAL_ENABLE or len(components) == 0:
        return terminal_mask, terminal_infos

    all_components = _all_components_for_terminal(components, rejected_components)
    if len(all_components) < 2:
        return terminal_mask, terminal_infos

    main_component = max(components, key=lambda c: (c.area, c.width))
    min_terminal_x = max(0, w - RIGHT_TERMINAL_SEARCH_PX)

    target_candidates: List[ComponentInfo] = []
    for component in all_components:
        if component.label == main_component.label:
            continue
        if component.area < RIGHT_TERMINAL_MIN_TARGET_AREA_PX:
            continue
        if component.x_max < min_terminal_x and component.x_max < main_component.x_max - RIGHT_TERMINAL_SEARCH_PX:
            continue
        if _is_border_artifact(component, w):
            terminal_infos.append(TerminalInfo(
                source_label=main_component.label,
                target_label=component.label,
                accepted=False,
                reason="rejected_vertical_border_artifact",
                distance_px=0.0,
                dx_px=0,
                dy_px=0,
                bridge_area=0,
                start_point=None,
                end_point=None,
                thickness=0,
            ))
            continue
        target_candidates.append(component)

    target_candidates.sort(key=lambda c: (c.x_max, c.area), reverse=True)
    target_candidates = target_candidates[:RIGHT_TERMINAL_MAX_TARGETS]

    for target in target_candidates:
        source_candidates = [
            c for c in all_components
            if c.label != target.label
            and c.area >= RIGHT_TERMINAL_SOURCE_MIN_AREA_PX
            and c.x_min <= target.x_max
            and not _is_border_artifact(c, w)
        ]

        if len(source_candidates) == 0:
            terminal_infos.append(TerminalInfo(
                source_label=-1,
                target_label=target.label,
                accepted=False,
                reason="no_source_candidate",
                distance_px=0.0,
                dx_px=0,
                dy_px=0,
                bridge_area=0,
                start_point=None,
                end_point=None,
                thickness=0,
            ))
            continue

        best_bridge = np.zeros((h, w), dtype=np.uint8)
        best_info: Optional[TerminalInfo] = None
        best_score = float("inf")

        for source in source_candidates:
            local_bridge, info = _build_single_terminal_bridge(shape, source, target)
            if not info.accepted:
                continue
            score = info.distance_px + 0.25 * abs(info.dy_px) - 0.01 * source.area
            if score < best_score:
                best_score = score
                best_bridge = local_bridge
                best_info = info

        if best_info is None:
            terminal_infos.append(TerminalInfo(
                source_label=main_component.label,
                target_label=target.label,
                accepted=False,
                reason="no_accepted_source_pair",
                distance_px=0.0,
                dx_px=0,
                dy_px=0,
                bridge_area=0,
                start_point=None,
                end_point=None,
                thickness=0,
            ))
            continue

        terminal_infos.append(best_info)
        terminal_mask = cv2.bitwise_or(terminal_mask, best_bridge)

    return terminal_mask, terminal_infos


def _draw_terminal_debug(
    base_bgr: np.ndarray,
    components: List[ComponentInfo],
    rejected_components: List[ComponentInfo],
    terminal_infos: List[TerminalInfo],
    terminal_mask: np.ndarray,
) -> np.ndarray:
    out = base_bgr.copy()
    h, w = out.shape[:2]
    x0 = max(0, w - RIGHT_TERMINAL_SEARCH_PX)
    cv2.rectangle(out, (x0, 0), (w - 1, h - 1), (255, 255, 0), 1)

    all_components = _all_components_for_terminal(components, rejected_components)
    for comp in all_components:
        color = (0, 255, 255) if comp.area >= RIGHT_TERMINAL_SOURCE_MIN_AREA_PX else (0, 0, 255)
        if comp.x_max >= x0 or comp.x_max >= max((c.x_max for c in components), default=0) - RIGHT_TERMINAL_SEARCH_PX:
            cv2.rectangle(out, (comp.x_min, comp.y_min), (comp.x_max, comp.y_max), color, 1)
            cv2.putText(out, f"L{comp.label}", (comp.x_min, max(0, comp.y_min - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)

    contours, _ = cv2.findContours(terminal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 0, 255), 1)

    for info in terminal_infos:
        color = (255, 0, 255) if info.accepted else (0, 0, 255)
        if info.start_point is not None and info.end_point is not None:
            cv2.line(out, info.start_point, info.end_point, color, max(1, info.thickness), cv2.LINE_AA)
            cv2.circle(out, info.start_point, 3, (0, 255, 255), -1)
            cv2.circle(out, info.end_point, 3, (255, 255, 0), -1)
            xm = int(round((info.start_point[0] + info.end_point[0]) / 2.0))
            ym = int(round((info.start_point[1] + info.end_point[1]) / 2.0))
            cv2.putText(out, "TERM", (xm, ym), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return out


def _overlay_mask(base_bgr: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.65) -> np.ndarray:
    out = base_bgr.copy()
    colored = np.zeros_like(out)
    colored[mask > 0] = color
    return cv2.addWeighted(out, 1.0, colored, alpha, 0.0)


def _draw_contours_on_crop(base_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), CONTOUR_THICKNESS_PX)
    return out


def _draw_components_debug(
    base_bgr: np.ndarray,
    components: List[ComponentInfo],
    rejected_components: List[ComponentInfo],
) -> np.ndarray:
    out = base_bgr.copy()

    colors = [
        (0, 255, 0),
        (0, 180, 255),
        (255, 180, 0),
        (255, 0, 180),
        (180, 255, 0),
        (180, 0, 255),
        (0, 255, 180),
    ]

    for idx, comp in enumerate(components):
        color = colors[idx % len(colors)]
        contours, _ = cv2.findContours(comp.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, color, 1)
        cv2.rectangle(out, (comp.x_min, comp.y_min), (comp.x_max, comp.y_max), color, 1)
        cv2.putText(
            out,
            f"V{idx + 1}",
            (comp.x_min, max(0, comp.y_min - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    for comp in rejected_components:
        cv2.rectangle(out, (comp.x_min, comp.y_min), (comp.x_max, comp.y_max), (0, 0, 255), 1)

    return out


def _draw_chain_debug(
    base_bgr: np.ndarray,
    components: List[ComponentInfo],
    bridge_infos: List[BridgeInfo],
) -> np.ndarray:
    out = base_bgr.copy()

    for comp in components:
        cv2.circle(out, (int(round(comp.center_x)), int(round(comp.center_y))), 3, (0, 255, 255), -1)

    for info in bridge_infos:
        left = next((c for c in components if c.label == info.left_label), None)
        right = next((c for c in components if c.label == info.right_label), None)
        if left is None or right is None:
            continue

        color = (255, 0, 0) if info.accepted else (0, 0, 255)
        if info.start_point is not None and info.end_point is not None:
            cv2.line(out, info.start_point, info.end_point, color, 1, cv2.LINE_AA)
            x_mid = int(round((info.start_point[0] + info.end_point[0]) / 2.0))
            y_mid = int(round((info.start_point[1] + info.end_point[1]) / 2.0))
        else:
            x_mid = int(round((left.center_x + right.center_x) / 2.0))
            y_mid = int(round((left.center_y + right.center_y) / 2.0))

        label = "OK" if info.accepted else "SKIP"
        cv2.putText(out, label, (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return out


def _draw_bridge_boxes_debug(
    base_bgr: np.ndarray,
    bridge_mask: np.ndarray,
    bridge_infos: List[BridgeInfo],
) -> np.ndarray:
    out = base_bgr.copy()
    contours, _ = cv2.findContours(bridge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 1)

    for info in bridge_infos:
        if info.start_point is not None and info.end_point is not None:
            cv2.line(out, info.start_point, info.end_point, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(out, info.start_point, 2, (0, 255, 255), -1)
            cv2.circle(out, info.end_point, 2, (0, 255, 255), -1)

    return out


def _draw_anchor_profiles_debug(
    base_bgr: np.ndarray,
    components: List[ComponentInfo],
    bridge_infos: List[BridgeInfo],
) -> np.ndarray:
    out = base_bgr.copy()

    for info in bridge_infos:
        left = next((c for c in components if c.label == info.left_label), None)
        right = next((c for c in components if c.label == info.right_label), None)
        if left is None or right is None:
            continue

        left_anchor = _get_side_anchor(left, "right")
        right_anchor = _get_side_anchor(right, "left")

        if left_anchor.valid:
            for x, top, bridge_y in zip(left_anchor.xs, left_anchor.tops, left_anchor.bridge_ys):
                cv2.circle(out, (int(x), int(round(top))), 1, (0, 255, 255), -1)
                cv2.circle(out, (int(x), int(round(bridge_y))), 1, (255, 0, 0), -1)

        if right_anchor.valid:
            for x, top, bridge_y in zip(right_anchor.xs, right_anchor.tops, right_anchor.bridge_ys):
                cv2.circle(out, (int(x), int(round(top))), 1, (0, 255, 255), -1)
                cv2.circle(out, (int(x), int(round(bridge_y))), 1, (255, 0, 0), -1)

        if info.start_point is not None and info.end_point is not None:
            cv2.line(out, info.start_point, info.end_point, (0, 255, 0), info.thickness, cv2.LINE_AA)

    return out


def _build_report(
    components: List[ComponentInfo],
    rejected_components: List[ComponentInfo],
    bridge_infos: List[BridgeInfo],
    bridge_mask: np.ndarray,
    unified_mask: np.ndarray,
) -> str:
    accepted = [b for b in bridge_infos if b.accepted]
    rejected = [b for b in bridge_infos if not b.accepted]

    lines: List[str] = []
    lines.append("TOP2 UNIFICATION - FORCED COMPONENT CHAIN THIN CENTERLINE BRIDGE")
    lines.append(f"valid_components={len(components)}")
    lines.append(f"rejected_small_components={len(rejected_components)}")
    lines.append(f"pairs_checked={len(bridge_infos)}")
    lines.append(f"bridges_accepted={len(accepted)}")
    lines.append(f"bridges_rejected={len(rejected)}")
    lines.append(f"bridge_pixels={int(cv2.countNonZero(bridge_mask))}")
    lines.append(f"unified_pixels={int(cv2.countNonZero(unified_mask))}")
    lines.append("")

    for idx, comp in enumerate(components):
        lines.append(
            "component "
            f"{idx + 1}: label={comp.label} area={comp.area} "
            f"x=[{comp.x_min},{comp.x_max}] y=[{comp.y_min},{comp.y_max}] "
            f"median_top={comp.median_top:.1f} median_center_y={comp.median_center_y:.1f} "
            f"thickness={comp.median_thickness:.1f}"
        )

    lines.append("")
    for idx, info in enumerate(bridge_infos):
        lines.append(
            "bridge "
            f"{idx + 1}: {info.left_label}->{info.right_label} "
            f"accepted={info.accepted} reason={info.reason} "
            f"gap_px={info.gap_px} bridge_y_diff={info.bridge_y_diff:.1f} "
            f"area={info.bridge_area} thickness={info.thickness} "
            f"start={info.start_point} end={info.end_point}"
        )

    return "\n".join(lines)



def _build_report_v14(
    components: List[ComponentInfo],
    rejected_components: List[ComponentInfo],
    bridge_infos: List[BridgeInfo],
    terminal_infos: List[TerminalInfo],
    bridge_mask: np.ndarray,
    terminal_mask: np.ndarray,
    unified_mask: np.ndarray,
) -> str:
    accepted = [b for b in bridge_infos if b.accepted]
    rejected = [b for b in bridge_infos if not b.accepted]
    terminal_accepted = [t for t in terminal_infos if t.accepted]
    terminal_rejected = [t for t in terminal_infos if not t.accepted]

    lines: List[str] = []
    lines.append("TOP2 UNIFICATION - THIN CHAIN + RIGHT TERMINAL RESCUE")
    lines.append(f"valid_components={len(components)}")
    lines.append(f"rejected_small_components={len(rejected_components)}")
    lines.append(f"chain_pairs_checked={len(bridge_infos)}")
    lines.append(f"chain_bridges_accepted={len(accepted)}")
    lines.append(f"chain_bridges_rejected={len(rejected)}")
    lines.append(f"chain_bridge_pixels={int(cv2.countNonZero(bridge_mask))}")
    lines.append(f"right_terminal_checked={len(terminal_infos)}")
    lines.append(f"right_terminal_accepted={len(terminal_accepted)}")
    lines.append(f"right_terminal_rejected={len(terminal_rejected)}")
    lines.append(f"right_terminal_pixels={int(cv2.countNonZero(terminal_mask))}")
    lines.append(f"unified_pixels={int(cv2.countNonZero(unified_mask))}")
    lines.append("")

    for idx, comp in enumerate(components):
        lines.append(
            "component "
            f"{idx + 1}: label={comp.label} area={comp.area} "
            f"x=[{comp.x_min},{comp.x_max}] y=[{comp.y_min},{comp.y_max}] "
            f"median_top={comp.median_top:.1f} median_center_y={comp.median_center_y:.1f} "
            f"thickness={comp.median_thickness:.1f}"
        )

    lines.append("")
    for idx, info in enumerate(bridge_infos):
        lines.append(
            "chain_bridge "
            f"{idx + 1}: {info.left_label}->{info.right_label} "
            f"accepted={info.accepted} reason={info.reason} "
            f"gap_px={info.gap_px} bridge_y_diff={info.bridge_y_diff:.1f} "
            f"area={info.bridge_area} thickness={info.thickness} "
            f"start={info.start_point} end={info.end_point}"
        )

    lines.append("")
    for idx, info in enumerate(terminal_infos):
        lines.append(
            "right_terminal "
            f"{idx + 1}: {info.source_label}->{info.target_label} "
            f"accepted={info.accepted} reason={info.reason} "
            f"distance={info.distance_px:.1f} dx={info.dx_px} dy={info.dy_px} "
            f"area={info.bridge_area} thickness={info.thickness} "
            f"start={info.start_point} end={info.end_point}"
        )

    return "\n".join(lines)

def build_top2_unification_debug(
    crop_bgr: np.ndarray,
    top2_final_mask: np.ndarray,
) -> Dict[str, object]:
    base_bgr = _to_bgr(crop_bgr)
    final_mask = _to_binary_mask(top2_final_mask)

    components, rejected_components = _extract_components(final_mask)

    chain_bridge_mask = np.zeros_like(final_mask)
    bridge_infos: List[BridgeInfo] = []

    if len(components) >= 2:
        for left_component, right_component in zip(components[:-1], components[1:]):
            local_bridge_mask, info = _build_thin_centerline_bridge(
                final_mask.shape,
                left_component,
                right_component,
            )
            bridge_infos.append(info)
            if info.accepted:
                chain_bridge_mask = cv2.bitwise_or(chain_bridge_mask, local_bridge_mask)

    terminal_mask, terminal_infos = _build_right_terminal_rescue(
        final_mask.shape,
        components,
        rejected_components,
    )

    bridge_mask = cv2.bitwise_or(chain_bridge_mask, terminal_mask)
    added_chain_mask = cv2.bitwise_and(chain_bridge_mask, cv2.bitwise_not(final_mask))
    added_terminal_mask = cv2.bitwise_and(terminal_mask, cv2.bitwise_not(final_mask))
    added_bridge_mask = cv2.bitwise_or(added_chain_mask, added_terminal_mask)
    unified_mask = cv2.bitwise_or(final_mask, bridge_mask)

    chain_added_overlay = _overlay_mask(base_bgr, added_chain_mask, (255, 0, 0), alpha=0.90)
    terminal_added_overlay = _overlay_mask(base_bgr, added_terminal_mask, (255, 0, 255), alpha=0.90)
    all_added_overlay = _overlay_mask(base_bgr, added_bridge_mask, (255, 0, 255), alpha=0.85)

    images: Dict[str, np.ndarray] = {
        "00_top2_final_mask_on_crop": _overlay_mask(base_bgr, final_mask, (0, 255, 0), alpha=0.55),
        "01_valid_components_labeled": _draw_components_debug(base_bgr, components, rejected_components),
        "02_component_chain_pairs": _draw_chain_debug(base_bgr, components, bridge_infos),
        "03_thin_centerline_bridges_added_only": chain_added_overlay,
        "04_right_terminal_rescue_added_only": terminal_added_overlay,
        "05_all_bridges_added_only": all_added_overlay,
        "06_unified_contour_on_crop": _draw_contours_on_crop(base_bgr, unified_mask),
        "07_unified_mask_on_crop": _overlay_mask(base_bgr, unified_mask, (0, 255, 0), alpha=0.50),
        "08_original_plus_right_terminal_rescue": terminal_added_overlay,
        "09_right_terminal_debug": _draw_terminal_debug(base_bgr, components, rejected_components, terminal_infos, terminal_mask),
        "10_anchor_profiles_and_bridges": _draw_anchor_profiles_debug(base_bgr, components, bridge_infos),
        "11_bridge_boxes": _draw_bridge_boxes_debug(base_bgr, bridge_mask, bridge_infos),
        "03_profile_bridges_added_only": all_added_overlay,
        "08_original_plus_profile_bridges": all_added_overlay,
        "09_anchor_profiles_and_bridges": _draw_anchor_profiles_debug(base_bgr, components, bridge_infos),
    }

    report_text = _build_report_v14(
        components=components,
        rejected_components=rejected_components,
        bridge_infos=bridge_infos,
        terminal_infos=terminal_infos,
        bridge_mask=chain_bridge_mask,
        terminal_mask=terminal_mask,
        unified_mask=unified_mask,
    )

    return {
        "images": images,
        "report_text": report_text,
        "bridge_mask": bridge_mask,
        "chain_bridge_mask": chain_bridge_mask,
        "terminal_mask": terminal_mask,
        "added_bridge_mask": added_bridge_mask,
        "added_terminal_mask": added_terminal_mask,
        "unified_mask": unified_mask,
        "components": components,
        "rejected_components": rejected_components,
        "bridge_infos": bridge_infos,
        "terminal_infos": terminal_infos,
    }
