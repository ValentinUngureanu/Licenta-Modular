from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

# ============================================================
# PLEURAL NODULES 36 - NODUL = INGROSARE LOCALA A PLEUREI
# ============================================================
# Ideea acestei variante:
#   - NU mai cautam nodulul ca pe o componenta separata sub pleura;
#   - masca finala a pleurei este sursa principala;
#   - masuram grosimea pleurei pe fiecare coloana;
#   - cautam zone unde grosimea/marginea inferioara devine local mai mare
#     decat vecinatatea;
#   - nodulii finali sunt returnati ca dreptunghiuri, pentru folderul 14;
#   - zonele cu intreruperi sunt excluse din candidatii finali.
# ============================================================

NODULE_ENABLE = True
FILTER_STAGE = 5

# Profil pleura.
PROFILE_SMOOTH_WINDOW_PX = 7
PROFILE_BASELINE_WINDOW_PX = 65

# Pentru a nu lua capetele pleurei, unde masca poate fi instabila.
IGNORE_END_MARGIN_PX = 10

# Stage 2 - praguri pentru ingrosare locala.
# thickness_prominence = grosimea locala - grosimea normala din vecinatate.
# bottom_prominence    = cat coboara marginea inferioara fata de vecinatate.
MIN_ABSOLUTE_THICKNESS_PX = 4.0
MIN_THICKNESS_PROMINENCE_PX = 2.0
MIN_BOTTOM_PROMINENCE_PX = 1.2
STRONG_THICKNESS_PROMINENCE_PX = 3.8
STRONG_BOTTOM_PROMINENCE_PX = 4.0

# Grupare pe coloane candidate.
GROUP_MAX_X_GAP_PX = 3
GROUP_EXTRA_X_PAD_PX = 2
GROUP_EXTRA_Y_PAD_PX = 2

# Stage 3 - dimensiuni minime.
MIN_NODULE_WIDTH_PX = 5
MIN_NODULE_HEIGHT_PX = 5
MIN_NODULE_AREA_PX = 18
MIN_NODULE_BBOX_AREA_PX = 30
MIN_NODULE_ACTIVE_COLUMNS = 4
MIN_NODULE_MEAN_THICKNESS_PROMINENCE_PX = 1.1
MIN_NODULE_MAX_THICKNESS_PROMINENCE_PX = 2.0
MIN_NODULE_MAX_BOTTOM_PROMINENCE_PX = 1.0

# Stage 4 - eliminare zone exagerate.
MAX_NODULE_WIDTH_PX = 85
MAX_NODULE_HEIGHT_PX = 35
MAX_NODULE_AREA_PX = 2600
MAX_NODULE_BBOX_AREA_PX = 3000

# Stage 5 - forma finala.
# Pentru metoda pe grosimea pleurei NU mai cerem ca nodulul sa fie vertical/lung.
# Cerem doar sa fie o ingrosare locala reala si sa nu fie doar o bucata punctiforma.
FINAL_MIN_WIDTH_PX = 5
FINAL_MIN_HEIGHT_PX = 5
FINAL_MIN_BBOX_AREA_PX = 35
FINAL_MIN_SCORE = 3.0

# Excludere noduli pe intreruperi.
EXCLUDE_NODULES_ON_INTERRUPTION_ZONES = True
INTERRUPTION_EXCLUDE_DILATE_PX = 8
INTERRUPTION_EXCLUDE_MIN_OVERLAP_PX = 1

# Culori debug BGR.
COLOR_FINAL_CONTOUR = (0, 0, 255)  # rosu in debug crop = pleura reper
COLOR_SEARCH_BAND = (255, 0, 0)  # albastru
COLOR_CANDIDATE = (255, 255, 0)  # cyan/galben
COLOR_REJECTED = (0, 120, 255)  # portocaliu
COLOR_NODULE = (0, 255, 255)  # galben/cyan in debug
COLOR_PROFILE = (255, 0, 255)  # magenta
COLOR_BOX = (0, 255, 255)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Imaginea de intrare este goala sau None.")

    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGRA2BGR)

    return image.astype(np.uint8).copy()


def _as_binary_mask(
    mask: np.ndarray | None,
    shape: Tuple[int, int] | None = None,
) -> np.ndarray:
    if mask is None:
        if shape is None:
            raise ValueError("Masca este None si nu exista shape fallback.")
        return np.zeros(shape, dtype=np.uint8)

    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask.copy()

    if shape is not None and gray.shape[:2] != shape:
        gray = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    return np.where(gray > 0, 255, 0).astype(np.uint8)


def _dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    binary = _as_binary_mask(mask)

    if radius_px <= 0 or np.count_nonzero(binary > 0) == 0:
        return binary

    size = 2 * int(radius_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    result = cv2.dilate(binary, kernel, iterations=1)
    result[result > 0] = 255
    return result


def _rolling_median_float(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0 or window <= 1:
        return values.copy()

    if window % 2 == 0:
        window += 1

    if len(values) < window:
        return values.copy()

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    result = np.zeros_like(values, dtype=np.float32)

    for index in range(len(values)):
        result[index] = float(np.median(padded[index : index + window]))

    return result


def _smooth_profile(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1

    if window <= 1 or len(values) < window:
        return values.astype(np.float32).copy()

    return _rolling_median_float(values, window)


def _merge_masks(
    *masks: np.ndarray | None,
    shape: Tuple[int, int] | None = None,
) -> np.ndarray:
    base_shape = shape

    for mask in masks:
        if mask is not None:
            base_shape = mask.shape[:2]
            break

    if base_shape is None:
        raise ValueError("Nu pot determina dimensiunea pentru merge_masks.")

    result = np.zeros(base_shape, dtype=np.uint8)

    for mask in masks:
        if mask is None:
            continue
        binary = _as_binary_mask(mask, base_shape)
        result[binary > 0] = 255

    return result


def _build_pleura_profiles(pleura_mask: np.ndarray) -> Dict[str, object]:
    pleura = _as_binary_mask(pleura_mask)
    _height, width = pleura.shape[:2]

    valid_xs: List[int] = []
    top_values: List[float] = []
    bottom_values: List[float] = []

    for x in range(width):
        ys = np.flatnonzero(pleura[:, x] > 0)
        if len(ys) == 0:
            continue

        valid_xs.append(int(x))
        top_values.append(float(ys[0]))
        bottom_values.append(float(ys[-1]))

    valid_mask = np.zeros(width, dtype=bool)

    if len(valid_xs) == 0:
        return {
            "valid": valid_mask,
            "top_y": np.zeros(width, dtype=np.float32),
            "bottom_y": np.zeros(width, dtype=np.float32),
            "thickness": np.zeros(width, dtype=np.float32),
            "x_min": 0,
            "x_max": -1,
        }

    xs_arr = np.array(valid_xs, dtype=np.float32)
    top_arr = np.array(top_values, dtype=np.float32)
    bottom_arr = np.array(bottom_values, dtype=np.float32)

    x_min = int(xs_arr[0])
    x_max = int(xs_arr[-1])

    all_x = np.arange(width, dtype=np.float32)
    top_interp = np.interp(all_x, xs_arr, top_arr).astype(np.float32)
    bottom_interp = np.interp(all_x, xs_arr, bottom_arr).astype(np.float32)

    top_smooth = _smooth_profile(top_interp, PROFILE_SMOOTH_WINDOW_PX)
    bottom_smooth = _smooth_profile(bottom_interp, PROFILE_SMOOTH_WINDOW_PX)
    thickness = np.maximum(0.0, bottom_smooth - top_smooth + 1.0)

    valid_mask[x_min : x_max + 1] = True

    # Scoatem capetele instabile.
    if IGNORE_END_MARGIN_PX > 0:
        valid_mask[: max(0, x_min + IGNORE_END_MARGIN_PX)] = False
        valid_mask[min(width, x_max - IGNORE_END_MARGIN_PX + 1) :] = False

    thickness_baseline = _rolling_median_float(thickness, PROFILE_BASELINE_WINDOW_PX)
    bottom_baseline = _rolling_median_float(bottom_smooth, PROFILE_BASELINE_WINDOW_PX)

    thickness_prominence = thickness - thickness_baseline
    bottom_prominence = bottom_smooth - bottom_baseline

    return {
        "valid": valid_mask,
        "top_y": top_smooth,
        "bottom_y": bottom_smooth,
        "thickness": thickness,
        "thickness_baseline": thickness_baseline,
        "bottom_baseline": bottom_baseline,
        "thickness_prominence": thickness_prominence,
        "bottom_prominence": bottom_prominence,
        "x_min": x_min,
        "x_max": x_max,
    }


def _build_search_band_from_profile(
    shape: Tuple[int, int],
    profile: Dict[str, object],
) -> np.ndarray:
    height, width = shape[:2]
    band = np.zeros((height, width), dtype=np.uint8)

    valid = np.asarray(profile["valid"], dtype=bool)
    top_y = np.asarray(profile["top_y"], dtype=np.float32)
    bottom_y = np.asarray(profile["bottom_y"], dtype=np.float32)

    for x in range(width):
        if x >= len(valid) or not bool(valid[x]):
            continue

        y1 = max(0, int(round(float(top_y[x]))) - GROUP_EXTRA_Y_PAD_PX)
        y2 = min(height - 1, int(round(float(bottom_y[x]))) + GROUP_EXTRA_Y_PAD_PX)
        if y2 >= y1:
            band[y1 : y2 + 1, x] = 255

    return band


def _build_candidate_columns(profile: Dict[str, object]) -> np.ndarray:
    valid = np.asarray(profile["valid"], dtype=bool)
    thickness = np.asarray(profile["thickness"], dtype=np.float32)
    thickness_prom = np.asarray(profile["thickness_prominence"], dtype=np.float32)
    bottom_prom = np.asarray(profile["bottom_prominence"], dtype=np.float32)

    width = len(thickness)
    candidate_columns = np.zeros(width, dtype=bool)

    for x in range(width):
        if x >= len(valid) or not bool(valid[x]):
            continue

        if thickness[x] < MIN_ABSOLUTE_THICKNESS_PX:
            continue

        local_thickening = (
            thickness_prom[x] >= MIN_THICKNESS_PROMINENCE_PX
            and bottom_prom[x] >= MIN_BOTTOM_PROMINENCE_PX
        )
        strong_thickening = thickness_prom[x] >= STRONG_THICKNESS_PROMINENCE_PX
        strong_down_bulge = (
            bottom_prom[x] >= STRONG_BOTTOM_PROMINENCE_PX and thickness_prom[x] >= 0.5
        )

        if local_thickening or strong_thickening or strong_down_bulge:
            candidate_columns[x] = True

    return candidate_columns


def _split_true_segments(flags: np.ndarray, max_gap: int) -> List[Tuple[int, int]]:
    flags = np.asarray(flags, dtype=bool)
    segments: List[Tuple[int, int]] = []

    start: int | None = None
    last_true: int | None = None
    gap = 0

    for index, value in enumerate(flags):
        if value:
            if start is None:
                start = index
            last_true = index
            gap = 0
            continue

        if start is None:
            continue

        gap += 1
        if gap > max_gap:
            end = int(last_true if last_true is not None else index - gap)
            segments.append((int(start), int(end)))
            start = None
            last_true = None
            gap = 0

    if start is not None and last_true is not None:
        segments.append((int(start), int(last_true)))

    return segments


def _build_group_mask_from_segment(
    pleura_mask: np.ndarray,
    profile: Dict[str, object],
    x1: int,
    x2: int,
) -> np.ndarray:
    pleura = _as_binary_mask(pleura_mask)
    height, width = pleura.shape[:2]

    x1 = max(0, min(width - 1, int(x1) - GROUP_EXTRA_X_PAD_PX))
    x2 = max(0, min(width - 1, int(x2) + GROUP_EXTRA_X_PAD_PX))

    result = np.zeros_like(pleura, dtype=np.uint8)
    if x2 < x1:
        return result

    top_y = np.asarray(profile["top_y"], dtype=np.float32)
    bottom_y = np.asarray(profile["bottom_y"], dtype=np.float32)

    for x in range(x1, x2 + 1):
        y1 = max(0, int(round(float(top_y[x]))) - GROUP_EXTRA_Y_PAD_PX)
        y2 = min(height - 1, int(round(float(bottom_y[x]))) + GROUP_EXTRA_Y_PAD_PX)
        if y2 < y1:
            continue
        column = pleura[y1 : y2 + 1, x]
        result[y1 : y2 + 1, x][column > 0] = 255

    return result


def _group_local_thickening_regions(
    pleura_mask: np.ndarray,
    profile: Dict[str, object],
    candidate_columns: np.ndarray,
) -> List[Dict[str, object]]:
    segments = _split_true_segments(candidate_columns, GROUP_MAX_X_GAP_PX)

    thickness = np.asarray(profile["thickness"], dtype=np.float32)
    thickness_prom = np.asarray(profile["thickness_prominence"], dtype=np.float32)
    bottom_prom = np.asarray(profile["bottom_prominence"], dtype=np.float32)

    groups: List[Dict[str, object]] = []

    for group_index, (x1, x2) in enumerate(segments, start=1):
        group_mask = _build_group_mask_from_segment(pleura_mask, profile, x1, x2)
        ys, xs = np.where(group_mask > 0)

        if len(xs) == 0:
            continue

        min_x = int(np.min(xs))
        max_x = int(np.max(xs))
        min_y = int(np.min(ys))
        max_y = int(np.max(ys))
        width = max(1, max_x - min_x + 1)
        height = max(1, max_y - min_y + 1)
        area = int(len(xs))
        bbox_area = int(width * height)
        active_columns = int(len(np.unique(xs)))

        seg_x1 = max(0, int(x1))
        seg_x2 = min(len(thickness) - 1, int(x2))
        local_slice = slice(seg_x1, seg_x2 + 1)

        max_thickness = (
            float(np.max(thickness[local_slice])) if seg_x2 >= seg_x1 else 0.0
        )
        mean_thickness = (
            float(np.mean(thickness[local_slice])) if seg_x2 >= seg_x1 else 0.0
        )
        max_thickness_prom = (
            float(np.max(thickness_prom[local_slice])) if seg_x2 >= seg_x1 else 0.0
        )
        mean_thickness_prom = (
            float(np.mean(thickness_prom[local_slice])) if seg_x2 >= seg_x1 else 0.0
        )
        max_bottom_prom = (
            float(np.max(bottom_prom[local_slice])) if seg_x2 >= seg_x1 else 0.0
        )
        mean_bottom_prom = (
            float(np.mean(bottom_prom[local_slice])) if seg_x2 >= seg_x1 else 0.0
        )

        score = float(
            max_thickness_prom + 0.8 * max_bottom_prom + 0.3 * mean_thickness_prom
        )

        groups.append(
            {
                "index": int(group_index),
                "mask": group_mask,
                "x_min": int(min_x),
                "x_max": int(max_x),
                "y_min": int(min_y),
                "y_max": int(max_y),
                "width": int(width),
                "height": int(height),
                "area": int(area),
                "bbox_area": int(bbox_area),
                "active_columns": int(active_columns),
                "max_thickness": float(max_thickness),
                "mean_thickness": float(mean_thickness),
                "max_thickness_prominence": float(max_thickness_prom),
                "mean_thickness_prominence": float(mean_thickness_prom),
                "max_bottom_prominence": float(max_bottom_prom),
                "mean_bottom_prominence": float(mean_bottom_prom),
                "height_to_width_ratio": float(height / max(width, 1)),
                "score": float(score),
            }
        )

    return groups


def _build_interruption_exclusion_zone(
    interruption_mask: np.ndarray | None,
    shape: Tuple[int, int],
) -> np.ndarray:
    if not EXCLUDE_NODULES_ON_INTERRUPTION_ZONES:
        return np.zeros(shape, dtype=np.uint8)

    if interruption_mask is None:
        return np.zeros(shape, dtype=np.uint8)

    interruption = _as_binary_mask(interruption_mask, shape)
    if np.count_nonzero(interruption > 0) == 0:
        return np.zeros(shape, dtype=np.uint8)

    exclusion = _dilate(interruption, INTERRUPTION_EXCLUDE_DILATE_PX)
    exclusion[exclusion > 0] = 255
    return exclusion


def _filter_groups(
    groups: List[Dict[str, object]],
    stage: int,
    shape: Tuple[int, int],
    interruption_exclusion_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, List[Dict[str, object]], np.ndarray]:
    result = np.zeros(shape, dtype=np.uint8)
    rejected = np.zeros(shape, dtype=np.uint8)
    accepted_infos: List[Dict[str, object]] = []

    if interruption_exclusion_mask is None:
        interruption_exclusion = np.zeros(shape, dtype=np.uint8)
    else:
        interruption_exclusion = _as_binary_mask(interruption_exclusion_mask, shape)

    for group in groups:
        group_mask = _as_binary_mask(group["mask"], shape)
        accepted = True
        reason = "accepted"

        if stage >= 3:
            if int(group["width"]) < MIN_NODULE_WIDTH_PX:
                accepted = False
                reason = "stage3_width_too_small"
            elif int(group["height"]) < MIN_NODULE_HEIGHT_PX:
                accepted = False
                reason = "stage3_height_too_small"
            elif int(group["area"]) < MIN_NODULE_AREA_PX:
                accepted = False
                reason = "stage3_area_too_small"
            elif int(group["bbox_area"]) < MIN_NODULE_BBOX_AREA_PX:
                accepted = False
                reason = "stage3_bbox_too_small"
            elif int(group["active_columns"]) < MIN_NODULE_ACTIVE_COLUMNS:
                accepted = False
                reason = "stage3_too_few_columns"
            elif (
                float(group["mean_thickness_prominence"])
                < MIN_NODULE_MEAN_THICKNESS_PROMINENCE_PX
            ):
                accepted = False
                reason = "stage3_mean_thickness_prominence_too_small"
            elif (
                float(group["max_thickness_prominence"])
                < MIN_NODULE_MAX_THICKNESS_PROMINENCE_PX
            ):
                accepted = False
                reason = "stage3_max_thickness_prominence_too_small"
            elif (
                float(group["max_bottom_prominence"])
                < MIN_NODULE_MAX_BOTTOM_PROMINENCE_PX
            ):
                accepted = False
                reason = "stage3_bottom_prominence_too_small"

        if accepted and stage >= 4:
            if int(group["width"]) > MAX_NODULE_WIDTH_PX:
                accepted = False
                reason = "stage4_width_too_large"
            elif int(group["height"]) > MAX_NODULE_HEIGHT_PX:
                accepted = False
                reason = "stage4_height_too_large"
            elif int(group["area"]) > MAX_NODULE_AREA_PX:
                accepted = False
                reason = "stage4_area_too_large"
            elif int(group["bbox_area"]) > MAX_NODULE_BBOX_AREA_PX:
                accepted = False
                reason = "stage4_bbox_too_large"

        if accepted and stage >= 5:
            if int(group["width"]) < FINAL_MIN_WIDTH_PX:
                accepted = False
                reason = "stage5_final_width_too_small"
            elif int(group["height"]) < FINAL_MIN_HEIGHT_PX:
                accepted = False
                reason = "stage5_final_height_too_small"
            elif int(group["bbox_area"]) < FINAL_MIN_BBOX_AREA_PX:
                accepted = False
                reason = "stage5_final_bbox_too_small"
            elif float(group["score"]) < FINAL_MIN_SCORE:
                accepted = False
                reason = "stage5_score_too_small"

        if accepted and stage >= 5 and EXCLUDE_NODULES_ON_INTERRUPTION_ZONES:
            x1 = max(0, int(group["x_min"]))
            y1 = max(0, int(group["y_min"]))
            x2 = min(shape[1] - 1, int(group["x_max"]))
            y2 = min(shape[0] - 1, int(group["y_max"]))

            if x2 >= x1 and y2 >= y1:
                overlap_pixels = int(
                    np.count_nonzero(
                        interruption_exclusion[y1 : y2 + 1, x1 : x2 + 1] > 0
                    )
                )
            else:
                overlap_pixels = 0

            if overlap_pixels >= INTERRUPTION_EXCLUDE_MIN_OVERLAP_PX:
                accepted = False
                reason = "stage5_on_interruption_zone"

        if not accepted:
            rejected[group_mask > 0] = 255
            continue

        result[group_mask > 0] = 255
        info = dict(group)
        info.pop("mask", None)
        info["reason"] = reason
        accepted_infos.append(info)

    return result, accepted_infos, rejected


def _build_box_mask_from_infos(
    shape: Tuple[int, int],
    infos: List[Dict[str, object]],
    pad: int = 2,
) -> np.ndarray:
    height, width = shape[:2]
    box_mask = np.zeros((height, width), dtype=np.uint8)

    for info in infos:
        x1 = max(0, int(info["x_min"]) - pad)
        y1 = max(0, int(info["y_min"]) - pad)
        x2 = min(width - 1, int(info["x_max"]) + pad)
        y2 = min(height - 1, int(info["y_max"]) + pad)
        cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, thickness=-1)

    return box_mask


def _draw_contours(
    base_image: np.ndarray,
    mask: np.ndarray,
    color,
    thickness: int = 1,
) -> np.ndarray:
    result = _to_bgr(base_image)
    binary = _as_binary_mask(mask, result.shape[:2])

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return result

    cv2.drawContours(result, contours, -1, color, thickness, cv2.LINE_AA)
    return result


def _draw_boxes_from_infos(
    base_image: np.ndarray,
    infos: List[Dict[str, object]],
    color=COLOR_BOX,
    thickness: int = 2,
    pad: int = 2,
) -> np.ndarray:
    result = _to_bgr(base_image)
    height, width = result.shape[:2]

    for index, info in enumerate(infos, start=1):
        x1 = max(0, int(info["x_min"]) - pad)
        y1 = max(0, int(info["y_min"]) - pad)
        x2 = min(width - 1, int(info["x_max"]) + pad)
        y2 = min(height - 1, int(info["y_max"]) + pad)

        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        cv2.putText(
            result,
            f"N{index}",
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    return result


def _draw_profile(
    base_image: np.ndarray,
    profile: Dict[str, object] | None,
) -> np.ndarray:
    result = _to_bgr(base_image)

    if profile is None or "bottom_y" not in profile:
        return result

    bottom_y = np.asarray(profile["bottom_y"], dtype=np.float32)
    valid = np.asarray(profile["valid"], dtype=bool)
    h, w = result.shape[:2]

    points: List[Tuple[int, int]] = []

    for x in range(w):
        if x >= len(valid) or not bool(valid[x]):
            continue
        y = int(round(float(bottom_y[x])))
        if 0 <= y < h:
            points.append((x, y))

    if len(points) >= 2:
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(result, [pts], False, COLOR_PROFILE, 1, cv2.LINE_AA)

    return result


def _put_title(image: np.ndarray, title_text: str) -> np.ndarray:
    result = _to_bgr(image)

    cv2.putText(
        result,
        title_text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        title_text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return result


def draw_nodule_marking(
    base_bgr: np.ndarray,
    pleura_mask: np.ndarray,
    nodule_mask: np.ndarray,
    profile: Dict[str, object] | None = None,
    title_text: str | None = None,
    nodule_infos: List[Dict[str, object]] | None = None,
) -> np.ndarray:
    result = _to_bgr(base_bgr)
    result = _draw_contours(result, pleura_mask, COLOR_FINAL_CONTOUR, thickness=1)
    result = _draw_contours(result, nodule_mask, COLOR_NODULE, thickness=1)

    if nodule_infos:
        result = _draw_boxes_from_infos(
            result, nodule_infos, color=COLOR_BOX, thickness=2, pad=2
        )

    result = _draw_profile(result, profile)

    if title_text is None:
        title_text = "NODULI = ingrosare locala a pleurei | dreptunghi=candidat"

    return _put_title(result, title_text)


def draw_nodule_candidate_debug(
    base_bgr: np.ndarray,
    pleura_mask: np.ndarray,
    search_band_mask: np.ndarray,
    working_mask: np.ndarray,
    candidate_mask: np.ndarray,
    rejected_mask: np.ndarray,
    nodule_mask: np.ndarray,
    profile: Dict[str, object] | None = None,
    title_text: str | None = None,
    nodule_infos: List[Dict[str, object]] | None = None,
) -> np.ndarray:
    result = _to_bgr(base_bgr)

    result = _draw_contours(result, search_band_mask, COLOR_SEARCH_BAND, thickness=1)
    result = _draw_contours(result, working_mask, COLOR_CANDIDATE, thickness=1)
    result = _draw_contours(result, candidate_mask, COLOR_CANDIDATE, thickness=1)
    result = _draw_contours(result, rejected_mask, COLOR_REJECTED, thickness=1)
    result = _draw_contours(result, nodule_mask, COLOR_NODULE, thickness=2)

    if nodule_infos:
        result = _draw_boxes_from_infos(
            result, nodule_infos, color=COLOR_BOX, thickness=2, pad=2
        )

    result = _draw_contours(result, pleura_mask, COLOR_FINAL_CONTOUR, thickness=1)
    result = _draw_profile(result, profile)

    if title_text is None:
        title_text = "DEBUG NODULI | galben=candidat | portocaliu=respins | cyan=final"

    return _put_title(result, title_text)


def _empty_result(
    base: np.ndarray,
    pleura: np.ndarray,
    profile: Dict[str, object],
) -> Dict[str, object]:
    empty = np.zeros_like(pleura, dtype=np.uint8)
    debug = draw_nodule_candidate_debug(
        base,
        pleura,
        empty,
        empty,
        empty,
        empty,
        empty,
        profile,
    )

    result: Dict[str, object] = {
        "nodule_mask": empty,
        "nodule_core_mask": empty,
        "nodule_box_mask": empty,
        "candidate_mask": empty,
        "rejected_mask": empty,
        "under_structure_mask": empty,
        "under_structure_rejected_mask": empty,
        "working_mask": empty,
        "excluded_mask": empty,
        "search_band_mask": empty,
        "contact_zone_mask": empty,
        "interruption_exclusion_mask": empty,
        "nodule_image": draw_nodule_marking(base, pleura, empty, profile),
        "candidate_debug_image": debug,
        "nodule_infos": [],
        "nodule_count": 0,
        "stage5_removed_small_mask": empty,
    }

    for stage in range(0, max(0, min(int(FILTER_STAGE), 5)) + 1):
        result[f"stage{stage}_mask"] = empty.copy()
        result[f"stage{stage}_rejected_mask"] = empty.copy()
        result[f"stage{stage}_debug_image"] = debug.copy()

    return result


def detect_pleural_nodules(
    base_bgr: np.ndarray,
    pleura_mask: np.ndarray,
    bridge_mask: np.ndarray | None = None,
    interruption_mask: np.ndarray | None = None,
    binary_top3: np.ndarray | None = None,
    binary_top4: np.ndarray | None = None,
) -> Dict[str, object]:
    _ = binary_top3
    _ = binary_top4

    base = _to_bgr(base_bgr)
    pleura = _as_binary_mask(pleura_mask, base.shape[:2])
    profile = _build_pleura_profiles(pleura)

    if not NODULE_ENABLE or np.count_nonzero(pleura > 0) == 0:
        return _empty_result(base, pleura, profile)

    search_band = _build_search_band_from_profile(pleura.shape, profile)

    # Stage 0: pleura finala in zona valida a profilului.
    stage0_mask = cv2.bitwise_and(pleura, search_band)

    # Stage 1: in varianta asta contactul cu pleura este implicit,
    # pentru ca sursa este chiar masca pleurei finale.
    stage1_mask = stage0_mask.copy()

    candidate_columns = _build_candidate_columns(profile)

    # Stage 2: coloane unde pleura este local mai groasa / coboara local.
    stage2_mask = np.zeros_like(pleura, dtype=np.uint8)
    height, width = pleura.shape[:2]
    top_y = np.asarray(profile["top_y"], dtype=np.float32)
    bottom_y = np.asarray(profile["bottom_y"], dtype=np.float32)

    for x in np.flatnonzero(candidate_columns):
        if x < 0 or x >= width:
            continue
        y1 = max(0, int(round(float(top_y[x]))) - GROUP_EXTRA_Y_PAD_PX)
        y2 = min(height - 1, int(round(float(bottom_y[x]))) + GROUP_EXTRA_Y_PAD_PX)
        if y2 < y1:
            continue
        column = pleura[y1 : y2 + 1, x]
        stage2_mask[y1 : y2 + 1, x][column > 0] = 255

    groups = _group_local_thickening_regions(pleura, profile, candidate_columns)

    interruption_exclusion = _build_interruption_exclusion_zone(
        interruption_mask,
        pleura.shape,
    )
    excluded = _merge_masks(
        bridge_mask, interruption_mask, interruption_exclusion, shape=pleura.shape
    )

    stage3_mask, stage3_infos, stage3_rejected_local = _filter_groups(
        groups,
        stage=3,
        shape=pleura.shape,
    )
    stage4_mask, stage4_infos, stage4_rejected_local = _filter_groups(
        groups,
        stage=4,
        shape=pleura.shape,
    )
    stage5_mask, stage5_infos, stage5_rejected_local = _filter_groups(
        groups,
        stage=5,
        shape=pleura.shape,
        interruption_exclusion_mask=interruption_exclusion,
    )

    stage_masks: Dict[int, np.ndarray] = {
        0: stage0_mask,
        1: stage1_mask,
        2: stage2_mask,
        3: stage3_mask,
        4: stage4_mask,
        5: stage5_mask,
    }

    final_stage = max(0, min(int(FILTER_STAGE), 5))
    nodule_core_mask = stage_masks[final_stage].copy()

    if final_stage >= 5:
        selected_infos = stage5_infos
    elif final_stage >= 4:
        selected_infos = stage4_infos
    elif final_stage >= 3:
        selected_infos = stage3_infos
    else:
        selected_infos = []

    nodule_infos: List[Dict[str, object]] = []
    for index, info in enumerate(selected_infos, start=1):
        nodule_infos.append(
            {
                "index": int(index),
                "score": float(info.get("score", 0.0)),
                "area": int(info["area"]),
                "bbox_area": int(info["bbox_area"]),
                "width": int(info["width"]),
                "height": int(info["height"]),
                "x_min": int(info["x_min"]),
                "x_max": int(info["x_max"]),
                "y_min": int(info["y_min"]),
                "y_max": int(info["y_max"]),
                "active_columns": int(info["active_columns"]),
                "max_thickness": float(info.get("max_thickness", 0.0)),
                "mean_thickness": float(info.get("mean_thickness", 0.0)),
                "max_thickness_prominence": float(
                    info.get("max_thickness_prominence", 0.0)
                ),
                "mean_thickness_prominence": float(
                    info.get("mean_thickness_prominence", 0.0)
                ),
                "max_bottom_prominence": float(info.get("max_bottom_prominence", 0.0)),
                "mean_bottom_prominence": float(
                    info.get("mean_bottom_prominence", 0.0)
                ),
                "height_to_width_ratio": float(info.get("height_to_width_ratio", 0.0)),
                "filter_stage": int(FILTER_STAGE),
                "reason": str(info.get("reason", "accepted")),
            }
        )

    nodule_box_mask = _build_box_mask_from_infos(pleura.shape, nodule_infos, pad=2)
    nodule_mask = nodule_core_mask.copy()

    candidate_mask = stage2_mask.copy()
    rejected_mask = np.zeros_like(stage0_mask, dtype=np.uint8)
    rejected_mask[(stage0_mask > 0) & (nodule_mask == 0)] = 255

    stage_rejected_masks: Dict[int, np.ndarray] = {}
    for stage, current in stage_masks.items():
        rejected = np.zeros_like(stage0_mask, dtype=np.uint8)
        rejected[(stage0_mask > 0) & (current == 0)] = 255
        stage_rejected_masks[stage] = rejected

    stage5_removed_small_mask = np.zeros_like(stage0_mask, dtype=np.uint8)
    stage5_removed_small_mask[(stage4_mask > 0) & (stage5_mask == 0)] = 255
    stage5_removed_small_mask[stage5_rejected_local > 0] = 255

    stage_titles = {
        0: "STAGE 0: pleura finala valida",
        1: "STAGE 1: contact implicit, sursa este pleura",
        2: "STAGE 2: coloane cu ingrosare locala",
        3: "STAGE 3: candidati cu dimensiuni minime",
        4: "STAGE 4: elimina candidati exagerat de mari",
        5: "STAGE 5: score final + fara zone cu intreruperi",
    }

    infos_by_stage = {
        3: stage3_infos,
        4: stage4_infos,
        5: nodule_infos,
    }

    stage_debug_images: Dict[int, np.ndarray] = {}
    for stage, stage_mask in stage_masks.items():
        stage_debug_images[stage] = draw_nodule_candidate_debug(
            base_bgr=base,
            pleura_mask=pleura,
            search_band_mask=search_band,
            working_mask=stage0_mask,
            candidate_mask=candidate_mask,
            rejected_mask=stage_rejected_masks[stage],
            nodule_mask=stage_mask,
            profile=profile,
            title_text=stage_titles.get(stage, f"STAGE {stage}"),
            nodule_infos=infos_by_stage.get(stage, None),
        )

    nodule_image = draw_nodule_marking(
        base_bgr=base,
        pleura_mask=pleura,
        nodule_mask=nodule_mask,
        profile=profile,
        nodule_infos=nodule_infos,
    )

    candidate_debug_image = draw_nodule_candidate_debug(
        base_bgr=base,
        pleura_mask=pleura,
        search_band_mask=search_band,
        working_mask=stage0_mask,
        candidate_mask=candidate_mask,
        rejected_mask=rejected_mask,
        nodule_mask=nodule_mask,
        profile=profile,
        nodule_infos=nodule_infos,
    )

    result: Dict[str, object] = {
        "nodule_mask": nodule_mask,
        "nodule_core_mask": nodule_core_mask,
        "nodule_box_mask": nodule_box_mask,
        "candidate_mask": candidate_mask,
        "rejected_mask": rejected_mask,
        "under_structure_mask": nodule_mask,
        "under_structure_rejected_mask": rejected_mask,
        "working_mask": stage0_mask,
        "excluded_mask": excluded,
        "search_band_mask": search_band,
        "contact_zone_mask": stage1_mask,
        "interruption_exclusion_mask": interruption_exclusion,
        "nodule_image": nodule_image,
        "candidate_debug_image": candidate_debug_image,
        "nodule_infos": nodule_infos,
        "nodule_count": len(nodule_infos),
        "stage5_removed_small_mask": stage5_removed_small_mask,
    }

    for stage, stage_mask in stage_masks.items():
        result[f"stage{stage}_mask"] = stage_mask
        result[f"stage{stage}_rejected_mask"] = stage_rejected_masks[stage]
        result[f"stage{stage}_debug_image"] = stage_debug_images[stage]

    return result
