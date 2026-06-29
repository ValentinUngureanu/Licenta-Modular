from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

INTERRUPTION_MIN_WIDTH_PX = 15
INTERRUPTION_MIN_AREA_PX = 6
INTERRUPTION_MIN_HEIGHT_PX = 2
INTERRUPTION_MIN_MEAN_THICKNESS_PX = 3.0
INTERRUPTION_CLOSE_KERNEL_PX = 3

NEAR_ORIGINAL_REMOVE_ENABLE = True
NEAR_ORIGINAL_DILATE_PX = 5


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        raise ValueError("Masca de intrare este goala sau None.")

    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask.copy()

    return np.where(gray > 0, 255, 0).astype(np.uint8)


def _component_mean_thickness(area: int, width: int) -> float:
    return float(area / max(width, 1))


def _build_near_original_guard(original_mask: np.ndarray) -> np.ndarray:
    original = _to_binary_mask(original_mask)

    if not NEAR_ORIGINAL_REMOVE_ENABLE:
        return original

    dilate_px = int(max(0, NEAR_ORIGINAL_DILATE_PX))

    if dilate_px <= 0:
        return original

    kernel_size = 2 * dilate_px + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    guard = cv2.dilate(original, kernel, iterations=1)
    return np.where(guard > 0, 255, 0).astype(np.uint8)


def _is_interruption_component_valid(area: int, width: int, height: int) -> bool:
    if area < INTERRUPTION_MIN_AREA_PX:
        return False

    if width < INTERRUPTION_MIN_WIDTH_PX:
        return False

    if height < INTERRUPTION_MIN_HEIGHT_PX:
        return False

    mean_thickness = _component_mean_thickness(area, width)

    if mean_thickness < INTERRUPTION_MIN_MEAN_THICKNESS_PX:
        return False

    return True


def build_interruption_mask(
    original_mask: np.ndarray,
    bridge_mask: np.ndarray,
) -> np.ndarray:
    interruption_mask = _to_binary_mask(bridge_mask)
    near_original_guard = _build_near_original_guard(original_mask)

    interruption_mask[near_original_guard > 0] = 0

    if INTERRUPTION_CLOSE_KERNEL_PX > 1:
        close_size = int(INTERRUPTION_CLOSE_KERNEL_PX)

        if close_size % 2 == 0:
            close_size += 1

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size),
        )

        interruption_mask = cv2.morphologyEx(
            interruption_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=1,
        )

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        interruption_mask,
        8,
    )

    clean_mask = np.zeros_like(interruption_mask)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        if not _is_interruption_component_valid(area, width, height):
            continue

        clean_mask[labels == label] = 255

    return np.where(clean_mask > 0, 255, 0).astype(np.uint8)


def extract_interruption_infos(
    interruption_mask: np.ndarray,
) -> List[Dict[str, int]]:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        interruption_mask,
        8,
    )

    infos: List[Dict[str, int]] = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        if not _is_interruption_component_valid(area, width, height):
            continue

        mean_thickness = _component_mean_thickness(area, width)

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])

        infos.append(
            {
                "index": len(infos) + 1,
                "area": area,
                "width": width,
                "height": height,
                "mean_thickness": round(mean_thickness, 2),
                "x_min": x,
                "x_max": x + width - 1,
                "y_min": y,
                "y_max": y + height - 1,
            }
        )

    infos.sort(key=lambda item: (item["x_min"], item["y_min"]))

    for index, item in enumerate(infos, start=1):
        item["index"] = index

    return infos


def draw_interruption_marking(
    base_bgr: np.ndarray,
    final_mask: np.ndarray,
    interruption_mask: np.ndarray,
) -> np.ndarray:
    out = base_bgr.copy()

    final_contours, _hierarchy = cv2.findContours(
        _to_binary_mask(final_mask),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    cv2.drawContours(
        out,
        final_contours,
        -1,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    interruption_mask = _to_binary_mask(interruption_mask)

    overlay = out.copy()
    overlay[interruption_mask > 0] = (0, 255, 255)

    out = cv2.addWeighted(
        overlay,
        0.55,
        out,
        0.45,
        0.0,
    )

    interruption_contours, _hierarchy = cv2.findContours(
        interruption_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    cv2.drawContours(
        out,
        interruption_contours,
        -1,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def detect_pleural_interruptions(
    base_bgr: np.ndarray,
    original_mask: np.ndarray,
    final_mask: np.ndarray,
    bridge_mask: np.ndarray,
) -> Dict[str, object]:
    interruption_mask = build_interruption_mask(
        original_mask=original_mask,
        bridge_mask=bridge_mask,
    )

    interruption_infos = extract_interruption_infos(interruption_mask)

    interruption_image = draw_interruption_marking(
        base_bgr=base_bgr,
        final_mask=final_mask,
        interruption_mask=interruption_mask,
    )

    return {
        "interruption_mask": interruption_mask,
        "interruption_image": interruption_image,
        "interruption_infos": interruption_infos,
        "interruption_count": len(interruption_infos),
    }
