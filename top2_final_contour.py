from typing import Dict, Tuple

import cv2
import numpy as np

MIN_FINAL_COMPONENT_AREA_PX = 3
FINAL_CONTOUR_THICKNESS_PX = 1
COLOR_FINAL_CONTOUR = (0, 255, 0)


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("mask nu poate fi None")

    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask

    result = np.zeros(gray.shape[:2], dtype=np.uint8)
    result[gray > 0] = 255
    return result


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _subtract_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    a = _as_binary_mask(mask_a)
    b = _as_binary_mask(mask_b)

    result = np.zeros_like(a, dtype=np.uint8)
    result[(a > 0) & (b == 0)] = 255
    return result


def _clean_tiny_components(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    binary = _as_binary_mask(mask)
    kept = np.zeros_like(binary, dtype=np.uint8)
    removed = np.zeros_like(binary, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label

        if area >= MIN_FINAL_COMPONENT_AREA_PX:
            kept[component] = 255
        else:
            removed[component] = 255

    return kept, removed


def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    binary = _as_binary_mask(mask)

    if np.count_nonzero(binary) == 0:
        return binary

    height, width = binary.shape[:2]
    flood = binary.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(binary, flood_inv)
    filled[filled > 0] = 255
    return filled


def draw_top2_final_contour_on_crop(
    crop_bgr: np.ndarray,
    final_mask: np.ndarray,
    color: Tuple[int, int, int] = COLOR_FINAL_CONTOUR,
    thickness: int = FINAL_CONTOUR_THICKNESS_PX,
) -> np.ndarray:
    result = _to_bgr(crop_bgr)
    binary = _as_binary_mask(final_mask)

    if np.count_nonzero(binary) == 0:
        return result

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        result,
        contours,
        contourIdx=-1,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return result


def build_top2_final_contour(
    crop_bgr: np.ndarray,
    current_pleura_mask: np.ndarray,
    top2_guided_mask: np.ndarray,
    top2_added_mask: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    current = _as_binary_mask(current_pleura_mask)
    guided = _as_binary_mask(top2_guided_mask)

    cleaned_mask, removed_tiny_mask = _clean_tiny_components(guided)
    final_mask = _fill_internal_holes(cleaned_mask)

    if top2_added_mask is None:
        added_to_current_mask = _subtract_masks(final_mask, current)
    else:
        added_to_current_mask = _as_binary_mask(top2_added_mask)

    contour_on_crop = draw_top2_final_contour_on_crop(crop_bgr, final_mask)

    return {
        "final_top2_mask": final_mask,
        "contour_on_crop": contour_on_crop,
        "added_to_current_mask": added_to_current_mask,
        "removed_tiny_mask": removed_tiny_mask,
    }
