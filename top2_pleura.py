from typing import Dict

import cv2
import numpy as np

from final_contour import remove_very_small_components

SEARCH_BAND_DILATE_X_PX = 18
SEARCH_BAND_DILATE_Y_PX = 7


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


def _merge_masks(*masks: np.ndarray) -> np.ndarray:
    if len(masks) == 0:
        raise ValueError("Trebuie cel putin o masca pentru merge")

    result = np.zeros(_as_binary_mask(masks[0]).shape, dtype=np.uint8)

    for mask in masks:
        binary = _as_binary_mask(mask)
        result[binary > 0] = 255

    return result


def _subtract_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    a = _as_binary_mask(mask_a)
    b = _as_binary_mask(mask_b)

    result = np.zeros_like(a, dtype=np.uint8)
    result[(a > 0) & (b == 0)] = 255
    return result


def _intersect_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    a = _as_binary_mask(mask_a)
    b = _as_binary_mask(mask_b)

    result = np.zeros_like(a, dtype=np.uint8)
    result[(a > 0) & (b > 0)] = 255
    return result


def _build_search_band(anchor_mask: np.ndarray) -> np.ndarray:
    anchor = _as_binary_mask(anchor_mask)

    if np.count_nonzero(anchor) == 0:
        return np.zeros_like(anchor, dtype=np.uint8)

    kernel_width = 2 * SEARCH_BAND_DILATE_X_PX + 1
    kernel_height = 2 * SEARCH_BAND_DILATE_Y_PX + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_width, kernel_height),
    )

    search_band = cv2.dilate(anchor, kernel, iterations=1)
    search_band[search_band > 0] = 255
    return search_band


def build_top2_guided_pleura(
    binary_top1: np.ndarray,
    binary_top2: np.ndarray,
    current_pleura_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    top1 = _as_binary_mask(binary_top1)
    top2 = _as_binary_mask(binary_top2)
    current = _as_binary_mask(current_pleura_mask)

    search_band = _build_search_band(current)
    top2_in_search_band = _intersect_masks(top2, search_band)
    anchor_mask = _merge_masks(current, top1)

    top2_added_candidate = _subtract_masks(top2_in_search_band, anchor_mask)
    top2_added_clean, top2_added_removed = remove_very_small_components(
        top2_added_candidate
    )

    top2_guided_mask = _merge_masks(current, top2_added_clean)
    top2_added_to_current = _subtract_masks(top2_guided_mask, current)

    return {
        "top2_guided_mask": top2_guided_mask,
        "top2_added_to_current": top2_added_to_current,
        "top2_added_candidate": top2_added_candidate,
        "top2_added_removed": top2_added_removed,
        "search_band": search_band,
    }
