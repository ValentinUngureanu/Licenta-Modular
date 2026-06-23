from typing import Dict, Tuple

import cv2
import numpy as np

FINAL_CONTOUR_THICKNESS_PX = 1
COLOR_FINAL_CONTOUR = (0, 255, 0)

# Pragurile filtrului pentru componente foarte mici din top2.
# Daca filtrul sterge prea putin, cresti valorile.
# Daca filtrul sterge bucati bune de pleura, scazi valorile.
TOP2_REMOVE_AREA_ALWAYS_LT = 100
TOP2_REMOVE_AREA_THIN_LT = 100
TOP2_REMOVE_THIN_WIDTH_LE = 15
TOP2_REMOVE_THIN_HEIGHT_LE = 8


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


def _merge_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    a = _as_binary_mask(mask_a)
    b = _as_binary_mask(mask_b)

    result = np.zeros_like(a, dtype=np.uint8)
    result[a > 0] = 255
    result[b > 0] = 255
    return result


def _subtract_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    a = _as_binary_mask(mask_a)
    b = _as_binary_mask(mask_b)

    result = np.zeros_like(a, dtype=np.uint8)
    result[(a > 0) & (b == 0)] = 255
    return result


def _remove_current_pixels(mask: np.ndarray, current_mask: np.ndarray) -> np.ndarray:
    binary = _as_binary_mask(mask)
    current = _as_binary_mask(current_mask)

    result = np.zeros_like(binary, dtype=np.uint8)
    result[(binary > 0) & (current == 0)] = 255
    return result


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


def remove_very_small_top2_components(
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    binary = _as_binary_mask(mask)

    cleaned = np.zeros_like(binary, dtype=np.uint8)
    removed = np.zeros_like(binary, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        should_remove = False

        if area < TOP2_REMOVE_AREA_ALWAYS_LT:
            should_remove = True

        if area < TOP2_REMOVE_AREA_THIN_LT and width <= TOP2_REMOVE_THIN_WIDTH_LE:
            should_remove = True

        if area < TOP2_REMOVE_AREA_THIN_LT and height <= TOP2_REMOVE_THIN_HEIGHT_LE:
            should_remove = True

        component_pixels = labels == label_id

        if should_remove:
            removed[component_pixels] = 255
        else:
            cleaned[component_pixels] = 255

    return cleaned, removed


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


def draw_removed_tiny_top2_on_crop(
    crop_bgr: np.ndarray,
    removed_mask: np.ndarray,
) -> np.ndarray:
    result = _to_bgr(crop_bgr)
    removed = _as_binary_mask(removed_mask)
    result[removed > 0] = (0, 0, 255)
    return result


def draw_kept_removed_overlay_on_crop(
    crop_bgr: np.ndarray,
    kept_mask: np.ndarray,
    removed_mask: np.ndarray,
) -> np.ndarray:
    result = _to_bgr(crop_bgr)
    kept = _as_binary_mask(kept_mask)
    removed = _as_binary_mask(removed_mask)

    result[kept > 0] = (0, 255, 0)
    result[removed > 0] = (0, 0, 255)
    return result


def build_top2_final_contour(
    crop_bgr: np.ndarray,
    current_pleura_mask: np.ndarray,
    top2_guided_mask: np.ndarray,
    top2_added_mask: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    current = _as_binary_mask(current_pleura_mask)
    guided = _as_binary_mask(top2_guided_mask)

    cleaned_guided_mask, removed_guided_mask = remove_very_small_top2_components(
        guided,
    )

    protected_mask = _merge_masks(current, cleaned_guided_mask)
    final_mask = _fill_internal_holes(protected_mask)

    removed_tiny_mask = _remove_current_pixels(removed_guided_mask, current)

    if top2_added_mask is None:
        added_to_current_mask = _subtract_masks(final_mask, current)
    else:
        added_to_current_mask = _as_binary_mask(top2_added_mask)
        added_to_current_mask = _subtract_masks(
            added_to_current_mask, removed_tiny_mask
        )

    contour_on_crop = draw_top2_final_contour_on_crop(crop_bgr, final_mask)
    raw_contour_on_crop = draw_top2_final_contour_on_crop(crop_bgr, guided)
    removed_tiny_on_crop = draw_removed_tiny_top2_on_crop(crop_bgr, removed_tiny_mask)
    kept_removed_overlay_on_crop = draw_kept_removed_overlay_on_crop(
        crop_bgr,
        final_mask,
        removed_tiny_mask,
    )

    return {
        "raw_final_top2_mask": guided,
        "cleaned_guided_mask": cleaned_guided_mask,
        "final_top2_mask": final_mask,
        "contour_on_crop": contour_on_crop,
        "raw_contour_on_crop": raw_contour_on_crop,
        "added_to_current_mask": added_to_current_mask,
        "removed_tiny_mask": removed_tiny_mask,
        "removed_tiny_on_crop": removed_tiny_on_crop,
        "kept_removed_overlay_on_crop": kept_removed_overlay_on_crop,
    }
