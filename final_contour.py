import cv2
import numpy as np

from postprocessing import empty_mask_like, mask_area

FINAL_MASK_COLOR = (0, 255, 0)
FINAL_CONTOUR_COLOR = (0, 0, 255)

FINAL_MIN_COMPONENT_AREA = 8
FINAL_MIN_COMPONENT_WIDTH_PX = 4
FINAL_MIN_COMPONENT_WIDTH_FRAC = 0.006

FINAL_CLOSE_KERNEL_WIDTH = 5
FINAL_CLOSE_KERNEL_HEIGHT = 3
FINAL_CLOSE_ITERATIONS = 1

FINAL_CONTOUR_THICKNESS = 1

SMALL_COMPONENT_CLEAN_ENABLE = True
SMALL_COMPONENT_MIN_AREA = 28
SMALL_COMPONENT_MIN_WIDTH = 7
SMALL_COMPONENT_MIN_HEIGHT = 4
SMALL_COMPONENT_REMOVE_COMPACT_AREA = 35
SMALL_COMPONENT_REMOVE_COMPACT_MAX_WIDTH = 14
SMALL_COMPONENT_REMOVE_COMPACT_MAX_HEIGHT = 14


def normalize_binary_mask(mask):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return (mask > 0).astype(np.uint8) * 255


def to_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image.copy()


def remove_tiny_components(mask):
    mask = normalize_binary_mask(mask)
    _, width = mask.shape[:2]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask, dtype=np.uint8)

    min_width = max(
        FINAL_MIN_COMPONENT_WIDTH_PX,
        int(round(FINAL_MIN_COMPONENT_WIDTH_FRAC * width)),
    )

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])

        if area < FINAL_MIN_COMPONENT_AREA:
            continue

        if component_width < min_width and area < 2 * FINAL_MIN_COMPONENT_AREA:
            continue

        cleaned[labels == label] = 255

    return cleaned


def close_small_gaps(mask):
    mask = normalize_binary_mask(mask)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (FINAL_CLOSE_KERNEL_WIDTH, FINAL_CLOSE_KERNEL_HEIGHT),
    )

    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=FINAL_CLOSE_ITERATIONS,
    )

    return normalize_binary_mask(closed)


def build_final_mask(merged_mask):
    merged_mask = normalize_binary_mask(merged_mask)

    cleaned = remove_tiny_components(merged_mask)
    closed = close_small_gaps(cleaned)
    cleaned_again = remove_tiny_components(closed)

    return cleaned_again


def get_external_contours(mask):
    mask = normalize_binary_mask(mask)

    found = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(found) == 2:
        contours, _ = found
    else:
        _, contours, _ = found

    return contours


def build_final_contour(merged_mask):
    final_mask = build_final_mask(merged_mask)
    contours = get_external_contours(final_mask)

    return {
        "final_mask": final_mask,
        "contours": contours,
        "raw_points": np.empty((0, 2), dtype=np.int32),
        "contour_points": np.empty((0, 2), dtype=np.int32),
    }


def draw_mask_overlay(base_image, mask, color, alpha=0.55):
    result = to_bgr(base_image)
    mask = normalize_binary_mask(mask)

    mask_bool = mask > 0

    if np.count_nonzero(mask_bool) == 0:
        return result

    color_array = np.array(color, dtype=np.float32)
    result_float = result.astype(np.float32)

    result_float[mask_bool] = (1.0 - alpha) * result_float[
        mask_bool
    ] + alpha * color_array

    return np.clip(result_float, 0, 255).astype(np.uint8)


def draw_contours_around_mask(base_image, mask):
    result = to_bgr(base_image)
    mask = normalize_binary_mask(mask)

    contours = get_external_contours(mask)

    if len(contours) == 0:
        return result

    cv2.drawContours(
        result,
        contours,
        contourIdx=-1,
        color=FINAL_CONTOUR_COLOR,
        thickness=FINAL_CONTOUR_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    return result


def project_mask_to_original(mask_crop, original_shape, crop_box):
    mask_crop = normalize_binary_mask(mask_crop)

    original_height, original_width = original_shape[:2]
    result = np.zeros((original_height, original_width), dtype=np.uint8)

    crop_height, crop_width = mask_crop.shape[:2]

    top = int(crop_box.top)
    left = int(crop_box.left)

    bottom = min(original_height, top + crop_height)
    right = min(original_width, left + crop_width)

    valid_height = max(0, bottom - top)
    valid_width = max(0, right - left)

    if valid_height == 0 or valid_width == 0:
        return result

    result[top:bottom, left:right] = mask_crop[:valid_height, :valid_width]

    return result


def draw_final_mask_on_crop(crop_image, final_mask):
    return draw_mask_overlay(
        crop_image,
        final_mask,
        FINAL_MASK_COLOR,
        alpha=0.55,
    )


def draw_final_contour_on_crop(crop_image, final_mask, contour_points=None):
    return draw_contours_around_mask(
        crop_image,
        final_mask,
    )


def draw_final_mask_on_original(original_image, final_mask_crop, crop_box):
    projected_mask = project_mask_to_original(
        final_mask_crop,
        original_image.shape,
        crop_box,
    )

    return draw_mask_overlay(
        original_image,
        projected_mask,
        FINAL_MASK_COLOR,
        alpha=0.55,
    )


def draw_final_contour_on_original(
    original_image,
    final_mask_crop,
    contour_points_crop,
    crop_box,
):
    projected_mask = project_mask_to_original(
        final_mask_crop,
        original_image.shape,
        crop_box,
    )

    return draw_contours_around_mask(
        original_image,
        projected_mask,
    )


def remove_very_small_components(mask):
    if not SMALL_COMPONENT_CLEAN_ENABLE:
        return mask, empty_mask_like(mask)

    if mask is None or mask_area(mask) == 0:
        return mask, empty_mask_like(mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )

    cleaned = np.zeros_like(mask, dtype=np.uint8)
    removed = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        component_pixels = labels == label

        is_tiny = (
            area < SMALL_COMPONENT_MIN_AREA
            or width < SMALL_COMPONENT_MIN_WIDTH
            or height < SMALL_COMPONENT_MIN_HEIGHT
        )

        is_compact_speckle = (
            area < SMALL_COMPONENT_REMOVE_COMPACT_AREA
            and width <= SMALL_COMPONENT_REMOVE_COMPACT_MAX_WIDTH
            and height <= SMALL_COMPONENT_REMOVE_COMPACT_MAX_HEIGHT
        )

        if is_tiny or is_compact_speckle:
            removed[component_pixels] = 255
        else:
            cleaned[component_pixels] = 255

    return cleaned, removed
