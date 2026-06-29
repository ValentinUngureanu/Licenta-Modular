from dataclasses import dataclass

import cv2
import numpy as np

MIN_BORDER_AREA_FRAC = 0.0007
SMALL_CONTOUR_AREA_FRAC = 3e-05

LEFT_MARGIN_SEARCH_FRAC = 0.07
DEFAULT_LEFT_BOUND_FRAC = 0.018
CROP_RIGHT_MARGIN_FRAC = 0.014

MIN_CROP_SIZE_FRAC = 0.1
SUSPECT_FULL_CROP_FRAC = 0.9

HORIZONTAL_KERNEL_WIDTH_FRAC = 0.0015
MORPH_ITERATIONS = 3

DEFAULT_BINARY_THRESHOLD = 128
BINARY_THRESHOLD = 110
MAX_PIXEL_VALUE = 255

EXTRA_CROP_TOP_FRAC = 0.035
EXTRA_CROP_BOTTOM_FRAC = 0.035
EXTRA_CROP_LEFT_FRAC = 0.0
EXTRA_CROP_RIGHT_FRAC = 0.0

EXTRA_CROP_TOP_PX = 0
EXTRA_CROP_BOTTOM_PX = 0
EXTRA_CROP_LEFT_PX = 0
EXTRA_CROP_RIGHT_PX = 0

HORIZONTAL_CROP_TOP_IGNORE_FRAC = 0.08
HORIZONTAL_CROP_BOTTOM_IGNORE_FRAC = 0.08
HORIZONTAL_MIN_ACTIVE_WIDTH_FRAC = 0.35
HORIZONTAL_MAX_FULL_WIDTH_FRAC = 0.96
HORIZONTAL_PAD_FRAC = 0.012
HORIZONTAL_SMOOTH_FRAC = 0.03

HORIZONTAL_THRESHOLD_PERCENTILES = [45, 50, 55, 60, 65, 70, 75]
HORIZONTAL_THRESHOLD_SCALE = 0.65


@dataclass
class CropBox:
    top: int
    left: int
    bottom: int
    right: int
    valid: bool = True

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)


def to_gray(image_bgr):
    if image_bgr.ndim == 2:
        return image_bgr.copy()

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def get_contours(binary):
    found = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(found) == 2:
        contours, _ = found
    else:
        _, contours, _ = found

    return contours


def clamp_box(box: CropBox, shape) -> CropBox:
    height, width = shape[:2]

    top = int(max(0, min(box.top, height - 1)))
    bottom = int(max(top + 1, min(box.bottom, height)))
    left = int(max(0, min(box.left, width - 1)))
    right = int(max(left + 1, min(box.right, width)))

    return CropBox(
        top=top,
        left=left,
        bottom=bottom,
        right=right,
        valid=box.valid,
    )


def is_box_valid(box: CropBox, shape) -> bool:
    height, width = shape[:2]
    min_size = int(max(30, MIN_CROP_SIZE_FRAC * min(height, width)))

    return box.width >= min_size and box.height >= min_size


def crop_is_suspect_full(box: CropBox, shape) -> bool:
    height, width = shape[:2]

    return (
        box.width >= SUSPECT_FULL_CROP_FRAC * width
        and box.height >= SUSPECT_FULL_CROP_FRAC * height
    )


def largest_true_segment(flags, min_len: int):
    best_start = None
    best_end = None
    best_len = 0
    start = None

    for index, value in enumerate(flags):
        if value and start is None:
            start = index
            continue

        if not value and start is not None:
            end = index
            length = end - start

            if length > best_len and length >= min_len:
                best_len = length
                best_start = start
                best_end = end

            start = None

    if start is not None:
        end = len(flags)
        length = end - start

        if length > best_len and length >= min_len:
            best_start = start
            best_end = end

    if best_start is None:
        return None

    return best_start, best_end


def estimate_crop_box_from_bar(gray) -> CropBox:
    height, width = gray.shape[:2]
    image_area = height * width

    min_border_area = max(150, int(MIN_BORDER_AREA_FRAC * image_area))
    small_contour_area = max(8, int(SMALL_CONTOUR_AREA_FRAC * image_area))

    left_margin_limit = max(30, int(LEFT_MARGIN_SEARCH_FRAC * width))
    default_left_bound = max(5, int(DEFAULT_LEFT_BOUND_FRAC * width))
    crop_right_margin = max(8, int(CROP_RIGHT_MARGIN_FRAC * width))
    kernel_width = max(2, int(HORIZONTAL_KERNEL_WIDTH_FRAC * width))

    _, image_binary = cv2.threshold(
        gray,
        DEFAULT_BINARY_THRESHOLD,
        MAX_PIXEL_VALUE,
        cv2.THRESH_BINARY,
    )

    _, threshold = cv2.threshold(
        gray,
        BINARY_THRESHOLD,
        MAX_PIXEL_VALUE,
        cv2.THRESH_BINARY,
    )

    contours = get_contours(threshold)
    large_mask = np.zeros_like(image_binary)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area <= min_border_area:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)

        if len(approx) in [2, 4]:
            cv2.drawContours(large_mask, [approx], 0, 255, -1)

    left_roi = large_mask[:, :left_margin_limit]
    left_pixels = cv2.findNonZero(left_roi)

    if left_pixels is None:
        left_bound = default_left_bound
    else:
        left_candidates = left_pixels[:, 0, 0]

        if len(left_candidates) == 0:
            left_bound = default_left_bound
        else:
            left_bound = int(np.percentile(left_candidates, 95))

    small_mask = np.zeros_like(image_binary)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= small_contour_area:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 1e-05 * perimeter, True)

        if len(approx) in [2, 4]:
            cv2.drawContours(small_mask, [approx], 0, 255, -1)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    temp = cv2.erode(small_mask, horizontal_kernel, iterations=MORPH_ITERATIONS)
    horizontal_lines = cv2.dilate(temp, horizontal_kernel, iterations=MORPH_ITERATIONS)

    columns = np.count_nonzero(horizontal_lines, axis=0)

    if np.max(columns) == 0:
        return CropBox(0, 0, height, width, False)

    bar_position = int(np.argmax(columns))
    bar_pixels = np.flatnonzero(horizontal_lines[:, bar_position] > 0)

    if len(bar_pixels) == 0:
        return CropBox(0, 0, height, width, False)

    box = CropBox(
        top=int(bar_pixels[0]),
        left=int(left_bound),
        bottom=int(bar_pixels[-1]),
        right=int(bar_position - crop_right_margin),
        valid=True,
    )

    box = clamp_box(box, gray.shape)

    if not is_box_valid(box, gray.shape) or crop_is_suspect_full(box, gray.shape):
        box.valid = False

    return box


def estimate_crop_box_from_activity(gray) -> CropBox:
    height, width = gray.shape[:2]
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    non_black = cv2.inRange(gray_blur, 13, 244)

    row_density = np.count_nonzero(non_black, axis=1) / max(width, 1)
    col_density = np.count_nonzero(non_black, axis=0) / max(height, 1)

    row_smooth = cv2.blur(
        row_density.astype(np.float32).reshape(-1, 1),
        (1, 21),
    ).reshape(-1)

    col_smooth = cv2.blur(
        col_density.astype(np.float32).reshape(1, -1),
        (31, 1),
    ).reshape(-1)

    row_threshold = max(0.025, float(np.percentile(row_smooth, 65)) * 0.45)
    col_threshold = max(0.02, float(np.percentile(col_smooth, 65)) * 0.45)

    row_segment = largest_true_segment(
        row_smooth > row_threshold,
        int(0.25 * height),
    )

    col_segment = largest_true_segment(
        col_smooth > col_threshold,
        int(0.35 * width),
    )

    if row_segment is None or col_segment is None:
        return CropBox(0, 0, height, width, False)

    top, bottom = row_segment
    left, right = col_segment

    pad_y = int(0.015 * height)
    pad_x = int(0.015 * width)

    box = CropBox(
        top=max(0, top - pad_y),
        left=max(0, left - pad_x),
        bottom=min(height, bottom + pad_y),
        right=min(width, right + pad_x),
        valid=True,
    )

    box = clamp_box(box, gray.shape)

    if not is_box_valid(box, gray.shape) or crop_is_suspect_full(box, gray.shape):
        box.valid = False

    return box


def choose_vertical_crop_box(gray) -> CropBox:
    bar_box = estimate_crop_box_from_bar(gray)
    activity_box = estimate_crop_box_from_activity(gray)

    candidates = []

    if bar_box.valid:
        candidates.append(bar_box)

    if activity_box.valid:
        candidates.append(activity_box)

    if len(candidates) == 0:
        return CropBox(0, 0, gray.shape[0], gray.shape[1], False)

    def score_box(box: CropBox) -> float:
        height, width = gray.shape[:2]

        area_frac = box.width * box.height / max(height * width, 1)
        center_y = (box.top + box.bottom) / 2.0 / max(height, 1)
        center_x = (box.left + box.right) / 2.0 / max(width, 1)

        score = 0.0
        score += 2.0 * min(1.0, box.width / max(0.55 * width, 1))
        score += 2.0 * min(1.0, box.height / max(0.45 * height, 1))
        score -= 1.2 * abs(area_frac - 0.55)
        score -= 0.6 * abs(center_x - 0.5)

        if 0.25 <= center_y <= 0.7:
            score += 0.5

        return score

    return max(candidates, key=score_box)


def estimate_left_right_from_vertical_crop(gray_vertical) -> CropBox:
    height, width = gray_vertical.shape[:2]

    if height < 30 or width < 30:
        return CropBox(0, 0, height, width, False)

    y1 = int(round(HORIZONTAL_CROP_TOP_IGNORE_FRAC * height))
    y2 = int(round((1.0 - HORIZONTAL_CROP_BOTTOM_IGNORE_FRAC) * height))

    y1 = max(0, min(y1, height - 2))
    y2 = max(y1 + 1, min(y2, height))

    roi = gray_vertical[y1:y2, :].copy()

    if roi.size == 0:
        return CropBox(0, 0, height, width, False)

    blur = cv2.GaussianBlur(roi, (5, 5), 0)

    non_black = cv2.inRange(blur, 11, 249)
    non_black_density = np.count_nonzero(non_black, axis=0) / max(roi.shape[0], 1)

    edges = cv2.Canny(blur, 30, 90)
    edge_density = np.count_nonzero(edges > 0, axis=0) / max(roi.shape[0], 1)

    if np.max(edge_density) > 0:
        edge_density = edge_density / np.max(edge_density)

    signal = 0.8 * non_black_density + 0.2 * edge_density
    smooth_width = max(9, int(round(HORIZONTAL_SMOOTH_FRAC * width)))

    if smooth_width % 2 == 0:
        smooth_width += 1

    signal_smooth = cv2.blur(
        signal.astype(np.float32).reshape(1, -1),
        (smooth_width, 1),
    ).reshape(-1)

    candidates = []
    min_len = max(20, int(round(HORIZONTAL_MIN_ACTIVE_WIDTH_FRAC * width)))

    for percentile in HORIZONTAL_THRESHOLD_PERCENTILES:
        base = float(np.percentile(signal_smooth, percentile))
        threshold = max(0.01, base * HORIZONTAL_THRESHOLD_SCALE)
        segment = largest_true_segment(signal_smooth > threshold, min_len)

        if segment is None:
            continue

        left, right = segment
        pad = max(4, int(round(HORIZONTAL_PAD_FRAC * width)))

        left = max(0, left - pad)
        right = min(width, right + pad)

        crop_width = max(1, right - left)
        width_frac = crop_width / max(width, 1)
        center_frac = (left + right) / 2.0 / max(width, 1)

        if width_frac < HORIZONTAL_MIN_ACTIVE_WIDTH_FRAC:
            continue

        score = 0.0
        score += 2.5 * min(1.0, width_frac / 0.75)
        score -= 1.2 * abs(center_frac - 0.5)
        score -= 2.0 * max(0.0, width_frac - HORIZONTAL_MAX_FULL_WIDTH_FRAC)
        score += 0.01 * percentile

        candidates.append((score, left, right))

    if len(candidates) == 0:
        threshold = max(0.01, float(np.median(signal_smooth)))
        xs = np.where(signal_smooth > threshold)[0]

        if len(xs) == 0:
            return CropBox(0, 0, height, width, False)

        pad = max(4, int(round(HORIZONTAL_PAD_FRAC * width)))
        left = max(0, int(np.min(xs)) - pad)
        right = min(width, int(np.max(xs)) + 1 + pad)

        return clamp_box(
            CropBox(0, left, height, right, True),
            gray_vertical.shape,
        )

    _, left, right = max(candidates, key=lambda item: item[0])

    box = CropBox(0, left, height, right, True)
    box = clamp_box(box, gray_vertical.shape)

    if not is_box_valid(box, gray_vertical.shape):
        box.valid = False

    return box


def estimate_crop_box(image_bgr) -> CropBox:
    gray = to_gray(image_bgr)
    height, width = gray.shape[:2]

    vertical_base = choose_vertical_crop_box(gray)

    vertical_box = CropBox(
        top=vertical_base.top,
        left=0,
        bottom=vertical_base.bottom,
        right=width,
        valid=vertical_base.valid,
    )

    vertical_box = clamp_box(vertical_box, gray.shape)

    gray_vertical = gray[
        vertical_box.top : vertical_box.bottom,
        vertical_box.left : vertical_box.right,
    ].copy()

    horizontal_box = estimate_left_right_from_vertical_crop(gray_vertical)

    if horizontal_box.valid:
        final_box = CropBox(
            top=vertical_box.top,
            left=horizontal_box.left,
            bottom=vertical_box.bottom,
            right=horizontal_box.right,
            valid=True,
        )
    else:
        final_box = CropBox(
            top=vertical_box.top,
            left=0,
            bottom=vertical_box.bottom,
            right=width,
            valid=vertical_box.valid,
        )

    final_box = clamp_box(final_box, gray.shape)

    add_top = max(EXTRA_CROP_TOP_PX, int(round(EXTRA_CROP_TOP_FRAC * final_box.height)))
    add_bottom = max(
        EXTRA_CROP_BOTTOM_PX, int(round(EXTRA_CROP_BOTTOM_FRAC * final_box.height))
    )
    add_left = max(
        EXTRA_CROP_LEFT_PX, int(round(EXTRA_CROP_LEFT_FRAC * final_box.width))
    )
    add_right = max(
        EXTRA_CROP_RIGHT_PX, int(round(EXTRA_CROP_RIGHT_FRAC * final_box.width))
    )

    final_box = CropBox(
        top=final_box.top + add_top,
        left=final_box.left + add_left,
        bottom=final_box.bottom - add_bottom,
        right=final_box.right - add_right,
        valid=final_box.valid,
    )

    return clamp_box(final_box, gray.shape)


def apply_crop(image_bgr, crop_box: CropBox):
    return image_bgr[
        crop_box.top : crop_box.bottom,
        crop_box.left : crop_box.right,
    ].copy()


def crop_border(image_bgr):
    crop_box = estimate_crop_box(image_bgr)
    crop = apply_crop(image_bgr, crop_box)

    return crop, crop_box


def crop_ultrasound(image_bgr):
    return crop_border(image_bgr)


def draw_crop_box_on_original(image_bgr, crop_box: CropBox):
    result = image_bgr.copy()
    color = (0, 255, 255) if crop_box.valid else (0, 0, 255)

    cv2.rectangle(
        result,
        (crop_box.left, crop_box.top),
        (crop_box.right - 1, crop_box.bottom - 1),
        color,
        2,
    )

    text = (
        f"crop valid={crop_box.valid} "
        f"x={crop_box.left}:{crop_box.right} "
        f"y={crop_box.top}:{crop_box.bottom}"
    )

    cv2.putText(
        result,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    return result


def draw_crop_box(image_bgr, crop_box: CropBox):
    return draw_crop_box_on_original(image_bgr, crop_box)
