import cv2
import numpy as np


def mask_area(mask) -> int:
    if mask is None:
        return 0

    return int(np.count_nonzero(mask > 0))


def get_mask_bounds(mask):
    if mask is None:
        return None

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return {
        "min_x": int(np.min(xs)),
        "max_x": int(np.max(xs)),
        "min_y": int(np.min(ys)),
        "max_y": int(np.max(ys)),
        "width": int(np.max(xs) - np.min(xs) + 1),
        "height": int(np.max(ys) - np.min(ys) + 1),
        "area": int(len(xs)),
    }


def mask_density(mask):
    bounds = get_mask_bounds(mask)

    if bounds is None:
        return 0.0

    box_area = max(bounds["width"] * bounds["height"], 1)
    return float(bounds["area"] / box_area)


def mask_median_y(mask):
    if mask is None:
        return None

    ys, _ = np.where(mask > 0)

    if len(ys) == 0:
        return None

    return float(np.median(ys))


def empty_mask_like(mask):
    return np.zeros_like(mask, dtype=np.uint8)


def merge_masks(mask_a, mask_b):
    result = np.zeros_like(mask_a, dtype=np.uint8)

    if mask_a is not None:
        result[mask_a > 0] = 255

    if mask_b is not None:
        result[mask_b > 0] = 255

    return result


LEFT_LOW_FAR_ARTIFACT_GUARD_ENABLE = True
LEFT_LOW_FAR_ARTIFACT_MIN_MAIN_AREA = 2500
LEFT_LOW_FAR_ARTIFACT_MIN_MAIN_X = 330
LEFT_LOW_FAR_ARTIFACT_MIN_GAP_FROM_MAIN = 170
LEFT_LOW_FAR_ARTIFACT_MIN_BELOW_MAIN_CY = 70
LEFT_LOW_FAR_ARTIFACT_MAX_AREA = 1300
LEFT_LOW_FAR_ARTIFACT_MAX_WIDTH = 100
LEFT_LOW_FAR_ARTIFACT_MAX_HEIGHT = 55


def get_largest_component_geometry(mask):
    if mask is None or mask_area(mask) == 0:
        return None

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return None

    best_label = None
    best_area = -1

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area > best_area:
            best_area = area
            best_label = label

    if best_label is None:
        return None

    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    width = int(stats[best_label, cv2.CC_STAT_WIDTH])
    height = int(stats[best_label, cv2.CC_STAT_HEIGHT])

    return {
        "label": best_label,
        "area": int(best_area),
        "min_x": x,
        "max_x": x + width - 1,
        "min_y": y,
        "max_y": y + height - 1,
        "width": width,
        "height": height,
        "centroid_x": float(centroids[best_label][0]),
        "centroid_y": float(centroids[best_label][1]),
    }


def filter_left_low_far_artifact_components(mask, reference_mask):
    if not LEFT_LOW_FAR_ARTIFACT_GUARD_ENABLE:
        return mask, empty_mask_like(mask)

    if mask is None or reference_mask is None:
        return mask, empty_mask_like(mask)

    if mask_area(mask) == 0 or mask_area(reference_mask) == 0:
        return mask, empty_mask_like(mask)

    reference_main = get_largest_component_geometry(reference_mask)

    if reference_main is None:
        return mask, empty_mask_like(mask)

    if (
        reference_main["area"] < LEFT_LOW_FAR_ARTIFACT_MIN_MAIN_AREA
        or reference_main["min_x"] < LEFT_LOW_FAR_ARTIFACT_MIN_MAIN_X
    ):
        return mask, empty_mask_like(mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return mask, empty_mask_like(mask)

    kept = np.zeros_like(mask, dtype=np.uint8)
    removed = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):
        component_pixels = labels == label

        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        x2 = x + width - 1
        component_cy = float(centroids[label][1])

        far_left_from_main = (
            x2 <= reference_main["min_x"] - LEFT_LOW_FAR_ARTIFACT_MIN_GAP_FROM_MAIN
        )

        much_lower_than_main = (
            component_cy
            >= reference_main["centroid_y"] + LEFT_LOW_FAR_ARTIFACT_MIN_BELOW_MAIN_CY
            or y
            >= reference_main["centroid_y"] + LEFT_LOW_FAR_ARTIFACT_MIN_BELOW_MAIN_CY
        )

        size_matches = (
            area <= LEFT_LOW_FAR_ARTIFACT_MAX_AREA
            and width <= LEFT_LOW_FAR_ARTIFACT_MAX_WIDTH
            and height <= LEFT_LOW_FAR_ARTIFACT_MAX_HEIGHT
        )

        if size_matches and far_left_from_main and much_lower_than_main:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    return kept, removed


PRINCIPAL_UNDER_HORIZONTAL_TAIL_GUARD_ENABLE = True
PRINCIPAL_UNDER_HORIZONTAL_MAX_AREA = 520
PRINCIPAL_UNDER_HORIZONTAL_MAX_WIDTH = 70
PRINCIPAL_UNDER_HORIZONTAL_MAX_HEIGHT = 45
PRINCIPAL_UNDER_HORIZONTAL_MIN_HORIZONTAL_AREA = 180
PRINCIPAL_UNDER_HORIZONTAL_MIN_BELOW_HORIZONTAL_PX = 16
PRINCIPAL_UNDER_HORIZONTAL_MAX_LEFT_GAP_TO_HORIZONTAL = 28
PRINCIPAL_UNDER_HORIZONTAL_MIN_X_OVERLAP_FRAC = 0.10


def interval_overlap_fraction(*args):
    if len(args) == 4:
        a_min, a_max, b_min, b_max = args
        overlap = min(a_max, b_max) - max(a_min, b_min) + 1

        if overlap <= 0:
            return 0.0

        a_width = max(1, a_max - a_min + 1)
        b_width = max(1, b_max - b_min + 1)

        return float(overlap / max(1, min(a_width, b_width)))

    if len(args) == 5:
        x, start_x, end_x, width, max_distance = args

        if start_x <= x <= end_x:
            return 1.0

        distance = min(abs(x - start_x), abs(x - end_x))

        if distance > max_distance:
            return 0.0

        return float(1.0 - distance / max(max_distance, 1))

    raise TypeError(
        "interval_overlap_fraction accepts either 4 arguments or 5 arguments"
    )


def filter_principal_tail_under_horizontal(principal_mask, horizontal_rescue_mask):
    if not PRINCIPAL_UNDER_HORIZONTAL_TAIL_GUARD_ENABLE:
        return principal_mask, empty_mask_like(principal_mask)

    if principal_mask is None or horizontal_rescue_mask is None:
        return principal_mask, empty_mask_like(principal_mask)

    if mask_area(principal_mask) == 0 or mask_area(horizontal_rescue_mask) == 0:
        return principal_mask, empty_mask_like(principal_mask)

    horizontal_bounds = get_mask_bounds(horizontal_rescue_mask)
    horizontal_median_y = mask_median_y(horizontal_rescue_mask)

    if horizontal_bounds is None or horizontal_median_y is None:
        return principal_mask, empty_mask_like(principal_mask)

    if horizontal_bounds["area"] < PRINCIPAL_UNDER_HORIZONTAL_MIN_HORIZONTAL_AREA:
        return principal_mask, empty_mask_like(principal_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (principal_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 2:
        return principal_mask, empty_mask_like(principal_mask)

    kept = np.zeros_like(principal_mask, dtype=np.uint8)
    removed = np.zeros_like(principal_mask, dtype=np.uint8)

    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels)]
    largest_label = int(np.argmax(areas)) + 1

    for label in range(1, num_labels):
        component_pixels = labels == label

        if label == largest_label:
            kept[component_pixels] = 255
            continue

        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        x2 = x + width - 1
        y2 = y + height - 1

        component_median_y = float(centroids[label][1])

        is_small_tail_piece = (
            area <= PRINCIPAL_UNDER_HORIZONTAL_MAX_AREA
            and width <= PRINCIPAL_UNDER_HORIZONTAL_MAX_WIDTH
            and height <= PRINCIPAL_UNDER_HORIZONTAL_MAX_HEIGHT
        )

        overlap_frac = interval_overlap_fraction(
            x,
            x2,
            horizontal_bounds["min_x"],
            horizontal_bounds["max_x"],
            width,
        )

        left_gap_to_horizontal = horizontal_bounds["min_x"] - x2

        touches_horizontal_zone = (
            overlap_frac >= PRINCIPAL_UNDER_HORIZONTAL_MIN_X_OVERLAP_FRAC
            or (
                0
                <= left_gap_to_horizontal
                <= PRINCIPAL_UNDER_HORIZONTAL_MAX_LEFT_GAP_TO_HORIZONTAL
            )
        )

        is_below_accepted_horizontal = (
            component_median_y
            >= horizontal_median_y + PRINCIPAL_UNDER_HORIZONTAL_MIN_BELOW_HORIZONTAL_PX
            or y2
            >= horizontal_median_y
            + PRINCIPAL_UNDER_HORIZONTAL_MIN_BELOW_HORIZONTAL_PX
            + 8
        )

        if (
            is_small_tail_piece
            and touches_horizontal_zone
            and is_below_accepted_horizontal
        ):
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    if mask_area(removed) > 0.18 * mask_area(principal_mask):
        return principal_mask, empty_mask_like(principal_mask)

    return kept, removed


def draw_artifact_source_overlay(
    crop,
    principal_mask,
    horizontal_rescue_mask,
    secondary_mask_normal,
    gap_rescue_mask,
    secondary_rescue_mask,
    final_mask=None,
    traveler_points=None,
):
    output = crop.copy()
    overlay = output.copy()

    if principal_mask is not None:
        overlay[principal_mask > 0] = (255, 0, 0)

    if horizontal_rescue_mask is not None:
        overlay[horizontal_rescue_mask > 0] = (255, 255, 0)

    if secondary_mask_normal is not None:
        overlay[secondary_mask_normal > 0] = (0, 255, 0)

    if gap_rescue_mask is not None:
        overlay[gap_rescue_mask > 0] = (0, 165, 255)

    if secondary_rescue_mask is not None:
        overlay[secondary_rescue_mask > 0] = (255, 0, 255)

    output = cv2.addWeighted(overlay, 0.50, output, 0.50, 0)

    if final_mask is not None:
        final_u8 = np.zeros_like(final_mask, dtype=np.uint8)
        final_u8[final_mask > 0] = 255
        contours, _ = cv2.findContours(
            final_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, (0, 0, 255), 1)

    if traveler_points is not None:
        for point in traveler_points:
            px = None
            py = None

            if hasattr(point, "x") and hasattr(point, "y"):
                px = float(point.x)
                py = float(point.y)
            else:
                try:
                    values = np.asarray(point).reshape(-1)

                    if len(values) >= 2:
                        px = float(values[0])
                        py = float(values[1])
                except Exception:
                    px = None
                    py = None

            if px is None or py is None:
                continue

            x = int(round(px))
            y = int(round(py))

            if 0 <= y < output.shape[0] and 0 <= x < output.shape[1]:
                cv2.circle(output, (x, y), 1, (255, 255, 255), -1)

    cv2.putText(
        output,
        "blue=principal cyan=horizontal green=secondary "
        "orange=gap magenta=secondary_rescue",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output
