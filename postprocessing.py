import cv2
import numpy as np

SMALL_COMPONENT_CLEAN_ENABLE = True
SMALL_COMPONENT_MIN_AREA = 28
SMALL_COMPONENT_MIN_WIDTH = 7
SMALL_COMPONENT_MIN_HEIGHT = 4
SMALL_COMPONENT_REMOVE_COMPACT_AREA = 35
SMALL_COMPONENT_REMOVE_COMPACT_MAX_WIDTH = 14
SMALL_COMPONENT_REMOVE_COMPACT_MAX_HEIGHT = 14

HORIZONTAL_RESCUE_MAX_AREA_FACTOR = 0.85
HORIZONTAL_RESCUE_MAX_WIDTH_FACTOR = 1.75
HORIZONTAL_RESCUE_MAX_WIDTH_PX = 240
HORIZONTAL_RESCUE_MAX_HEIGHT_FRAC = 0.18
HORIZONTAL_RESCUE_MIN_EXTENSION_PX = 8
HORIZONTAL_RESCUE_MAX_BOTH_SIDE_EXTENSION_PX = 50

SECONDARY_RESCUE_OVERGROWTH_AREA_RATIO = 3.0
SECONDARY_RESCUE_OVERGROWTH_MIN_AREA = 9000
SECONDARY_RESCUE_OVERGROWTH_MIN_DENSITY = 0.18
SECONDARY_RESCUE_OVERGROWTH_MIN_HEIGHT = 125
SECONDARY_RESCUE_MIN_KEEP_FRAC_WHEN_NOT_OVERGROWN = 0.65
SECONDARY_RESCUE_MIN_WIDTH_KEEP_FRAC_WHEN_NOT_OVERGROWN = 0.88


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


def should_accept_horizontal_rescue(horizontal_result, principal_mask) -> bool:
    rescue_mask = horizontal_result["rescue_mask"]
    rescue_area = mask_area(rescue_mask)

    if rescue_area == 0:
        return False

    principal_area = mask_area(principal_mask)
    principal_bounds = get_mask_bounds(principal_mask)
    rescue_bounds = get_mask_bounds(rescue_mask)

    if principal_area == 0 or principal_bounds is None or rescue_bounds is None:
        return False

    image_height, _ = principal_mask.shape[:2]

    left_gain = max(0, principal_bounds["min_x"] - rescue_bounds["min_x"])
    right_gain = max(0, rescue_bounds["max_x"] - principal_bounds["max_x"])
    max_gain = max(left_gain, right_gain)

    if max_gain < HORIZONTAL_RESCUE_MIN_EXTENSION_PX:
        return False

    if (
        left_gain > HORIZONTAL_RESCUE_MAX_BOTH_SIDE_EXTENSION_PX
        and right_gain > HORIZONTAL_RESCUE_MAX_BOTH_SIDE_EXTENSION_PX
    ):
        return False

    max_allowed_area = max(
        350,
        int(round(HORIZONTAL_RESCUE_MAX_AREA_FACTOR * principal_area)),
    )

    if rescue_area > max_allowed_area:
        return False

    max_allowed_width = max(
        HORIZONTAL_RESCUE_MAX_WIDTH_PX,
        int(round(HORIZONTAL_RESCUE_MAX_WIDTH_FACTOR * principal_bounds["width"])),
    )

    if rescue_bounds["width"] > max_allowed_width:
        return False

    max_allowed_height = max(
        36,
        int(round(HORIZONTAL_RESCUE_MAX_HEIGHT_FRAC * image_height)),
    )

    if rescue_bounds["height"] > max_allowed_height:
        return False

    return True


def should_accept_secondary_rescue(
    principal_mask,
    secondary_mask_before_rescue,
    merged_before_rescue,
    secondary_rescue_result,
) -> bool:
    principal_area = mask_area(principal_mask)
    secondary_before_area = mask_area(secondary_mask_before_rescue)
    secondary_after_area = mask_area(secondary_rescue_result["secondary_mask"])
    removed_area = mask_area(secondary_rescue_result["removed_secondary_mask"])
    rescue_area = mask_area(secondary_rescue_result["rescue_mask"])

    if principal_area == 0:
        return False

    if secondary_before_area == 0:
        max_rescue_area = max(600, int(round(1.20 * principal_area)))
        return rescue_area > 0 and rescue_area <= max_rescue_area

    secondary_before_bounds = get_mask_bounds(secondary_mask_before_rescue)
    secondary_before_density = mask_density(secondary_mask_before_rescue)

    overgrown_secondary = (
        secondary_before_area >= SECONDARY_RESCUE_OVERGROWTH_MIN_AREA
        and secondary_before_area
        > SECONDARY_RESCUE_OVERGROWTH_AREA_RATIO * principal_area
        and secondary_before_bounds is not None
        and (
            secondary_before_density >= SECONDARY_RESCUE_OVERGROWTH_MIN_DENSITY
            or secondary_before_bounds["height"]
            >= SECONDARY_RESCUE_OVERGROWTH_MIN_HEIGHT
        )
    )

    if overgrown_secondary:
        return True

    if removed_area > 0:
        min_allowed_after_area = int(
            round(
                SECONDARY_RESCUE_MIN_KEEP_FRAC_WHEN_NOT_OVERGROWN
                * secondary_before_area
            )
        )

        if secondary_after_area < min_allowed_after_area:
            return False

        before_bounds = get_mask_bounds(merged_before_rescue)
        after_bounds = get_mask_bounds(secondary_rescue_result["merged_mask"])

        if before_bounds is not None and after_bounds is not None:
            min_allowed_width = int(
                round(
                    SECONDARY_RESCUE_MIN_WIDTH_KEEP_FRAC_WHEN_NOT_OVERGROWN
                    * before_bounds["width"]
                )
            )

            if after_bounds["width"] < min_allowed_width:
                return False

    return True


RIGHT_ISOLATED_SECONDARY_RESCUE_GUARD_ENABLE = True
RIGHT_ISOLATED_SECONDARY_RESCUE_MIN_GAP_PX = 60
RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_AREA = 650
RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_WIDTH = 130
RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_HEIGHT = 80
RIGHT_ISOLATED_SECONDARY_RESCUE_MIN_BELOW_MEDIAN_PX = 45


def filter_right_isolated_secondary_rescue(rescue_mask, base_mask):
    if not RIGHT_ISOLATED_SECONDARY_RESCUE_GUARD_ENABLE:
        return rescue_mask, empty_mask_like(rescue_mask)

    if rescue_mask is None or base_mask is None:
        return rescue_mask, empty_mask_like(rescue_mask)

    if mask_area(rescue_mask) == 0 or mask_area(base_mask) == 0:
        return rescue_mask, empty_mask_like(rescue_mask)

    base_bounds = get_mask_bounds(base_mask)
    base_median_y = mask_median_y(base_mask)

    if base_bounds is None or base_median_y is None:
        return rescue_mask, empty_mask_like(rescue_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (rescue_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    kept = np.zeros_like(rescue_mask, dtype=np.uint8)
    removed = np.zeros_like(rescue_mask, dtype=np.uint8)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        x + width - 1
        component_median_y = float(centroids[label][1])
        component_pixels = labels == label

        right_gap = x - base_bounds["max_x"] - 1

        is_right_isolated = right_gap >= RIGHT_ISOLATED_SECONDARY_RESCUE_MIN_GAP_PX

        is_small_enough = (
            area <= RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_AREA
            and width <= RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_WIDTH
            and height <= RIGHT_ISOLATED_SECONDARY_RESCUE_MAX_HEIGHT
        )

        is_much_lower = (
            component_median_y
            >= base_median_y + RIGHT_ISOLATED_SECONDARY_RESCUE_MIN_BELOW_MEDIAN_PX
        )

        if is_right_isolated and is_small_enough and is_much_lower:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    return kept, removed


GAP_FLOATING_RIGHT_GUARD_ENABLE = True
GAP_FLOATING_RIGHT_MIN_AREA = 250
GAP_FLOATING_RIGHT_MAX_AREA = 1200
GAP_FLOATING_RIGHT_MIN_WIDTH = 35
GAP_FLOATING_RIGHT_MAX_WIDTH = 140
GAP_FLOATING_RIGHT_MAX_HEIGHT = 55
GAP_FLOATING_RIGHT_MIN_RIGHT_GAP_FROM_PRINCIPAL = 18
GAP_FLOATING_RIGHT_LEFT_SUPPORT_WINDOW = 90
GAP_FLOATING_RIGHT_LEFT_SUPPORT_Y_BAND = 35
GAP_FLOATING_RIGHT_MIN_LEFT_SUPPORT_PIXELS = 8

GAP_UPPER_RIGHT_GUARD_ENABLE = True
GAP_UPPER_RIGHT_MIN_AREA = 120
GAP_UPPER_RIGHT_MAX_AREA = 900
GAP_UPPER_RIGHT_MIN_WIDTH = 12
GAP_UPPER_RIGHT_MAX_WIDTH = 120
GAP_UPPER_RIGHT_MAX_HEIGHT = 60
GAP_UPPER_RIGHT_MIN_RIGHT_GAP_FROM_PRINCIPAL = 8
GAP_UPPER_RIGHT_CONTEXT_WINDOW = 120
GAP_UPPER_RIGHT_MIN_CONTEXT_PIXELS = 20
GAP_UPPER_RIGHT_MIN_ABOVE_LOCAL_PX = 35


def filter_upper_right_gap_rescue(
    gap_rescue_mask,
    principal_after_horizontal_mask,
    secondary_mask_normal,
):
    if not GAP_UPPER_RIGHT_GUARD_ENABLE:
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    if gap_rescue_mask is None or principal_after_horizontal_mask is None:
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    if (
        mask_area(gap_rescue_mask) == 0
        or mask_area(principal_after_horizontal_mask) == 0
    ):
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    principal_bounds = get_mask_bounds(principal_after_horizontal_mask)

    if principal_bounds is None:
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    support_mask = np.zeros_like(gap_rescue_mask, dtype=np.uint8)
    support_mask[principal_after_horizontal_mask > 0] = 255

    if secondary_mask_normal is not None:
        support_mask[secondary_mask_normal > 0] = 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (gap_rescue_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    kept = np.zeros_like(gap_rescue_mask, dtype=np.uint8)
    removed = np.zeros_like(gap_rescue_mask, dtype=np.uint8)

    image_h, image_w = gap_rescue_mask.shape[:2]

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component_pixels = labels == label
        component_median_y = float(centroids[label][1])

        right_gap_from_principal = x - principal_bounds["max_x"] - 1

        size_matches = (
            area >= GAP_UPPER_RIGHT_MIN_AREA
            and area <= GAP_UPPER_RIGHT_MAX_AREA
            and width >= GAP_UPPER_RIGHT_MIN_WIDTH
            and width <= GAP_UPPER_RIGHT_MAX_WIDTH
            and height <= GAP_UPPER_RIGHT_MAX_HEIGHT
        )

        is_right_of_principal = (
            right_gap_from_principal >= GAP_UPPER_RIGHT_MIN_RIGHT_GAP_FROM_PRINCIPAL
        )

        x1_context = max(0, x - GAP_UPPER_RIGHT_CONTEXT_WINDOW)
        x2_context = min(image_w, x + 1)

        context_region = support_mask[:, x1_context:x2_context]
        context_ys, context_xs = np.where(context_region > 0)

        has_context = len(context_ys) >= GAP_UPPER_RIGHT_MIN_CONTEXT_PIXELS

        if has_context:
            local_reference_y = float(np.median(context_ys))
        else:
            local_reference_y = None

        is_much_above_local_direction = (
            local_reference_y is not None
            and component_median_y
            <= local_reference_y - GAP_UPPER_RIGHT_MIN_ABOVE_LOCAL_PX
        )

        if (
            size_matches
            and is_right_of_principal
            and has_context
            and is_much_above_local_direction
        ):
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    return kept, removed


def filter_floating_right_gap_rescue(
    gap_rescue_mask,
    principal_after_horizontal_mask,
    secondary_mask_normal,
):
    if not GAP_FLOATING_RIGHT_GUARD_ENABLE:
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    if gap_rescue_mask is None or principal_after_horizontal_mask is None:
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    if (
        mask_area(gap_rescue_mask) == 0
        or mask_area(principal_after_horizontal_mask) == 0
    ):
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    principal_bounds = get_mask_bounds(principal_after_horizontal_mask)

    if principal_bounds is None:
        return gap_rescue_mask, empty_mask_like(gap_rescue_mask)

    support_mask = np.zeros_like(gap_rescue_mask, dtype=np.uint8)
    support_mask[principal_after_horizontal_mask > 0] = 255

    if secondary_mask_normal is not None:
        support_mask[secondary_mask_normal > 0] = 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (gap_rescue_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    kept = np.zeros_like(gap_rescue_mask, dtype=np.uint8)
    removed = np.zeros_like(gap_rescue_mask, dtype=np.uint8)

    height_img, width_img = gap_rescue_mask.shape[:2]

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        x + width - 1
        y + height - 1
        component_median_y = float(centroids[label][1])
        component_pixels = labels == label

        right_gap_from_principal = x - principal_bounds["max_x"] - 1

        size_matches = (
            area >= GAP_FLOATING_RIGHT_MIN_AREA
            and area <= GAP_FLOATING_RIGHT_MAX_AREA
            and width >= GAP_FLOATING_RIGHT_MIN_WIDTH
            and width <= GAP_FLOATING_RIGHT_MAX_WIDTH
            and height <= GAP_FLOATING_RIGHT_MAX_HEIGHT
        )

        is_right_of_principal = (
            right_gap_from_principal >= GAP_FLOATING_RIGHT_MIN_RIGHT_GAP_FROM_PRINCIPAL
        )

        y1_support = max(
            0, int(round(component_median_y)) - GAP_FLOATING_RIGHT_LEFT_SUPPORT_Y_BAND
        )
        y2_support = min(
            height_img,
            int(round(component_median_y)) + GAP_FLOATING_RIGHT_LEFT_SUPPORT_Y_BAND + 1,
        )
        x1_support = max(0, x - GAP_FLOATING_RIGHT_LEFT_SUPPORT_WINDOW)
        x2_support = max(0, x)

        left_support_pixels = 0

        if x2_support > x1_support and y2_support > y1_support:
            left_support_pixels = int(
                np.count_nonzero(
                    support_mask[y1_support:y2_support, x1_support:x2_support] > 0
                )
            )

        lacks_local_left_support = (
            left_support_pixels < GAP_FLOATING_RIGHT_MIN_LEFT_SUPPORT_PIXELS
        )

        if size_matches and is_right_of_principal and lacks_local_left_support:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    return kept, removed


HORIZONTAL_LAYERED_TAIL_GUARD_ENABLE = True
HORIZONTAL_LAYERED_TAIL_MIN_COMPONENTS = 3
HORIZONTAL_LAYERED_TAIL_MIN_UPPER_AREA = 180
HORIZONTAL_LAYERED_TAIL_MAX_UPPER_HEIGHT = 24
HORIZONTAL_LAYERED_TAIL_MAX_LOWER_AREA = 900
HORIZONTAL_LAYERED_TAIL_MAX_LOWER_HEIGHT = 34
HORIZONTAL_LAYERED_TAIL_MIN_Y_GAP = 16
HORIZONTAL_LAYERED_TAIL_MIN_OVERLAP_FRAC = 0.35
HORIZONTAL_LAYERED_TAIL_MAX_RIGHT_GAP_FROM_UPPER = 95
HORIZONTAL_LAYERED_TAIL_MIN_RIGHT_REGION_GAIN = -80


def x_overlap_fraction(a, b):
    left = max(a["x"], b["x"])
    right = min(a["x2"], b["x2"])

    if right < left:
        return 0.0

    overlap = right - left + 1
    min_width = max(min(a["width"], b["width"]), 1)

    return float(overlap / min_width)


def filter_layered_horizontal_tail(horizontal_rescue_mask, principal_mask):
    if not HORIZONTAL_LAYERED_TAIL_GUARD_ENABLE:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    if horizontal_rescue_mask is None or principal_mask is None:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    if mask_area(horizontal_rescue_mask) == 0 or mask_area(principal_mask) == 0:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    principal_bounds = get_mask_bounds(principal_mask)

    if principal_bounds is None:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (horizontal_rescue_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if num_labels - 1 < HORIZONTAL_LAYERED_TAIL_MIN_COMPONENTS:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    components = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        x2 = x + width - 1
        y2 = y + height - 1

        components.append(
            {
                "label": label,
                "area": area,
                "x": x,
                "y": y,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
                "centroid_y": float(centroids[label][1]),
            }
        )

    upper_candidates = []

    for component in components:
        is_upper_shape = (
            component["area"] >= HORIZONTAL_LAYERED_TAIL_MIN_UPPER_AREA
            and component["height"] <= HORIZONTAL_LAYERED_TAIL_MAX_UPPER_HEIGHT
        )

        is_near_right_end_of_principal = (
            component["x2"]
            >= principal_bounds["max_x"] + HORIZONTAL_LAYERED_TAIL_MIN_RIGHT_REGION_GAIN
        )

        if is_upper_shape and is_near_right_end_of_principal:
            upper_candidates.append(component)

    if len(upper_candidates) == 0:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    upper = min(upper_candidates, key=lambda item: item["centroid_y"])

    kept = np.zeros_like(horizontal_rescue_mask, dtype=np.uint8)
    removed = np.zeros_like(horizontal_rescue_mask, dtype=np.uint8)

    for component in components:
        pixels = labels == component["label"]

        if component["label"] == upper["label"]:
            kept[pixels] = 255
            continue

        is_lower_than_upper = (
            component["centroid_y"]
            >= upper["centroid_y"] + HORIZONTAL_LAYERED_TAIL_MIN_Y_GAP
        )

        is_small_or_medium_tail = (
            component["area"] <= HORIZONTAL_LAYERED_TAIL_MAX_LOWER_AREA
            and component["height"] <= HORIZONTAL_LAYERED_TAIL_MAX_LOWER_HEIGHT
        )

        overlap_frac = x_overlap_fraction(component, upper)

        overlaps_upper = overlap_frac >= HORIZONTAL_LAYERED_TAIL_MIN_OVERLAP_FRAC

        right_gap_from_upper = component["x"] - upper["x2"] - 1

        is_right_tail_after_upper = (
            right_gap_from_upper >= 0
            and right_gap_from_upper <= HORIZONTAL_LAYERED_TAIL_MAX_RIGHT_GAP_FROM_UPPER
        )

        if (
            is_lower_than_upper
            and is_small_or_medium_tail
            and (overlaps_upper or is_right_tail_after_upper)
        ):
            removed[pixels] = 255
        else:
            kept[pixels] = 255

    if mask_area(kept) < 0.20 * mask_area(horizontal_rescue_mask):
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    return kept, removed


SECONDARY_AFTER_HORIZONTAL_TAIL_GUARD_ENABLE = True
SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_HORIZONTAL_AREA = 150
SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_AREA = 220
SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_WIDTH = 45
SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_HEIGHT = 28
SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_RIGHT_GAP = -2
SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_BELOW_HORIZONTAL_PX = 32

RIGHT_ISOLATED_HORIZONTAL_COMPONENT_GUARD_ENABLE = True
RIGHT_ISOLATED_HORIZONTAL_MIN_RIGHT_GAIN = 75
RIGHT_ISOLATED_HORIZONTAL_MIN_GAP_FROM_MAIN = 30
RIGHT_ISOLATED_HORIZONTAL_MAX_AREA = 220
RIGHT_ISOLATED_HORIZONTAL_MAX_WIDTH = 35
RIGHT_ISOLATED_HORIZONTAL_MAX_HEIGHT = 18
RIGHT_ISOLATED_HORIZONTAL_MIN_MAIN_AREA = 350
RIGHT_ISOLATED_HORIZONTAL_MIN_COMPONENT_Y = 200


def filter_right_isolated_horizontal_component(horizontal_rescue_mask, principal_mask):
    if not RIGHT_ISOLATED_HORIZONTAL_COMPONENT_GUARD_ENABLE:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    if horizontal_rescue_mask is None or principal_mask is None:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    if mask_area(horizontal_rescue_mask) == 0 or mask_area(principal_mask) == 0:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    principal_bounds = get_mask_bounds(principal_mask)

    if principal_bounds is None:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (horizontal_rescue_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 2:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    components = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        x2 = x + width - 1
        y2 = y + height - 1

        components.append(
            {
                "label": label,
                "area": area,
                "x": x,
                "x2": x2,
                "y": y,
                "y2": y2,
                "width": width,
                "height": height,
            }
        )

    main_component = max(components, key=lambda item: item["area"])

    if main_component["area"] < RIGHT_ISOLATED_HORIZONTAL_MIN_MAIN_AREA:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    kept = np.zeros_like(horizontal_rescue_mask, dtype=np.uint8)
    removed = np.zeros_like(horizontal_rescue_mask, dtype=np.uint8)

    for component in components:
        label = component["label"]
        component_pixels = labels == label

        if label == main_component["label"]:
            kept[component_pixels] = 255
            continue

        right_gain = component["x2"] - principal_bounds["max_x"]
        gap_from_main = component["x"] - main_component["x2"]

        is_small_right_isolated = (
            component["area"] <= RIGHT_ISOLATED_HORIZONTAL_MAX_AREA
            and component["width"] <= RIGHT_ISOLATED_HORIZONTAL_MAX_WIDTH
            and component["height"] <= RIGHT_ISOLATED_HORIZONTAL_MAX_HEIGHT
            and right_gain >= RIGHT_ISOLATED_HORIZONTAL_MIN_RIGHT_GAIN
            and gap_from_main >= RIGHT_ISOLATED_HORIZONTAL_MIN_GAP_FROM_MAIN
            and component["y"] >= RIGHT_ISOLATED_HORIZONTAL_MIN_COMPONENT_Y
        )

        if is_small_right_isolated:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    return kept, removed


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
        y + height - 1
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


def filter_secondary_tail_after_horizontal(
    secondary_mask,
    horizontal_rescue_mask,
):
    if not SECONDARY_AFTER_HORIZONTAL_TAIL_GUARD_ENABLE:
        return secondary_mask, empty_mask_like(secondary_mask)

    if secondary_mask is None or horizontal_rescue_mask is None:
        return secondary_mask, empty_mask_like(secondary_mask)

    if mask_area(secondary_mask) == 0:
        return secondary_mask, empty_mask_like(secondary_mask)

    horizontal_bounds = get_mask_bounds(horizontal_rescue_mask)
    horizontal_median_y = mask_median_y(horizontal_rescue_mask)

    if horizontal_bounds is None or horizontal_median_y is None:
        return secondary_mask, empty_mask_like(secondary_mask)

    if horizontal_bounds["area"] < SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_HORIZONTAL_AREA:
        return secondary_mask, empty_mask_like(secondary_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (secondary_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    kept = np.zeros_like(secondary_mask, dtype=np.uint8)
    removed = np.zeros_like(secondary_mask, dtype=np.uint8)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        x + width - 1
        y2 = y + height - 1

        component_pixels = labels == label
        component_median_y = float(centroids[label][1])

        is_small_tail = (
            area <= SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_AREA
            and width <= SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_WIDTH
            and height <= SECONDARY_AFTER_HORIZONTAL_TAIL_MAX_HEIGHT
        )

        starts_after_horizontal_end = (
            x
            >= horizontal_bounds["max_x"]
            + SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_RIGHT_GAP
        )

        is_far_below_horizontal = (
            component_median_y
            >= horizontal_median_y
            + SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_BELOW_HORIZONTAL_PX
            or y2
            >= horizontal_median_y
            + SECONDARY_AFTER_HORIZONTAL_TAIL_MIN_BELOW_HORIZONTAL_PX
            + 8
        )

        vertical_overlap_top = max(y, horizontal_bounds["min_y"])
        vertical_overlap_bottom = min(y2, horizontal_bounds["max_y"])
        vertical_overlap = max(0, vertical_overlap_bottom - vertical_overlap_top + 1)
        vertical_overlap_frac = vertical_overlap / max(height, 1)

        is_thin_edge_fragment = area <= 90 and width <= 28 and height <= 10

        is_only_touching_horizontal_edge = (
            vertical_overlap_frac <= 0.30 or y2 <= horizontal_bounds["min_y"] + 1
        )

        is_after_or_at_horizontal_end = x >= horizontal_bounds["max_x"] - 1

        remove_low_tail = (
            is_small_tail and starts_after_horizontal_end and is_far_below_horizontal
        )

        remove_thin_edge_tail = (
            is_thin_edge_fragment
            and is_after_or_at_horizontal_end
            and is_only_touching_horizontal_edge
        )

        if remove_low_tail or remove_thin_edge_tail:
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


HORIZONTAL_FLOATING_UPPER_STRIP_GUARD_ENABLE = True
HORIZONTAL_FLOATING_UPPER_STRIP_MAX_AREA = 650
HORIZONTAL_FLOATING_UPPER_STRIP_MIN_WIDTH = 40
HORIZONTAL_FLOATING_UPPER_STRIP_MAX_WIDTH = 130
HORIZONTAL_FLOATING_UPPER_STRIP_MAX_HEIGHT = 22
HORIZONTAL_FLOATING_UPPER_STRIP_MIN_RIGHT_GAIN = 35
HORIZONTAL_FLOATING_UPPER_STRIP_MAX_RIGHT_GAIN = 90
HORIZONTAL_FLOATING_UPPER_STRIP_MAX_START_GAP_FROM_PRINCIPAL = 35
HORIZONTAL_FLOATING_UPPER_STRIP_CONTEXT_LEFT = 130
HORIZONTAL_FLOATING_UPPER_STRIP_CONTEXT_RIGHT = 15
HORIZONTAL_FLOATING_UPPER_STRIP_MIN_CONTEXT_PIXELS = 35
HORIZONTAL_FLOATING_UPPER_STRIP_MIN_ABOVE_LOCAL_PX = 6
HORIZONTAL_FLOATING_UPPER_STRIP_MAX_DIRECT_CONTACT_PIXELS = 6


def filter_floating_upper_horizontal_strip(horizontal_rescue_mask, principal_mask):
    if not HORIZONTAL_FLOATING_UPPER_STRIP_GUARD_ENABLE:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    if horizontal_rescue_mask is None or principal_mask is None:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    if mask_area(horizontal_rescue_mask) == 0 or mask_area(principal_mask) == 0:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    principal_bounds = get_mask_bounds(principal_mask)

    if principal_bounds is None:
        return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (horizontal_rescue_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    kept = np.zeros_like(horizontal_rescue_mask, dtype=np.uint8)
    removed = np.zeros_like(horizontal_rescue_mask, dtype=np.uint8)

    image_h, image_w = horizontal_rescue_mask.shape[:2]

    for label in range(1, num_labels):
        component_pixels = labels == label

        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        x2 = x + width - 1
        y + height - 1

        component_median_y = float(centroids[label][1])

        right_gain = x2 - principal_bounds["max_x"]

        size_matches = (
            area <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_AREA
            and width >= HORIZONTAL_FLOATING_UPPER_STRIP_MIN_WIDTH
            and width <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_WIDTH
            and height <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_HEIGHT
        )

        starts_near_principal_edge = (
            x
            <= principal_bounds["max_x"]
            + HORIZONTAL_FLOATING_UPPER_STRIP_MAX_START_GAP_FROM_PRINCIPAL
        )

        extends_right = (
            right_gain >= HORIZONTAL_FLOATING_UPPER_STRIP_MIN_RIGHT_GAIN
            and right_gain <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_RIGHT_GAIN
            and starts_near_principal_edge
        )

        x1_context = max(0, x - HORIZONTAL_FLOATING_UPPER_STRIP_CONTEXT_LEFT)
        x2_context = min(image_w, x + HORIZONTAL_FLOATING_UPPER_STRIP_CONTEXT_RIGHT + 1)

        context = principal_mask[:, x1_context:x2_context]
        context_ys, context_xs = np.where(context > 0)

        has_context = (
            len(context_ys) >= HORIZONTAL_FLOATING_UPPER_STRIP_MIN_CONTEXT_PIXELS
        )

        if has_context:
            local_reference_y = float(np.median(context_ys))
        else:
            local_reference_y = None

        is_above_local_principal_band = (
            local_reference_y is not None
            and component_median_y
            <= local_reference_y - HORIZONTAL_FLOATING_UPPER_STRIP_MIN_ABOVE_LOCAL_PX
        )

        dilated_principal = cv2.dilate(
            (principal_mask > 0).astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        direct_contact_pixels = int(
            np.count_nonzero(
                (component_pixels.astype(np.uint8) > 0) & (dilated_principal > 0)
            )
        )

        lacks_direct_contact = (
            direct_contact_pixels
            <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_DIRECT_CONTACT_PIXELS
        )

        if (
            size_matches
            and extends_right
            and has_context
            and is_above_local_principal_band
            and lacks_direct_contact
        ):
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    if mask_area(removed) == mask_area(horizontal_rescue_mask):
        if mask_area(horizontal_rescue_mask) > HORIZONTAL_FLOATING_UPPER_STRIP_MAX_AREA:
            return horizontal_rescue_mask, empty_mask_like(horizontal_rescue_mask)

    return kept, removed


SECONDARY_FLOATING_STRIP_AFTER_HORIZONTAL_REJECT_ENABLE = True

SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_AREA = 260
SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_WIDTH = 60
SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_HEIGHT = 35
SECONDARY_LOW_TAIL_AFTER_STRIP_MIN_RIGHT_GAIN = 55


def filter_secondary_floating_strip_after_horizontal_reject(
    secondary_mask,
    principal_mask,
):
    if not SECONDARY_FLOATING_STRIP_AFTER_HORIZONTAL_REJECT_ENABLE:
        return secondary_mask, empty_mask_like(secondary_mask)

    if secondary_mask is None or principal_mask is None:
        return secondary_mask, empty_mask_like(secondary_mask)

    if mask_area(secondary_mask) == 0 or mask_area(principal_mask) == 0:
        return secondary_mask, empty_mask_like(secondary_mask)

    principal_bounds = get_mask_bounds(principal_mask)
    principal_median_y = mask_median_y(principal_mask)

    if principal_bounds is None or principal_median_y is None:
        return secondary_mask, empty_mask_like(secondary_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (secondary_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    kept = np.zeros_like(secondary_mask, dtype=np.uint8)
    removed = np.zeros_like(secondary_mask, dtype=np.uint8)

    dilated_principal = cv2.dilate(
        (principal_mask > 0).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    for label in range(1, num_labels):
        component_pixels = labels == label

        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        x2 = x + width - 1
        y2 = y + height - 1

        component_median_y = float(centroids[label][1])
        right_gain = x2 - principal_bounds["max_x"]
        starts_near_principal_edge = (
            x
            <= principal_bounds["max_x"]
            + HORIZONTAL_FLOATING_UPPER_STRIP_MAX_START_GAP_FROM_PRINCIPAL
        )
        moderate_right_gain = (
            right_gain >= HORIZONTAL_FLOATING_UPPER_STRIP_MIN_RIGHT_GAIN
            and right_gain <= HORIZONTAL_FLOATING_UPPER_STRIP_MAX_RIGHT_GAIN
            and starts_near_principal_edge
        )

        contact_pixels = int(
            np.count_nonzero(
                (component_pixels.astype(np.uint8) > 0) & (dilated_principal > 0)
            )
        )

        is_long_thin_right_strip = (
            area >= 250
            and area <= 850
            and width >= 45
            and width <= 150
            and height <= 20
            and moderate_right_gain
            and contact_pixels <= 12
        )

        is_low_tail_after_right_extension = (
            area <= SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_AREA
            and width <= SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_WIDTH
            and height <= SECONDARY_LOW_TAIL_AFTER_STRIP_MAX_HEIGHT
            and moderate_right_gain
            and right_gain >= SECONDARY_LOW_TAIL_AFTER_STRIP_MIN_RIGHT_GAIN
            and (
                component_median_y >= principal_median_y + 40
                or y2 >= principal_median_y + 50
            )
        )

        if is_long_thin_right_strip or is_low_tail_after_right_extension:
            removed[component_pixels] = 255
        else:
            kept[component_pixels] = 255

    return kept, removed


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
            final_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
