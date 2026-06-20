import cv2
import numpy as np

PRINCIPAL_SELECTOR_ENABLE = True

SUSPECT_MAX_MEDIAN_Y_FRAC = 0.22
SUSPECT_MIN_HEIGHT_PX = 65
SUSPECT_MIN_AREA = 2500
SUSPECT_MIN_WIDTH_PX = 260

LOWER_START_EXTRA_Y = 18
LOWER_START_MIN_FRAC = 0.24
LOWER_END_FRAC = 0.58
MIN_MEDIAN_DELTA_Y = 70
MAX_ALTERNATIVE_MEDIAN_Y_FRAC = 0.46
MIN_ALTERNATIVE_MEDIAN_Y_FRAC = 0.24

MAX_CANDIDATES = 6
MIN_CANDIDATE_WIDTH = 220
MIN_CANDIDATE_AREA = 700
MAX_CANDIDATE_HEIGHT_FRAC = 0.28
MIN_X_TOUCH_OR_OVERLAP_PX = 45
MAX_START_GAP_FROM_CURRENT_PX = 120

CLOSE_KERNEL_W = 55
CLOSE_KERNEL_H = 5
REGION_DILATE_W = 17
REGION_DILATE_H = 9

TOP_BAND_ABOVE = 2
TOP_BAND_BELOW = 8

COMPLEX_BAND_ABOVE = 3
COMPLEX_BAND_BELOW = 165
COMPLEX_X_EXTENSION = 52
COMPLEX_CLOSE_W = 21
COMPLEX_CLOSE_H = 7
COMPLEX_MIN_FRAGMENT_AREA = 8
COMPLEX_MAX_FRAGMENT_HEIGHT = 190
COMPLEX_MAX_FRAGMENT_MEDIAN_BELOW = 160
COMPLEX_MAX_FRAGMENT_GAP_X = 115
COMPLEX_MIN_SELECTED_AREA_FACTOR = 0.04

CLEAN_CLOSE_W = 9
CLEAN_CLOSE_H = 3
MIN_SMALL_COMPONENT_AREA = 10


def normalize_mask(mask):
    result = np.zeros_like(mask, dtype=np.uint8)
    result[mask > 0] = 255
    return result


def empty_mask_like(mask):
    return np.zeros_like(mask, dtype=np.uint8)


def mask_area(mask):
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


def mask_median_y(mask):
    if mask is None:
        return None

    ys, _ = np.where(mask > 0)

    if len(ys) == 0:
        return None

    return float(np.median(ys))


def mask_density(mask):
    bounds = get_mask_bounds(mask)
    if bounds is None:
        return 0.0

    box_area = max(bounds["width"] * bounds["height"], 1)
    return float(bounds["area"] / box_area)


def merge_masks(mask_a, mask_b):
    result = np.zeros_like(mask_a, dtype=np.uint8)

    if mask_a is not None:
        result[mask_a > 0] = 255

    if mask_b is not None:
        result[mask_b > 0] = 255

    return result


def clean_mask(mask):
    mask = normalize_mask(mask)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (CLEAN_CLOSE_W, CLEAN_CLOSE_H),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    result = np.zeros_like(mask, dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= MIN_SMALL_COMPONENT_AREA:
            result[labels == label] = 255

    return result


def remove_tiny_components(mask, min_area):
    mask = normalize_mask(mask)
    result = np.zeros_like(mask, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            result[labels == label] = 255

    return result


def is_current_principal_suspect(principal_mask):
    bounds = get_mask_bounds(principal_mask)
    median_y = mask_median_y(principal_mask)

    if bounds is None or median_y is None:
        return False

    height = principal_mask.shape[0]

    return (
            median_y <= SUSPECT_MAX_MEDIAN_Y_FRAC * height
            and bounds["height"] >= SUSPECT_MIN_HEIGHT_PX
            and bounds["area"] >= SUSPECT_MIN_AREA
            and bounds["width"] >= SUSPECT_MIN_WIDTH_PX
    )


def build_lower_search_roi(binary_top2, principal_mask):
    height, width = binary_top2.shape[:2]
    principal_bounds = get_mask_bounds(principal_mask)

    roi = np.zeros((height, width), dtype=np.uint8)

    if principal_bounds is None:
        return roi

    start_y = max(
        int(principal_bounds["max_y"] + LOWER_START_EXTRA_Y),
        int(round(LOWER_START_MIN_FRAC * height)),
    )
    end_y = int(round(LOWER_END_FRAC * height))

    start_y = max(0, min(height - 1, start_y))
    end_y = max(start_y + 1, min(height, end_y))

    roi[start_y:end_y, :] = 255
    return roi


def component_top_edge_points(mask):
    ys_all = []
    xs_all = []
    height, width = mask.shape[:2]

    for x in range(width):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) == 0:
            continue

        xs_all.append(x)
        ys_all.append(int(np.min(ys)))

    if len(xs_all) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    return np.array(xs_all, dtype=np.float32), np.array(ys_all, dtype=np.float32)


def estimate_top_edge_slope(mask):
    xs, ys = component_top_edge_points(mask)

    if len(xs) < 8:
        return 0.0, 999.0

    coeff = np.polyfit(xs, ys, deg=1)
    pred = coeff[0] * xs + coeff[1]
    residual = float(np.median(np.abs(ys - pred)))
    return float(coeff[0]), residual


def estimate_top_edge_line(mask):
    xs, ys = component_top_edge_points(mask)

    if len(xs) < 8:
        return None

    coeff = np.polyfit(xs, ys, deg=1)
    slope = float(coeff[0])
    intercept = float(coeff[1])
    pred = slope * xs + intercept
    residual = float(np.median(np.abs(ys - pred)))

    return {
        "slope": slope,
        "intercept": intercept,
        "residual": residual,
        "xs": xs,
        "ys": ys,
    }


def band_by_top_edge(mask):
    mask = normalize_mask(mask)
    height, width = mask.shape[:2]
    result = np.zeros_like(mask, dtype=np.uint8)

    for x in range(width):
        ys = np.where(mask[:, x] > 0)[0]

        if len(ys) == 0:
            continue

        top_y = int(np.min(ys))
        y1 = max(0, top_y - TOP_BAND_ABOVE)
        y2 = min(height, top_y + TOP_BAND_BELOW + 1)

        column = mask[y1:y2, x]
        result[y1:y2, x][column > 0] = 255

    return clean_mask(result)


def build_oblique_band_roi_from_line(shape, line, bounds):
    height, width = shape[:2]
    roi = np.zeros((height, width), dtype=np.uint8)

    if line is None or bounds is None:
        return roi

    x1 = max(0, bounds["min_x"] - COMPLEX_X_EXTENSION)
    x2 = min(width - 1, bounds["max_x"] + COMPLEX_X_EXTENSION)

    for x in range(x1, x2 + 1):
        y_line = int(round(line["slope"] * x + line["intercept"]))
        y1 = max(0, y_line - COMPLEX_BAND_ABOVE)
        y2 = min(height - 1, y_line + COMPLEX_BAND_BELOW)
        if y2 >= y1:
            roi[y1:y2 + 1, x] = 255

    return roi


def median_distance_to_line(mask, line):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or line is None:
        return 999.0

    pred = line["slope"] * xs + line["intercept"]
    return float(np.median(ys - pred))


def horizontal_gap_between(bounds_a, bounds_b):
    if bounds_a is None or bounds_b is None:
        return 999999

    if bounds_a["max_x"] < bounds_b["min_x"]:
        return bounds_b["min_x"] - bounds_a["max_x"]

    if bounds_b["max_x"] < bounds_a["min_x"]:
        return bounds_a["min_x"] - bounds_b["max_x"]

    return 0


def build_controlled_pleura_complex(anchor_mask, search_mask):
    anchor_mask = normalize_mask(anchor_mask)
    search_mask = normalize_mask(search_mask)
    empty = empty_mask_like(anchor_mask)

    anchor_bounds = get_mask_bounds(anchor_mask)
    if anchor_bounds is None:
        return {
            "selected_mask": empty,
            "accepted_mask": empty,
            "accepted_components": [],
            "complex_roi_mask": empty,
        }

    line = estimate_top_edge_line(anchor_mask)

    if line is None:
        selected = band_by_top_edge(anchor_mask)
        return {
            "selected_mask": selected,
            "accepted_mask": selected,
            "accepted_components": [],
            "complex_roi_mask": selected.copy(),
        }

    complex_roi = build_oblique_band_roi_from_line(anchor_mask.shape, line, anchor_bounds)
    complex_search = cv2.bitwise_and(search_mask, complex_roi)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (COMPLEX_CLOSE_W, COMPLEX_CLOSE_H),
    )
    linked_complex = cv2.morphologyEx(complex_search, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    linked_complex = remove_tiny_components(linked_complex, COMPLEX_MIN_FRAGMENT_AREA)

    selected = np.zeros_like(anchor_mask, dtype=np.uint8)
    accepted_components = []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(linked_complex, 8)

    for label in range(1, num_labels):
        linked_component = np.zeros_like(anchor_mask, dtype=np.uint8)
        linked_component[labels == label] = 255

        component = cv2.bitwise_and(complex_search, cv2.dilate(linked_component, None, iterations=1))
        component = remove_tiny_components(component, COMPLEX_MIN_FRAGMENT_AREA)

        bounds = get_mask_bounds(component)
        if bounds is None:
            continue

        if bounds["area"] < COMPLEX_MIN_FRAGMENT_AREA:
            continue

        if bounds["height"] > COMPLEX_MAX_FRAGMENT_HEIGHT:
            continue

        median_below = median_distance_to_line(component, line)

        if median_below < -COMPLEX_BAND_ABOVE - 5:
            continue

        if median_below > COMPLEX_MAX_FRAGMENT_MEDIAN_BELOW:
            continue

        gap_x = horizontal_gap_between(bounds, anchor_bounds)
        if gap_x > COMPLEX_MAX_FRAGMENT_GAP_X:
            continue

        overlap_x = min(bounds["max_x"], anchor_bounds["max_x"]) - max(bounds["min_x"], anchor_bounds["min_x"]) + 1
        overlap_x = max(0, overlap_x)
        if overlap_x < 8 and gap_x > 42:
            continue

        selected[component > 0] = 255
        accepted_components.append(
            {
                "mask": component,
                "bounds": bounds,
                "median_y": mask_median_y(component) or 0.0,
                "density": mask_density(component),
                "median_below_line": median_below,
                "gap_x": gap_x,
                "overlap_x": overlap_x,
            }
        )

    if mask_area(selected) < 0.28 * max(mask_area(anchor_mask), 1):
        fallback = np.zeros_like(anchor_mask, dtype=np.uint8)
        fallback_pixels = remove_tiny_components(complex_search, COMPLEX_MIN_FRAGMENT_AREA)

        num_fb, fb_labels, fb_stats, _ = cv2.connectedComponentsWithStats(fallback_pixels, 8)

        for fb_label in range(1, num_fb):
            fb_component = np.zeros_like(anchor_mask, dtype=np.uint8)
            fb_component[fb_labels == fb_label] = 255

            fb_bounds = get_mask_bounds(fb_component)
            if fb_bounds is None:
                continue

            if fb_bounds["area"] < COMPLEX_MIN_FRAGMENT_AREA:
                continue

            fb_median_below = median_distance_to_line(fb_component, line)

            if fb_median_below < -COMPLEX_BAND_ABOVE - 5:
                continue

            if fb_median_below > COMPLEX_MAX_FRAGMENT_MEDIAN_BELOW:
                continue

            fb_gap_x = horizontal_gap_between(fb_bounds, anchor_bounds)
            if fb_gap_x > COMPLEX_MAX_FRAGMENT_GAP_X:
                continue

            fb_overlap_x = min(fb_bounds["max_x"], anchor_bounds["max_x"]) - max(fb_bounds["min_x"],
                                                                                 anchor_bounds["min_x"]) + 1
            fb_overlap_x = max(0, fb_overlap_x)

            if fb_overlap_x < 6 and fb_gap_x > 42:
                continue

            fallback[fb_component > 0] = 255

        selected = merge_masks(selected, fallback)

    selected = merge_masks(selected, band_by_top_edge(anchor_mask))

    selected = remove_tiny_components(selected, MIN_SMALL_COMPONENT_AREA)

    return {
        "selected_mask": selected,
        "accepted_mask": selected.copy(),
        "accepted_components": accepted_components,
        "complex_roi_mask": complex_roi,
    }


def candidate_touch_score(bounds, current_bounds):
    if bounds is None or current_bounds is None:
        return 0.0

    horizontal_gap = bounds["min_x"] - current_bounds["max_x"]

    if horizontal_gap <= 0:
        overlap = min(bounds["max_x"], current_bounds["max_x"]) - max(bounds["min_x"], current_bounds["min_x"]) + 1
        return float(max(overlap, 0))

    return float(-horizontal_gap)


def score_candidate(component_original, linked_component, principal_mask):
    bounds = get_mask_bounds(component_original)
    linked_bounds = get_mask_bounds(linked_component)
    principal_bounds = get_mask_bounds(principal_mask)

    if bounds is None or linked_bounds is None or principal_bounds is None:
        return None

    height, width = principal_mask.shape[:2]
    principal_median_y = mask_median_y(principal_mask)
    candidate_median_y = mask_median_y(component_original)

    if principal_median_y is None or candidate_median_y is None:
        return None

    if bounds["width"] < MIN_CANDIDATE_WIDTH:
        return None

    if bounds["area"] < MIN_CANDIDATE_AREA:
        return None

    if linked_bounds["height"] > MAX_CANDIDATE_HEIGHT_FRAC * height:
        return None

    if candidate_median_y < principal_median_y + MIN_MEDIAN_DELTA_Y:
        return None

    if candidate_median_y < MIN_ALTERNATIVE_MEDIAN_Y_FRAC * height:
        return None

    if candidate_median_y > MAX_ALTERNATIVE_MEDIAN_Y_FRAC * height:
        return None

    touch = candidate_touch_score(bounds, principal_bounds)
    if touch < -MAX_START_GAP_FROM_CURRENT_PX:
        return None

    missing_touch_penalty = 0.0
    if touch < MIN_X_TOUCH_OR_OVERLAP_PX:
        missing_touch_penalty = abs(touch - MIN_X_TOUCH_OR_OVERLAP_PX)

    slope, residual = estimate_top_edge_slope(component_original)
    thinness = bounds["width"] / max(linked_bounds["height"], 1)
    density = mask_density(component_original)

    target_y = 0.35 * height
    y_penalty = abs(candidate_median_y - target_y)

    score = (
            3.5 * bounds["width"]
            + 1.15 * bounds["area"]
            + 30.0 * thinness
            + 160.0 * density
            + 0.8 * max(touch, 0.0)
            - 1.6 * linked_bounds["height"]
            - 28.0 * abs(slope)
            - 2.0 * residual
            - 0.45 * y_penalty
            - 3.0 * missing_touch_penalty
    )

    return float(score)


def build_principal_candidates(binary_top2, principal_mask):
    binary_top2 = normalize_mask(binary_top2)
    principal_mask = normalize_mask(principal_mask)

    roi_mask = build_lower_search_roi(binary_top2, principal_mask)
    search_mask = cv2.bitwise_and(binary_top2, roi_mask)

    empty = empty_mask_like(binary_top2)

    if mask_area(search_mask) == 0:
        return {
            "roi_mask": roi_mask,
            "search_mask": search_mask,
            "linked_mask": empty.copy(),
            "rejected_mask": empty.copy(),
            "candidate_mask": empty.copy(),
            "candidates": [],
        }

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (CLOSE_KERNEL_W, CLOSE_KERNEL_H),
    )
    linked_mask = cv2.morphologyEx(search_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(linked_mask, 8)

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (REGION_DILATE_W, REGION_DILATE_H),
    )

    candidates = []
    rejected_mask = empty.copy()
    candidate_mask = empty.copy()

    for label in range(1, num_labels):
        linked_component = np.zeros_like(binary_top2, dtype=np.uint8)
        linked_component[labels == label] = 255

        region_mask = cv2.dilate(linked_component, dilate_kernel, iterations=1)
        component_original = cv2.bitwise_and(search_mask, region_mask)
        component_original = clean_mask(component_original)

        score = score_candidate(component_original, linked_component, principal_mask)

        if score is None:
            rejected_mask = merge_masks(rejected_mask, component_original)
            continue

        top_mask = band_by_top_edge(component_original)
        complex_result = build_controlled_pleura_complex(component_original, search_mask)
        complex_mask = complex_result["selected_mask"]

        item = {
            "score": score,
            "full_mask": component_original,
            "top_mask": top_mask,
            "complex_mask": complex_mask,
            "complex_roi_mask": complex_result["complex_roi_mask"],
            "accepted_mask": complex_result["accepted_mask"],
            "accepted_components": complex_result["accepted_components"],
            "linked_mask": linked_component,
            "bounds": get_mask_bounds(component_original),
            "top_bounds": get_mask_bounds(top_mask),
            "complex_bounds": get_mask_bounds(complex_mask),
            "linked_bounds": get_mask_bounds(linked_component),
            "median_y": mask_median_y(component_original) or 0.0,
            "density": mask_density(component_original),
        }
        candidates.append(item)
        candidate_mask = merge_masks(candidate_mask, component_original)

    candidates.sort(key=lambda item: item["score"], reverse=True)

    if len(candidates) > MAX_CANDIDATES:
        for item in candidates[MAX_CANDIDATES:]:
            rejected_mask = merge_masks(rejected_mask, item["full_mask"])
        candidates = candidates[:MAX_CANDIDATES]

    return {
        "roi_mask": roi_mask,
        "search_mask": search_mask,
        "linked_mask": linked_mask,
        "rejected_mask": rejected_mask,
        "candidate_mask": candidate_mask,
        "candidates": candidates,
    }


def select_principal_by_lower_candidate(binary_top2, principal_mask):
    binary_top2 = normalize_mask(binary_top2)
    principal_mask = normalize_mask(principal_mask)

    empty = empty_mask_like(principal_mask)

    if not PRINCIPAL_SELECTOR_ENABLE:
        return {
            "principal_mask": principal_mask,
            "used_selector": False,
            "selected_mask": empty,
            "roi_mask": empty,
            "search_mask": empty,
            "candidate_mask": empty,
            "rejected_mask": empty,
            "linked_mask": empty,
            "reason": "disabled",
            "candidates": [],
            "accepted_mask": empty,
            "accepted_components": [],
            "anchor": None,
        }

    selector_result = build_principal_candidates(binary_top2, principal_mask)

    if not is_current_principal_suspect(principal_mask):
        return {
            "principal_mask": principal_mask,
            "used_selector": False,
            "selected_mask": empty,
            "roi_mask": selector_result["roi_mask"],
            "search_mask": selector_result["search_mask"],
            "candidate_mask": selector_result["candidate_mask"],
            "rejected_mask": selector_result["rejected_mask"],
            "linked_mask": selector_result["linked_mask"],
            "reason": "current_principal_not_suspect",
            "candidates": selector_result["candidates"],
            "accepted_mask": empty,
            "accepted_components": [],
            "anchor": None,
        }

    if len(selector_result["candidates"]) == 0:
        return {
            "principal_mask": principal_mask,
            "used_selector": False,
            "selected_mask": empty,
            "roi_mask": selector_result["roi_mask"],
            "search_mask": selector_result["search_mask"],
            "candidate_mask": selector_result["candidate_mask"],
            "rejected_mask": selector_result["rejected_mask"],
            "linked_mask": selector_result["linked_mask"],
            "reason": "no_lower_candidate",
            "candidates": [],
            "accepted_mask": empty,
            "accepted_components": [],
            "anchor": None,
        }

    best = selector_result["candidates"][0]
    selected_mask = normalize_mask(best.get("complex_mask", best["top_mask"]))

    if mask_area(selected_mask) < COMPLEX_MIN_SELECTED_AREA_FACTOR * mask_area(principal_mask):
        fallback_top = normalize_mask(best["top_mask"])
        if mask_area(fallback_top) >= COMPLEX_MIN_SELECTED_AREA_FACTOR * mask_area(principal_mask):
            selected_mask = fallback_top
        else:
            return {
                "principal_mask": principal_mask,
                "used_selector": False,
                "selected_mask": selected_mask,
                "roi_mask": selector_result["roi_mask"],
                "search_mask": selector_result["search_mask"],
                "candidate_mask": selector_result["candidate_mask"],
                "rejected_mask": selector_result["rejected_mask"],
                "linked_mask": selector_result["linked_mask"],
                "reason": "selected_too_small",
                "candidates": selector_result["candidates"],
                "accepted_mask": best.get("accepted_mask", empty),
                "accepted_components": best.get("accepted_components", []),
                "anchor": best,
            }

    return {
        "principal_mask": selected_mask,
        "used_selector": True,
        "selected_mask": selected_mask,
        "roi_mask": selector_result["roi_mask"],
        "search_mask": selector_result["search_mask"],
        "candidate_mask": selector_result["candidate_mask"],
        "rejected_mask": selector_result["rejected_mask"],
        "linked_mask": selector_result["linked_mask"],
        "reason": "selected_lower_pleura_complex_v3",
        "candidates": selector_result["candidates"],
        "accepted_mask": best.get("accepted_mask", selected_mask),
        "accepted_components": best.get("accepted_components", []),
        "anchor": best,
    }


def to_bgr(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def draw_mask_overlay(image_bgr, mask, color, alpha=0.60):
    output = to_bgr(image_bgr)
    mask = normalize_mask(mask)

    if mask_area(mask) == 0:
        return output

    overlay = output.copy()
    overlay[mask > 0] = color
    output = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0)
    return output


def draw_principal_selector_debug(crop, original_principal_mask, selector_result, traveler_points=None):
    output = to_bgr(crop)

    output = draw_mask_overlay(output, selector_result.get("roi_mask"), (180, 0, 180), alpha=0.15)
    output = draw_mask_overlay(output, selector_result.get("rejected_mask"), (0, 0, 255), alpha=0.35)
    output = draw_mask_overlay(output, selector_result.get("candidate_mask"), (0, 255, 255), alpha=0.25)
    output = draw_mask_overlay(output, original_principal_mask, (255, 255, 0), alpha=0.55)
    output = draw_mask_overlay(output, selector_result.get("accepted_mask"), (255, 128, 0), alpha=0.35)
    output = draw_mask_overlay(output, selector_result.get("selected_mask"), (0, 255, 0), alpha=0.75)

    if traveler_points is not None:
        for point in traveler_points:
            try:
                x = int(round(point.x))
                y = int(round(point.y))
            except AttributeError:
                x = int(round(point[0]))
                y = int(round(point[1]))

            if 0 <= y < output.shape[0] and 0 <= x < output.shape[1]:
                cv2.circle(output, (x, y), 1, (255, 180, 0), -1)

    reason = str(selector_result.get("reason", ""))
    used = bool(selector_result.get("used_selector", False))
    text = f"principal selector used={used} reason={reason}"

    cv2.putText(
        output,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        "cyan=initial | yellow=candidates | orange=complex accepted | green=selected | red=rejected | purple=ROI",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output
