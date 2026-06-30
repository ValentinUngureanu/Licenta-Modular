import re
import shutil

import cv2
import numpy as np

import config
from crop import crop_ultrasound
from final_contour import (
    build_final_contour,
    draw_final_contour_on_crop,
    draw_final_contour_on_original,
    draw_final_mask_on_crop,
    draw_final_mask_on_original,
    project_mask_to_original,
    remove_very_small_components,
)
from gap_rescue import (
    draw_gap_rescue_debug,
    filter_floating_right_gap_rescue,
    filter_upper_right_gap_rescue,
    gap_rescue_after_secondary,
)
from horizontal_rescue import (
    draw_horizontal_rescue_debug,
    filter_floating_upper_horizontal_strip,
    filter_layered_horizontal_tail,
    filter_right_isolated_horizontal_component,
    run_guarded_horizontal_rescue,
)
from image_io import (
    ensure_dir,
    find_image_path,
    get_indices_to_process,
    make_output_name,
    read_image_bgr,
    save_image,
)
from pleural_interruptions import detect_pleural_interruptions
from pleural_nodules import detect_pleural_nodules
from preprocessing import preprocess_crop
from principal_component import (
    build_principal_component,
    draw_principal_component,
)
from principal_selector import (
    draw_principal_selector_debug,
    select_principal_by_lower_candidate,
)
from secondary_component import (
    build_secondary_components,
    draw_merged_components,
    draw_secondary_candidates,
    draw_secondary_components,
    draw_secondary_roi,
    filter_secondary_floating_strip_after_horizontal_reject,
    filter_secondary_tail_after_horizontal,
)
from secondary_rescue import (
    draw_merged_after_rescue as draw_secondary_rescue_merged_debug,
)
from secondary_rescue import (
    draw_rescue_debug as draw_secondary_rescue_debug,
)
from secondary_rescue import (
    filter_right_isolated_secondary_rescue,
    run_guarded_secondary_rescue,
)
from top2_final_contour import build_top2_final_contour
from top2_pleura import build_top2_guided_pleura
from traveler import build_traveler, draw_extended_component

try:
    from unification import build_top2_unification_debug as build_unification
except ImportError:
    from unification import build_top2_unification as build_unification

FINAL_TEST_DIR = config.RESULTS_DIR / "FINAL_CONTOUR_TEST"

CROP_DIR = FINAL_TEST_DIR / "00_CROP"
PALETTE_7_DIR = FINAL_TEST_DIR / "00_PALETTE_7"
BINARY_TOP1_DIR = FINAL_TEST_DIR / "01_BINARY_TOP1"
BINARY_TOP2_DIR = FINAL_TEST_DIR / "02_BINARY_TOP2_SECONDARY_SEARCH"
BINARY_TOP3_DIR = FINAL_TEST_DIR / "02A_BINARY_TOP3_NODULE_SUPPORT"
BINARY_TOP4_DIR = FINAL_TEST_DIR / "02B_BINARY_TOP4_NODULE_SUPPORT"
BINARY_TOP3_CONTOUR_DIR = FINAL_TEST_DIR / "02C_BINARY_TOP3_CONTOUR_ON_CROP"
BINARY_TOP4_CONTOUR_DIR = FINAL_TEST_DIR / "02D_BINARY_TOP4_CONTOUR_ON_CROP"
PRINCIPAL_DIR = FINAL_TEST_DIR / "03_PRINCIPAL_COMPONENT"
SECONDARY_DIR = FINAL_TEST_DIR / "04_PRINCIPAL_HORIZONTAL_SECONDARY_COMPONENTS"
MERGED_FINAL_DIR = FINAL_TEST_DIR / "05_MERGED_FINAL_AFTER_ALL_RESCUES"
SMALL_COMPONENT_CLEAN_DIR = FINAL_TEST_DIR / "06_SMALL_COMPONENT_CLEAN"
FINAL_MASK_CROP_DIR = FINAL_TEST_DIR / "07_FINAL_MASK_ON_CROP"
FINAL_CONTOUR_CROP_DIR = FINAL_TEST_DIR / "08_FINAL_CONTOUR_ON_CROP"
FINAL_CONTOUR_ORIGINAL_DIR = FINAL_TEST_DIR / "09_FINAL_CONTOUR_ON_ORIGINAL"
TOP2_CONTOUR_ONLY_DIR = FINAL_TEST_DIR / "10_TOP2_FINAL_CONTOUR_ONLY_ON_CROP"
TOP2_UNIFIED_CONTOUR_ONLY_DIR = (
    FINAL_TEST_DIR / "11_TOP2_SUPERFRAGMENT_CHECK1_CONTOUR_ON_CROP"
)
PLEURAL_INTERRUPTION_DIR = FINAL_TEST_DIR / "12_PLEURAL_INTERRUPTION_MARKING"
PLEURAL_INTERRUPTION_COMPARISON_DIR = (
    PLEURAL_INTERRUPTION_DIR / "00_COMPARATIE_CU_COMPONENTE"
)
PLEURAL_NODULE_DIR = FINAL_TEST_DIR / "13_PLEURAL_NODULE_MARKING"
PLEURAL_NODULE_COMPARISON_DIR = PLEURAL_NODULE_DIR / "00_COMPARATIE_CU_COMPONENTE"
PLEURAL_NODULE_STEPS_DIR = PLEURAL_NODULE_DIR / "00_PASI_FILTRARE_TOATE_POZELE"
PLEURAL_NODULE_ORIGINAL_BOX_DIR = PLEURAL_NODULE_DIR / "01_NODULI_CHENAR_PE_ORIGINAL"
FINAL_FINDINGS_ORIGINAL_DIR = (
    FINAL_TEST_DIR / "14_VARIANTA_FINALA_PLEURA_INTRERUPERI_NODULI"
)
PLEURAL_NODULE_STAGE0_DIR = PLEURAL_NODULE_DIR / "01_STAGE_0_TOT_TOP3_SUB_PLEURA"
PLEURAL_NODULE_STAGE1_DIR = PLEURAL_NODULE_DIR / "02_STAGE_1_CONTACT_CU_PLEURA"
PLEURAL_NODULE_STAGE2_DIR = PLEURAL_NODULE_DIR / "03_STAGE_2_COBORARE_SUB_PLEURA"
PLEURAL_NODULE_STAGE2_DIFF_DIR = (
    PLEURAL_NODULE_DIR / "04_STAGE_2_DIFERENTA_RESPINSE_DE_DROP"
)
PLEURAL_NODULE_STAGE3_DIR = PLEURAL_NODULE_DIR / "05_STAGE_3_DIMENSIUNI_MINIME"
PLEURAL_NODULE_STAGE3_DIFF_DIR = (
    PLEURAL_NODULE_DIR / "06_STAGE_3_DIFERENTA_RESPINSE_DE_DIMENSIUNI"
)

REST_CONTACT_SHEET_PATH = config.RESULTS_DIR / "00_TOATE_POZELE_FINAL_CONTOUR.jpg"

from postprocessing import (
    draw_artifact_source_overlay,
    filter_left_low_far_artifact_components,
    filter_principal_tail_under_horizontal,
    mask_area,
    merge_masks,
)


def to_bgr_debug(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image.copy()


def build_labeled_panel(image, title, panel_width=360, panel_height=230):
    image = to_bgr_debug(image)

    title_height = 30
    canvas = np.zeros((panel_height + title_height, panel_width, 3), dtype=np.uint8)

    height, width = image.shape[:2]
    scale = min(panel_width / max(width, 1), panel_height / max(height, 1))

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    x0 = (panel_width - new_width) // 2
    y0 = title_height + (panel_height - new_height) // 2
    canvas[y0 : y0 + new_height, x0 : x0 + new_width] = resized

    cv2.putText(
        canvas,
        title,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas


def normalize_debug_mask(mask):
    if mask is None:
        return None

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return (mask > 0).astype(np.uint8) * 255


def draw_binary_contour_on_crop(crop_bgr, binary_mask, color=(0, 255, 0), thickness=1):
    result = to_bgr_debug(crop_bgr)
    binary = normalize_debug_mask(binary_mask)

    if binary is None or np.count_nonzero(binary > 0) == 0:
        return result

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return result

    cv2.drawContours(
        result,
        contours,
        contourIdx=-1,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    return result


def draw_mask_difference_on_crop(
    crop_bgr,
    kept_mask,
    removed_mask,
    kept_color=(0, 255, 255),
    removed_color=(0, 120, 255),
    title_text="cyan=ramane dupa stage2 | portocaliu=respins de drop",
):
    result = to_bgr_debug(crop_bgr)

    kept = normalize_debug_mask(kept_mask)
    removed = normalize_debug_mask(removed_mask)

    if kept is not None:
        result = draw_binary_contour_on_crop(result, kept, kept_color, thickness=2)

    if removed is not None:
        result = draw_binary_contour_on_crop(
            result, removed, removed_color, thickness=2
        )

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


def build_interruption_component_comparison(
    principal_image,
    secondary_image,
    source_origin_image,
    interruption_image,
):
    panels = [
        build_labeled_panel(principal_image, "1 Principal"),
        build_labeled_panel(secondary_image, "2 Principal + secundare"),
        build_labeled_panel(source_origin_image, "3 Surse componente"),
        build_labeled_panel(interruption_image, "4 Intreruperi marcate"),
    ]

    row_1 = np.hstack([panels[0], panels[1]])
    row_2 = np.hstack([panels[2], panels[3]])

    return np.vstack([row_1, row_2])


def build_nodule_component_comparison(
    principal_image,
    secondary_image,
    source_origin_image,
    nodule_image,
):
    panels = [
        build_labeled_panel(principal_image, "1 Principal"),
        build_labeled_panel(secondary_image, "2 Principal + secundare"),
        build_labeled_panel(source_origin_image, "3 Surse componente"),
        build_labeled_panel(nodule_image, "4 Noduli marcati"),
    ]

    row_1 = np.hstack([panels[0], panels[1]])
    row_2 = np.hstack([panels[2], panels[3]])

    return np.vstack([row_1, row_2])


def draw_nodule_boxes_on_original(
    original_image,
    nodule_mask_crop,
    crop_box,
    title_text="chenar galben = nodul detectat",
):
    result = to_bgr_debug(original_image)
    nodule_mask_original = project_mask_to_original(
        nodule_mask_crop,
        original_image.shape,
        crop_box,
    )
    nodule_mask_original = normalize_debug_mask(nodule_mask_original)

    if nodule_mask_original is None or np.count_nonzero(nodule_mask_original > 0) == 0:
        cv2.putText(
            result,
            "Nu exista noduli detectati",
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return result

    contours, _ = cv2.findContours(
        nodule_mask_original,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    box_index = 1
    pad = 7
    height, width = result.shape[:2]

    for contour in contours:
        if cv2.contourArea(contour) < 1:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width - 1, x + w + pad)
        y2 = min(height - 1, y + h + pad)

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            result,
            f"N{box_index}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        box_index += 1

    cv2.putText(
        result,
        title_text,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        title_text,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result


def draw_contour_mask_on_image(result, mask, color, thickness=2):
    binary = normalize_debug_mask(mask)

    if binary is None or np.count_nonzero(binary > 0) == 0:
        return result

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return result

    cv2.drawContours(
        result,
        contours,
        contourIdx=-1,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    return result


def draw_bounding_boxes_on_image(result, mask, color, thickness=2, pad=4):
    binary = normalize_debug_mask(mask)

    if binary is None or np.count_nonzero(binary > 0) == 0:
        return result

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return result

    image_height, image_width = result.shape[:2]

    for contour in contours:
        if cv2.contourArea(contour) < 1:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        x1 = max(0, int(x) - int(pad))
        y1 = max(0, int(y) - int(pad))
        x2 = min(image_width - 1, int(x + w - 1) + int(pad))
        y2 = min(image_height - 1, int(y + h - 1) + int(pad))

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            cv2.LINE_AA,
        )

    return result


def draw_final_findings_on_original(
    original_image,
    crop_box,
    pleura_mask_crop,
    interruption_mask_crop,
    nodule_mask_crop,
):
    result = to_bgr_debug(original_image)

    pleura_original = project_mask_to_original(
        pleura_mask_crop,
        original_image.shape,
        crop_box,
    )
    interruption_original = project_mask_to_original(
        interruption_mask_crop,
        original_image.shape,
        crop_box,
    )
    nodule_original = project_mask_to_original(
        nodule_mask_crop,
        original_image.shape,
        crop_box,
    )

    # BGR: verde = pleura, galben = intreruperi, rosu = noduli
    result = draw_contour_mask_on_image(
        result,
        pleura_original,
        color=(0, 255, 0),
        thickness=1,
    )
    result = draw_contour_mask_on_image(
        result,
        interruption_original,
        color=(0, 255, 255),
        thickness=1,
    )
    # Nodulii se marcheaza cu dreptunghi rosu, conform cerintei proiectului.
    result = draw_bounding_boxes_on_image(
        result,
        nodule_original,
        color=(0, 0, 255),
        thickness=1,
        pad=4,
    )

    cv2.putText(
        result,
        "verde=pleura | galben=intreruperi | rosu=noduli",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        "verde=pleura | galben=intreruperi | rosu=noduli",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return result


def process_image(index: int, current: int, total: int) -> None:
    print(f"[{current}/{total}] Imagine {index}")

    image_path = find_image_path(index)

    if image_path is None:
        return

    image = read_image_bgr(image_path)
    crop, crop_box = crop_ultrasound(image)

    preprocessing_result = preprocess_crop(crop)
    palette_7 = preprocessing_result["palette_7"]
    binary_top1 = preprocessing_result["binary_top1"]
    binary_top2 = preprocessing_result["binary_top2"]
    binary_top3 = preprocessing_result["binary_top3"]
    binary_top4 = preprocessing_result["binary_top4"]

    traveler_result = build_traveler(binary_top1)
    raw_points = traveler_result["raw_points"]
    components = traveler_result["components"]
    extended_points = traveler_result["extended_points"]
    added_component_ids = traveler_result["added_component_ids"]

    principal_result = build_principal_component(
        binary_top1,
        extended_points,
    )

    principal_mask_initial = principal_result["principal_mask"]
    rejected_principal_mask = principal_result["rejected_mask"]

    principal_selector_result = select_principal_by_lower_candidate(
        binary_top2,
        principal_mask_initial,
    )
    principal_mask = principal_selector_result["principal_mask"]

    principal_mask, left_low_principal_removed_mask = (
        filter_left_low_far_artifact_components(
            principal_mask,
            principal_mask,
        )
    )

    if mask_area(left_low_principal_removed_mask) > 0:
        rejected_principal_mask = merge_masks(
            rejected_principal_mask,
            left_low_principal_removed_mask,
        )

    horizontal_result = run_guarded_horizontal_rescue(
        binary_top2,
        principal_mask,
        traveler_points=extended_points,
    )

    horizontal_rescue_mask = horizontal_result["rescue_mask"]
    binary_top2_guarded = horizontal_result["binary_top2_guarded"]
    horizontal_roi_mask = horizontal_result["roi_mask"]
    horizontal_candidate_mask = horizontal_result["candidate_mask"]
    horizontal_rejected_mask = horizontal_result["rejected_mask"]

    horizontal_rescue_mask, horizontal_layered_tail_removed_mask = (
        filter_layered_horizontal_tail(
            horizontal_rescue_mask,
            principal_mask,
        )
    )

    horizontal_rescue_mask, horizontal_floating_strip_removed_mask = (
        filter_floating_upper_horizontal_strip(
            horizontal_rescue_mask,
            principal_mask,
        )
    )

    horizontal_rescue_mask, horizontal_right_isolated_removed_mask = (
        filter_right_isolated_horizontal_component(
            horizontal_rescue_mask,
            principal_mask,
        )
    )

    horizontal_removed_mask = merge_masks(
        horizontal_layered_tail_removed_mask,
        horizontal_floating_strip_removed_mask,
    )
    horizontal_removed_mask = merge_masks(
        horizontal_removed_mask,
        horizontal_right_isolated_removed_mask,
    )

    horizontal_accepted_mask = horizontal_rescue_mask
    horizontal_rejected_mask = merge_masks(
        horizontal_rejected_mask,
        horizontal_removed_mask,
    )

    if mask_area(horizontal_removed_mask) > 0:
        binary_top2_guarded = binary_top2_guarded.copy()
        binary_top2_guarded[horizontal_removed_mask > 0] = 0

    principal_mask, principal_under_horizontal_removed_mask = (
        filter_principal_tail_under_horizontal(
            principal_mask,
            horizontal_rescue_mask,
        )
    )

    if mask_area(principal_under_horizontal_removed_mask) > 0:
        rejected_principal_mask = merge_masks(
            rejected_principal_mask,
            principal_under_horizontal_removed_mask,
        )

        binary_top2_guarded = binary_top2_guarded.copy()
        binary_top2_guarded[principal_under_horizontal_removed_mask > 0] = 0

    principal_after_horizontal_mask = merge_masks(
        principal_mask,
        horizontal_rescue_mask,
    )

    secondary_result = build_secondary_components(
        binary_top1,
        binary_top2_guarded,
        principal_after_horizontal_mask,
        extended_points,
    )

    secondary_mask_normal = secondary_result["secondary_mask"]
    secondary_roi_mask = secondary_result["roi_mask"]
    secondary_candidate_mask = secondary_result["candidate_mask"]
    secondary_rejected_mask = secondary_result["rejected_mask"]

    secondary_mask_normal, secondary_tail_after_horizontal_removed_mask = (
        filter_secondary_tail_after_horizontal(
            secondary_mask_normal,
            horizontal_rescue_mask,
        )
    )

    secondary_mask_normal, secondary_floating_readd_removed_mask = (
        filter_secondary_floating_strip_after_horizontal_reject(
            secondary_mask_normal,
            principal_mask,
        )
    )

    secondary_mask_normal, left_low_secondary_removed_mask = (
        filter_left_low_far_artifact_components(
            secondary_mask_normal,
            principal_mask,
        )
    )

    secondary_total_removed_mask = merge_masks(
        secondary_tail_after_horizontal_removed_mask,
        secondary_floating_readd_removed_mask,
    )
    secondary_total_removed_mask = merge_masks(
        secondary_total_removed_mask,
        left_low_secondary_removed_mask,
    )

    secondary_rejected_mask = merge_masks(
        secondary_rejected_mask,
        secondary_total_removed_mask,
    )

    if mask_area(secondary_total_removed_mask) > 0:
        binary_top2_guarded = binary_top2_guarded.copy()
        binary_top2_guarded[secondary_total_removed_mask > 0] = 0

    merged_before_gap_rescue = merge_masks(
        principal_after_horizontal_mask,
        secondary_mask_normal,
    )

    gap_rescue_result = gap_rescue_after_secondary(
        binary_top2=binary_top2_guarded,
        principal_mask=principal_after_horizontal_mask,
        secondary_mask=secondary_mask_normal,
        merged_mask=merged_before_gap_rescue,
        traveler_points=extended_points,
    )

    gap_rescue_mask_raw = gap_rescue_result["rescue_mask"]

    gap_rescue_mask, floating_gap_removed_mask = filter_floating_right_gap_rescue(
        gap_rescue_mask_raw,
        principal_after_horizontal_mask,
        secondary_mask_normal,
    )

    gap_rescue_mask, upper_right_gap_removed_mask = filter_upper_right_gap_rescue(
        gap_rescue_mask,
        principal_after_horizontal_mask,
        secondary_mask_normal,
    )

    floating_gap_removed_mask = merge_masks(
        floating_gap_removed_mask,
        upper_right_gap_removed_mask,
    )

    secondary_mask_before_rescue = np.zeros_like(secondary_mask_normal, dtype=np.uint8)
    secondary_mask_before_rescue[secondary_mask_normal > 0] = 255
    secondary_mask_before_rescue[gap_rescue_mask > 0] = 255

    secondary_mask_before_rescue, combined_secondary_gap_removed_mask = (
        filter_secondary_floating_strip_after_horizontal_reject(
            secondary_mask_before_rescue,
            principal_mask,
        )
    )

    secondary_mask_before_rescue, left_low_combined_removed_mask = (
        filter_left_low_far_artifact_components(
            secondary_mask_before_rescue,
            principal_mask,
        )
    )

    combined_secondary_gap_removed_mask = merge_masks(
        combined_secondary_gap_removed_mask,
        left_low_combined_removed_mask,
    )

    if mask_area(combined_secondary_gap_removed_mask) > 0:
        binary_top2_guarded = binary_top2_guarded.copy()
        binary_top2_guarded[combined_secondary_gap_removed_mask > 0] = 0

    merged_before_secondary_rescue = np.zeros_like(
        principal_after_horizontal_mask, dtype=np.uint8
    )
    merged_before_secondary_rescue[principal_after_horizontal_mask > 0] = 255
    merged_before_secondary_rescue[secondary_mask_before_rescue > 0] = 255

    gap_rescue_roi_mask = gap_rescue_result["roi_mask"]
    gap_rescue_candidate_mask = gap_rescue_result["candidate_mask"]
    gap_rescue_rejected_mask = gap_rescue_result["rejected_mask"]

    gap_rescue_rejected_mask = merge_masks(
        gap_rescue_rejected_mask,
        floating_gap_removed_mask,
    )
    gap_rescue_accepted_mask = gap_rescue_mask

    secondary_rescue_result = run_guarded_secondary_rescue(
        binary_top2_guarded=binary_top2_guarded,
        principal_after_horizontal_mask=principal_after_horizontal_mask,
        secondary_mask_before_rescue=secondary_mask_before_rescue,
        merged_before_secondary_rescue=merged_before_secondary_rescue,
        traveler_points=extended_points,
    )

    secondary_mask_after_rescue = secondary_rescue_result["secondary_mask"]
    secondary_rescue_mask_raw = secondary_rescue_result["rescue_mask"]

    secondary_rescue_mask, right_isolated_rescue_removed_mask = (
        filter_right_isolated_secondary_rescue(
            secondary_rescue_mask_raw,
            merged_before_secondary_rescue,
        )
    )

    secondary_rescue_mask, left_low_secondary_rescue_removed_mask = (
        filter_left_low_far_artifact_components(
            secondary_rescue_mask,
            principal_mask,
        )
    )

    merged_final_mask_before_small_clean = np.zeros_like(
        merged_before_secondary_rescue, dtype=np.uint8
    )
    merged_final_mask_before_small_clean[merged_before_secondary_rescue > 0] = 255
    merged_final_mask_before_small_clean[secondary_rescue_mask > 0] = 255

    merged_final_mask, small_component_removed_mask = remove_very_small_components(
        merged_final_mask_before_small_clean,
    )
    secondary_rescue_removed_mask = secondary_rescue_result["removed_secondary_mask"]
    secondary_rescue_removed_mask = merge_masks(
        secondary_rescue_removed_mask,
        right_isolated_rescue_removed_mask,
    )
    secondary_rescue_removed_mask = merge_masks(
        secondary_rescue_removed_mask,
        left_low_secondary_rescue_removed_mask,
    )
    secondary_rescue_roi_mask = secondary_rescue_result["roi_mask"]
    secondary_rescue_candidate_mask = secondary_rescue_result["candidate_mask"]
    secondary_rescue_rejected_mask = secondary_rescue_result["rejected_mask"]

    final_result = build_final_contour(merged_final_mask)
    final_mask = final_result["final_mask"]
    contour_points = final_result["contour_points"]

    traveler_debug = draw_extended_component(
        crop,
        raw_points,
        extended_points,
        components,
        added_component_ids,
    )

    principal_debug = draw_principal_component(
        crop,
        principal_mask,
        rejected_mask=rejected_principal_mask,
        traveler_points=extended_points,
    )

    principal_selector_debug = draw_principal_selector_debug(
        crop,
        principal_mask_initial,
        principal_selector_result,
        traveler_points=extended_points,
    )

    horizontal_rescue_debug = draw_horizontal_rescue_debug(
        crop,
        principal_mask,
        horizontal_rescue_mask,
        horizontal_roi_mask,
        horizontal_candidate_mask,
        horizontal_accepted_mask,
        horizontal_rejected_mask,
        traveler_points=extended_points,
    )

    secondary_roi_debug = draw_secondary_roi(
        crop,
        secondary_roi_mask,
        principal_mask=principal_after_horizontal_mask,
        traveler_points=extended_points,
    )

    secondary_candidates_debug = draw_secondary_candidates(
        crop,
        secondary_candidate_mask,
        rejected_mask=secondary_rejected_mask,
        traveler_points=extended_points,
    )

    secondary_debug = draw_secondary_components(
        crop,
        principal_after_horizontal_mask,
        secondary_mask_normal,
        traveler_points=extended_points,
    )

    gap_rescue_debug = draw_gap_rescue_debug(
        crop,
        principal_after_horizontal_mask,
        secondary_mask_normal,
        gap_rescue_mask,
        gap_rescue_roi_mask,
        gap_rescue_candidate_mask,
        gap_rescue_accepted_mask,
        gap_rescue_rejected_mask,
        traveler_points=extended_points,
    )

    merged_before_secondary_rescue_debug = draw_merged_components(
        crop,
        merged_before_secondary_rescue,
        traveler_points=extended_points,
    )

    secondary_rescue_debug = draw_secondary_rescue_debug(
        crop,
        principal_after_horizontal_mask,
        secondary_mask_after_rescue,
        secondary_rescue_mask,
        removed_secondary_mask=secondary_rescue_removed_mask,
        roi_mask=secondary_rescue_roi_mask,
        candidate_mask=secondary_rescue_candidate_mask,
        rejected_mask=secondary_rescue_rejected_mask,
        traveler_points=extended_points,
    )

    merged_final_debug = draw_secondary_rescue_merged_debug(
        crop,
        merged_final_mask_before_small_clean,
        traveler_points=extended_points,
    )

    small_component_clean_debug = draw_merged_components(
        crop,
        merged_final_mask,
        traveler_points=extended_points,
    )

    source_origin_debug = draw_artifact_source_overlay(
        crop,
        principal_after_horizontal_mask,
        horizontal_rescue_mask,
        secondary_mask_normal,
        gap_rescue_mask,
        secondary_rescue_mask,
        final_mask=merged_final_mask,
        traveler_points=extended_points,
    )

    final_mask_crop_debug = draw_final_mask_on_crop(
        crop,
        final_mask,
    )

    final_contour_crop_debug = draw_final_contour_on_crop(
        crop,
        final_mask,
        contour_points,
    )

    final_mask_original_debug = draw_final_mask_on_original(
        image,
        final_mask,
        crop_box,
    )

    final_contour_original_debug = draw_final_contour_on_original(
        image,
        final_mask,
        contour_points,
        crop_box,
    )

    final_binary_original = project_mask_to_original(
        final_mask,
        image.shape,
        crop_box,
    )

    top2_guided_result = build_top2_guided_pleura(
        binary_top1=binary_top1,
        binary_top2=binary_top2,
        current_pleura_mask=merged_final_mask,
    )
    top2_final_contour_result = build_top2_final_contour(
        crop_bgr=crop,
        current_pleura_mask=merged_final_mask,
        top2_guided_mask=top2_guided_result["top2_guided_mask"],
    )

    top2_unification_result = build_unification(
        crop_bgr=crop,
        top2_final_mask=top2_final_contour_result["final_top2_mask"],
        support_mask=binary_top2,
    )

    binary_top3_contour_on_crop = draw_binary_contour_on_crop(
        crop,
        binary_top3,
        color=(0, 255, 0),
        thickness=1,
    )
    binary_top4_contour_on_crop = draw_binary_contour_on_crop(
        crop,
        binary_top4,
        color=(0, 255, 0),
        thickness=1,
    )

    save_image(CROP_DIR / make_output_name(index, "crop"), crop)
    save_image(PALETTE_7_DIR / make_output_name(index, "palette_7"), palette_7)
    save_image(BINARY_TOP1_DIR / make_output_name(index, "binary_top1"), binary_top1)
    save_image(
        BINARY_TOP2_DIR / make_output_name(index, "binary_top2_secondary_search"),
        binary_top2,
    )
    save_image(
        BINARY_TOP3_DIR / make_output_name(index, "binary_top3_nodule_support"),
        binary_top3,
    )
    save_image(
        BINARY_TOP4_DIR / make_output_name(index, "binary_top4_nodule_support"),
        binary_top4,
    )
    save_image(
        BINARY_TOP3_CONTOUR_DIR
        / make_output_name(index, "binary_top3_contour_on_crop"),
        binary_top3_contour_on_crop,
    )
    save_image(
        BINARY_TOP4_CONTOUR_DIR
        / make_output_name(index, "binary_top4_contour_on_crop"),
        binary_top4_contour_on_crop,
    )
    save_image(
        PRINCIPAL_DIR / make_output_name(index, "principal_component"), principal_debug
    )
    save_image(
        SECONDARY_DIR
        / make_output_name(index, "principal_horizontal_secondary_components"),
        secondary_debug,
    )
    save_image(
        MERGED_FINAL_DIR / make_output_name(index, "merged_final_after_all_rescues"),
        merged_final_debug,
    )
    save_image(
        SMALL_COMPONENT_CLEAN_DIR / make_output_name(index, "small_component_clean"),
        small_component_clean_debug,
    )
    save_image(
        FINAL_MASK_CROP_DIR / make_output_name(index, "final_mask_on_crop"),
        final_mask_crop_debug,
    )
    save_image(
        FINAL_CONTOUR_CROP_DIR / make_output_name(index, "final_contour_on_crop"),
        final_contour_crop_debug,
    )
    save_image(
        FINAL_CONTOUR_ORIGINAL_DIR
        / make_output_name(index, "final_contour_on_original"),
        final_contour_original_debug,
    )

    top2_contour_only = top2_final_contour_result["contour_on_crop"]
    save_image(
        TOP2_CONTOUR_ONLY_DIR
        / make_output_name(index, "top2_final_contour_only_on_crop"),
        top2_contour_only,
    )

    top2_unified_images = top2_unification_result.get("images", {})

    if "06_unified_contour_on_crop" in top2_unified_images:
        top2_unified_contour = top2_unified_images["06_unified_contour_on_crop"]
    elif "special_contour_with_middle_polyline_on_crop" in top2_unified_images:
        top2_unified_contour = top2_unified_images[
            "special_contour_with_middle_polyline_on_crop"
        ]
    elif "step9_final_output_on_crop" in top2_unified_images:
        top2_unified_contour = top2_unified_images["step9_final_output_on_crop"]
    elif "image" in top2_unification_result:
        top2_unified_contour = top2_unification_result["image"]
    else:
        top2_unified_contour = top2_contour_only
    save_image(
        TOP2_UNIFIED_CONTOUR_ONLY_DIR
        / make_output_name(index, "top2_superfragment_check1_contour_on_crop"),
        top2_unified_contour,
    )

    interruption_result = detect_pleural_interruptions(
        base_bgr=crop,
        original_mask=top2_final_contour_result["final_top2_mask"],
        final_mask=top2_unification_result["unified_mask"],
        bridge_mask=top2_unification_result["bridge_mask"],
    )

    save_image(
        PLEURAL_INTERRUPTION_DIR
        / make_output_name(index, "pleural_interruptions_on_unified_contour"),
        interruption_result["interruption_image"],
    )

    save_image(
        PLEURAL_INTERRUPTION_DIR
        / make_output_name(index, "pleural_interruptions_mask"),
        interruption_result["interruption_mask"],
    )

    interruption_comparison = build_interruption_component_comparison(
        principal_image=principal_debug,
        secondary_image=secondary_debug,
        source_origin_image=source_origin_debug,
        interruption_image=interruption_result["interruption_image"],
    )

    save_image(
        PLEURAL_INTERRUPTION_COMPARISON_DIR
        / make_output_name(index, "comparatie_componente_intreruperi"),
        interruption_comparison,
    )

    nodule_result = detect_pleural_nodules(
        base_bgr=crop,
        pleura_mask=top2_unification_result["unified_mask"],
        bridge_mask=top2_unification_result["bridge_mask"],
        interruption_mask=interruption_result["interruption_mask"],
        binary_top3=binary_top3,
        binary_top4=None,
    )

    # Pentru noduli folosim masca de dreptunghi daca detectorul o furnizeaza.
    # nodule_mask ramane masca reala/core, iar nodule_box_mask este pentru marcaj final.
    nodule_visual_mask = nodule_result.get(
        "nodule_box_mask",
        nodule_result["nodule_mask"],
    )

    save_image(
        PLEURAL_NODULE_DIR
        / make_output_name(index, "pleural_nodules_on_unified_contour"),
        nodule_result["nodule_image"],
    )

    save_image(
        PLEURAL_NODULE_DIR / make_output_name(index, "pleural_nodules_mask"),
        nodule_result["nodule_mask"],
    )

    if "nodule_box_mask" in nodule_result:
        save_image(
            PLEURAL_NODULE_DIR / make_output_name(index, "pleural_nodules_box_mask"),
            nodule_result["nodule_box_mask"],
        )

    nodule_boxes_original_debug = draw_nodule_boxes_on_original(
        image,
        nodule_visual_mask,
        crop_box,
    )

    save_image(
        PLEURAL_NODULE_ORIGINAL_BOX_DIR
        / make_output_name(index, "pleural_nodules_boxes_on_original"),
        nodule_boxes_original_debug,
    )

    final_findings_original_debug = draw_final_findings_on_original(
        original_image=image,
        crop_box=crop_box,
        pleura_mask_crop=top2_unification_result["unified_mask"],
        interruption_mask_crop=interruption_result["interruption_mask"],
        nodule_mask_crop=nodule_visual_mask,
    )

    save_image(
        FINAL_FINDINGS_ORIGINAL_DIR
        / make_output_name(
            index, "varianta_finala_pleura_intreruperi_noduli_pe_original"
        ),
        final_findings_original_debug,
    )

    save_image(
        PLEURAL_NODULE_DIR
        / make_output_name(index, "pleural_nodules_top3_working_mask"),
        nodule_result["working_mask"],
    )

    save_image(
        PLEURAL_NODULE_DIR
        / make_output_name(index, "pleural_nodules_candidates_debug"),
        nodule_result["candidate_debug_image"],
    )

    stage0_mask = nodule_result.get("stage0_mask")
    stage1_mask = nodule_result.get("stage1_mask")
    stage2_mask = nodule_result.get("stage2_mask")
    stage3_mask = nodule_result.get("stage3_mask")
    stage4_mask = nodule_result.get("stage4_mask")
    stage5_mask = nodule_result.get("stage5_mask")

    if stage0_mask is not None:
        save_image(
            PLEURAL_NODULE_STEPS_DIR
            / make_output_name(index, "00_stage0_tot_top3_sub_pleura"),
            nodule_result.get(
                "stage0_debug_image", nodule_result["candidate_debug_image"]
            ),
        )

    if stage1_mask is not None:
        save_image(
            PLEURAL_NODULE_STEPS_DIR
            / make_output_name(index, "01_stage1_contact_cu_pleura"),
            nodule_result.get(
                "stage1_debug_image", nodule_result["candidate_debug_image"]
            ),
        )

    if stage2_mask is not None:
        save_image(
            PLEURAL_NODULE_STEPS_DIR
            / make_output_name(index, "02_stage2_coborare_sub_pleura"),
            nodule_result.get(
                "stage2_debug_image", nodule_result["candidate_debug_image"]
            ),
        )

    if stage1_mask is not None and stage2_mask is not None:
        stage1_binary = normalize_debug_mask(stage1_mask)
        stage2_binary = normalize_debug_mask(stage2_mask)
        removed_by_stage2 = np.zeros_like(stage1_binary, dtype=np.uint8)
        removed_by_stage2[(stage1_binary > 0) & (stage2_binary == 0)] = 255

        difference_debug = draw_mask_difference_on_crop(
            crop,
            kept_mask=stage2_binary,
            removed_mask=removed_by_stage2,
            title_text="cyan=ramane dupa stage2 | portocaliu=respins de drop",
        )

        save_image(
            PLEURAL_NODULE_STEPS_DIR
            / make_output_name(index, "03_stage2_diferenta_contact_minus_drop"),
            difference_debug,
        )

    if stage3_mask is not None:
        save_image(
            PLEURAL_NODULE_STEPS_DIR
            / make_output_name(index, "04_stage3_dimensiuni_minime"),
            nodule_result.get(
                "stage3_debug_image", nodule_result["candidate_debug_image"]
            ),
        )

    if stage2_mask is not None and stage3_mask is not None:
        stage2_binary = normalize_debug_mask(stage2_mask)
        stage3_binary = normalize_debug_mask(stage3_mask)
        removed_by_stage3 = np.zeros_like(stage2_binary, dtype=np.uint8)
        removed_by_stage3[(stage2_binary > 0) & (stage3_binary == 0)] = 255

        stage3_difference_debug = draw_mask_difference_on_crop(
            crop,
            kept_mask=stage3_binary,
            removed_mask=removed_by_stage3,
            title_text="cyan=ramane dupa stage3 | portocaliu=respins de dimensiuni",
        )

        save_image(
            PLEURAL_NODULE_STEPS_DIR
            / make_output_name(index, "05_stage3_diferenta_drop_minus_dimensiuni"),
            stage3_difference_debug,
        )

    if stage4_mask is not None:
        save_image(
            PLEURAL_NODULE_STEPS_DIR
            / make_output_name(index, "06_stage4_elimina_componente_prea_mari"),
            nodule_result.get(
                "stage4_debug_image", nodule_result["candidate_debug_image"]
            ),
        )

    if stage3_mask is not None and stage4_mask is not None:
        stage3_binary = normalize_debug_mask(stage3_mask)
        stage4_binary = normalize_debug_mask(stage4_mask)
        removed_by_stage4 = np.zeros_like(stage3_binary, dtype=np.uint8)
        removed_by_stage4[(stage3_binary > 0) & (stage4_binary == 0)] = 255

        stage4_difference_debug = draw_mask_difference_on_crop(
            crop,
            kept_mask=stage4_binary,
            removed_mask=removed_by_stage4,
            title_text="cyan=ramane dupa stage4 | portocaliu=respins fiind prea mare",
        )

        save_image(
            PLEURAL_NODULE_STEPS_DIR
            / make_output_name(index, "07_stage4_diferenta_dimensiuni_minus_prea_mari"),
            stage4_difference_debug,
        )

    if stage5_mask is not None:
        save_image(
            PLEURAL_NODULE_STEPS_DIR
            / make_output_name(index, "08_stage5_elimina_artefacte_mici"),
            nodule_result.get(
                "stage5_debug_image", nodule_result["candidate_debug_image"]
            ),
        )

    if stage4_mask is not None and stage5_mask is not None:
        stage4_binary = normalize_debug_mask(stage4_mask)
        stage5_binary = normalize_debug_mask(stage5_mask)
        removed_by_stage5 = np.zeros_like(stage4_binary, dtype=np.uint8)
        removed_by_stage5[(stage4_binary > 0) & (stage5_binary == 0)] = 255

        stage5_difference_debug = draw_mask_difference_on_crop(
            crop,
            kept_mask=stage5_binary,
            removed_mask=removed_by_stage5,
            title_text="cyan=ramane dupa stage5 | portocaliu=artefact mic eliminat",
        )

        save_image(
            PLEURAL_NODULE_STEPS_DIR
            / make_output_name(
                index, "09_stage5_diferenta_stage4_minus_artefacte_mici"
            ),
            stage5_difference_debug,
        )

    save_image(
        PLEURAL_NODULE_STEPS_DIR / make_output_name(index, "10_final_noduli_marcati"),
        nodule_result["nodule_image"],
    )

    nodule_comparison = build_nodule_component_comparison(
        principal_image=principal_debug,
        secondary_image=secondary_debug,
        source_origin_image=source_origin_debug,
        nodule_image=nodule_result["nodule_image"],
    )

    save_image(
        PLEURAL_NODULE_COMPARISON_DIR
        / make_output_name(index, "comparatie_componente_noduli"),
        nodule_comparison,
    )


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)
    return path


def make_final_contours_contact_sheet(exclude_indices=None):
    if exclude_indices is None:
        exclude_indices = set()

    image_paths = []

    for path in sorted(
        FINAL_CONTOUR_ORIGINAL_DIR.glob("*_final_contour_on_original.png")
    ):
        match = re.match(r"(\d+)_", path.name)

        if match is None:
            continue

        index = int(match.group(1))

        if index in exclude_indices:
            continue

        image_paths.append((index, path))

    if len(image_paths) == 0:
        return

    thumbs = []
    tile_width = 360
    tile_height = 230
    label_height = 28

    for index, path in image_paths:
        image = cv2.imread(str(path))

        if image is None:
            continue

        height, width = image.shape[:2]
        scale = min(tile_width / max(width, 1), tile_height / max(height, 1))

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        tile = np.zeros((tile_height + label_height, tile_width, 3), dtype=np.uint8)

        y0 = label_height + (tile_height - new_height) // 2
        x0 = (tile_width - new_width) // 2
        tile[y0 : y0 + new_height, x0 : x0 + new_width] = resized

        cv2.putText(
            tile,
            f"Imagine {index}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        thumbs.append(tile)

    if len(thumbs) == 0:
        return

    cols = 5
    rows = int(np.ceil(len(thumbs) / cols))

    sheet_height = rows * (tile_height + label_height)
    sheet_width = cols * tile_width
    sheet = np.zeros((sheet_height, sheet_width, 3), dtype=np.uint8)

    for pos, tile in enumerate(thumbs):
        row = pos // cols
        col = pos % cols

        y0 = row * (tile_height + label_height)
        x0 = col * tile_width

        sheet[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile

    save_image(REST_CONTACT_SHEET_PATH, sheet)


def main() -> None:
    reset_dir(config.RESULTS_DIR)

    ensure_dir(CROP_DIR)
    ensure_dir(PALETTE_7_DIR)
    ensure_dir(BINARY_TOP1_DIR)
    ensure_dir(BINARY_TOP2_DIR)
    ensure_dir(BINARY_TOP3_DIR)
    ensure_dir(BINARY_TOP4_DIR)
    ensure_dir(BINARY_TOP3_CONTOUR_DIR)
    ensure_dir(BINARY_TOP4_CONTOUR_DIR)
    ensure_dir(PRINCIPAL_DIR)
    ensure_dir(SECONDARY_DIR)
    ensure_dir(MERGED_FINAL_DIR)
    ensure_dir(SMALL_COMPONENT_CLEAN_DIR)
    ensure_dir(FINAL_MASK_CROP_DIR)
    ensure_dir(FINAL_CONTOUR_CROP_DIR)
    ensure_dir(FINAL_CONTOUR_ORIGINAL_DIR)
    ensure_dir(TOP2_CONTOUR_ONLY_DIR)
    ensure_dir(TOP2_UNIFIED_CONTOUR_ONLY_DIR)
    ensure_dir(PLEURAL_INTERRUPTION_DIR)
    ensure_dir(PLEURAL_INTERRUPTION_COMPARISON_DIR)
    ensure_dir(PLEURAL_NODULE_DIR)
    ensure_dir(PLEURAL_NODULE_COMPARISON_DIR)
    ensure_dir(PLEURAL_NODULE_STEPS_DIR)
    ensure_dir(PLEURAL_NODULE_ORIGINAL_BOX_DIR)
    ensure_dir(FINAL_FINDINGS_ORIGINAL_DIR)

    indices = get_indices_to_process()

    if len(indices) == 0:
        print("Nu s-au gasit imagini de procesat.")
        return

    total = len(indices)

    for current, index in enumerate(indices, start=1):
        process_image(index, current, total)

    make_final_contours_contact_sheet(exclude_indices=set())


if __name__ == "__main__":
    main()
