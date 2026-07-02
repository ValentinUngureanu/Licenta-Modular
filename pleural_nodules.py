from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ============================================================
# PLEURAL NODULES 80 - TOP3 CU CONFIRMARE DIN TOP2
# ============================================================
# Rolul acestei variante:
#   - NU detecteaza noduli finali;
#   - NU calculeaza grosimea pleurei;
#   - NU cauta turturi;
#   - ia binary_top3 si sterge mai intai partea comuna cu pleura deja detectata;
#   - pastreaza apoi DOAR pixelii aflati sub pleura;
#   - sterge primele randuri/pixeli de sus ale fiecarei componente
#     ca sa rupa puntile subtiri orizontale;
#   - sparge bridge-urile / puntile subtiri dintre componente;
#   - abia dupa aceste operatii elimina componentele insulare, fiindca
#     top-peel-ul si spargerea bridge-urilor pot crea insule noi;
#   - aplica apoi filtrul de marime minima pe componentele ramase;
#   - aplica apoi filtrul de marime maxima pentru aria, latimea si inaltimea componentelor;
#   - elimina componentele TOP3 care NU au corespondent de turture si in TOP2;
#   - elimina componentele TOP3 care au deasupra o intrerupere pleurala;
#   - aplica la final un filtru de ingrosare pleurala ADAPTIV: un turture este pastrat
#     daca grosimea pleurei deasupra lui este mai mare decat referinta calculata
#     local din aceeasi pleura, fara praguri fixe in pixeli pentru grosime.
#
# Scop:
#   - Stage 0 arata TOP3 dupa stergerea intersectiei cu pleura detectata;
#   - Stage 1 arata TOP3 curatat, pastrat doar sub linia pleurala;
#   - Stage 2 sterge primele randuri/pixeli de sus din fiecare componenta;
#   - Stage 3 sparge bridge-urile mai subtiri de prag;
#   - Stage 4 elimina insulele create/ramase dupa spargeri;
#   - Stage 5 elimina componentele prea mici/prea mari, componentele sub intreruperi
#   - si componentele fara ingrosare pleurala;
#   - folderul 14 va marca rosu TOP3-ul ramas dupa acesti pasi.
# ============================================================

NODULE_ENABLE = True
FILTER_STAGE = 5

# In varianta 80 revenim la TOP3 ca sursa principala de candidati.
# TOP2 este folosit doar ca filtru de confirmare: daca turturele apare si in TOP2,
# componenta TOP3 poate ramane candidat de nodul.
NODULE_CANDIDATE_SOURCE = "binary_top3"  # "binary_top2" sau "binary_top3"

COLOR_PLEURA = (0, 255, 0)  # verde
COLOR_CANDIDATE = (0, 0, 255)  # rosu
UNDER_PLEURA_START_OFFSET_PX = 1  # pastreaza TOP3 de la bottom_y + offset in jos
CONTACT_WITH_PLEURA_MAX_GAP_PX = (
    5  # componenta trebuie sa aiba pixeli in primii N px sub pleura
)
REMOVE_COMMON_WITH_DETECTED_PLEURA = True
COMMON_PLEURA_DILATE_PX = 0  # 0 = sterge doar intersectia exacta TOP3 ∩ pleura
TOP_PEEL_ROWS_FROM_COMPONENTS_PX = (
    2  # sterge primii N pixeli de sus din fiecare componenta/coloana
)

# Spargere bridge-uri subtiri.
# Ideea: un bridge orizontal subtire are grosime verticala mica.
# Stergem segmentele verticale continue mai mici decat 5 px.
# Cu valoarea 5: se sterg grosimi 1, 2, 3, 4 px; grosimea 5+ ramane.
BREAK_THIN_BRIDGES = True
THIN_BRIDGE_MIN_VERTICAL_THICKNESS_PX = 10

# Filtru 1: marime minima.
# Il aplicam dupa top-peel, spargere bridge-uri si eliminare insule.
FILTER_MIN_COMPONENT_SIZE = True
MIN_COMPONENT_AREA_PX = 500
MIN_COMPONENT_WIDTH_PX = 15
MIN_COMPONENT_HEIGHT_PX = 20

# Filtru 2: marime maxima.
# Componenta este eliminata daca depaseste ORICARE dintre aceste limite.
# Daca vrei sa dezactivezi un criteriu separat, pune valoarea lui pe 0.
FILTER_MAX_COMPONENT_SIZE = True
MAX_COMPONENT_AREA_PX = 2500
MAX_COMPONENT_WIDTH_PX = 1500
MAX_COMPONENT_HEIGHT_PX = 1000

# Filtru 3: latimea medie a turturelui.
# Pentru fiecare componenta calculam latimea medie pe randurile ocupate:
#   average_width = media numarului de pixeli foreground pe fiecare rand al componentei.
# Asta separa cazul "turture subtire vertical" de artefactele late/blocuri.
# O limita pusa pe 0 este ignorata.
FILTER_BY_AVERAGE_TURTURE_WIDTH = True
MIN_AVERAGE_TURTURE_WIDTH_PX = 0.0
MAX_AVERAGE_TURTURE_WIDTH_PX = 65.0

# Filtru 4: confirmare prin TOP2.
# Plecam de la turturii gasiti in TOP3, dar pastram componenta doar daca
# aceeasi zona are suport si in TOP2. Astfel un turture vazut in ambele masti
# este considerat mai credibil pentru nodul.
FILTER_BY_TOP2_TURTURE_CONFIRMATION = True
TOP2_CONFIRM_X_PADDING_PX = 5
TOP2_CONFIRM_Y_PADDING_PX = 5
TOP2_CONFIRM_MIN_OVERLAP_PX = 1

# Filtru 5: regula medicala pentru intreruperi.
# Daca deasupra unei componente TOP3 exista o intrerupere pleurala,
# componenta nu este considerata nodul.
FILTER_COMPONENTS_ABOVE_INTERRUPTION = True
INTERRUPTION_X_PADDING_PX = 4
INTERRUPTION_ABOVE_VERTICAL_TOLERANCE_PX = 6
INTERRUPTION_MIN_OVERLAP_COLUMNS_PX = 1

# Filtru 6: ingrosarea pleurei deasupra turturelui - ADAPTIV.
# Pentru fiecare componenta ramasa, masuram grosimea locala a pleurei in zona
# aflata deasupra componentei si o comparam cu o referinta calculata din aceeasi
# pleura, in stanga/dreapta componentei.
#
# Nu mai folosim praguri fixe de tip: minim 3 px, delta 2 px, ratio 1.20.
# Decizia se adapteaza la fiecare imagine in parte:
#   - local_value = percentila 75 a grosimii deasupra componentei;
#   - reference_value = percentila 60 a grosimii din zonele laterale;
#   - componenta ramane daca local_value >= reference_value.
#
# Sursa recomandata este binary_top2, pentru ca pleura detectata finala poate fi
# doar o linie/forma finala, iar TOP2 pastreaza mai bine banda pleurala.
# Daca binary_top2 nu este disponibil, folosim pleura_mask ca fallback.
FILTER_BY_PLEURA_THICKENING = True
PLEURA_THICKENING_SOURCE = "binary_top2"  # "binary_top2" sau "pleura_mask"
PLEURA_THICKENING_BAND_UP_PX = (
    14  # cauta grosimea deasupra marginii inferioare a pleurei
)
PLEURA_THICKENING_BAND_DOWN_PX = 3  # include putin si sub marginea inferioara
PLEURA_THICKENING_X_PADDING_PX = 8  # extinde zona candidatului stanga/dreapta
PLEURA_THICKENING_REFERENCE_SIDE_PX = 35  # zona laterala folosita ca referinta locala
PLEURA_THICKENING_LOCAL_PERCENTILE = (
    75.0  # grosimea locala reprezentativa deasupra componentei
)
PLEURA_THICKENING_REFERENCE_PERCENTILE = (
    60.0  # referinta laterala normala, calculata local
)
PLEURA_THICKENING_GLOBAL_FALLBACK_PERCENTILE = (
    60.0  # fallback daca lipsesc zone laterale
)
PLEURA_THICKENING_PROFILE_MIN_USABLE_PX = 1.0  # sub acest nivel filtrul devine no-op

COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_SHADOW = (0, 0, 0)


# ============================================================
# Helpers
# ============================================================
def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Imaginea de intrare este goala sau None.")

    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGRA2BGR)

    return image.astype(np.uint8).copy()


def _as_binary_mask(
    mask: np.ndarray | None,
    shape: Tuple[int, int] | None = None,
) -> np.ndarray:
    if mask is None:
        if shape is None:
            raise ValueError("Masca este None si nu exista shape fallback.")
        return np.zeros(shape, dtype=np.uint8)

    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask.copy()

    if shape is not None and gray.shape[:2] != shape:
        gray = cv2.resize(
            gray,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return np.where(gray > 0, 255, 0).astype(np.uint8)


def _draw_contours(
    base_image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> np.ndarray:
    result = _to_bgr(base_image)
    binary = _as_binary_mask(mask, result.shape[:2])

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


def _put_title(image: np.ndarray, title_text: str) -> np.ndarray:
    result = _to_bgr(image)

    cv2.putText(
        result,
        title_text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        COLOR_TEXT,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        title_text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        COLOR_TEXT_SHADOW,
        1,
        cv2.LINE_AA,
    )

    return result


def _build_component_infos(mask: np.ndarray) -> List[Dict[str, object]]:
    binary = _as_binary_mask(mask)
    infos: List[Dict[str, object]] = []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label
        average_width = _compute_component_average_row_width(component)

        infos.append(
            {
                "index": int(len(infos) + 1),
                "x_min": x,
                "x_max": int(x + w - 1),
                "y_min": y,
                "y_max": int(y + h - 1),
                "width": w,
                "height": h,
                "area": area,
                "average_width": round(float(average_width), 2),
                "filter_stage": int(FILTER_STAGE),
                "reason": "top2_bridge_broken_no_islands_min_max_avg_width_no_interruption_with_pleural_thickening_component",
            }
        )

    return infos


def _build_pleura_bottom_profile(pleura_mask: np.ndarray) -> Dict[str, object]:
    """Calculeaza bottom_y[x] pentru pleura.

    Pentru coloanele fara pleura, bottom_y este interpolat intre coloanele valide,
    dar pastram si vectorul valid ca sa stim unde pleura exista real.
    """
    pleura = _as_binary_mask(pleura_mask)
    _height, width = pleura.shape[:2]

    bottom_raw = np.full(width, np.nan, dtype=np.float32)
    valid = np.zeros(width, dtype=bool)

    for x in range(width):
        ys = np.flatnonzero(pleura[:, x] > 0)
        if len(ys) == 0:
            continue
        bottom_raw[x] = float(ys[-1])
        valid[x] = True

    if np.count_nonzero(valid) == 0:
        return {
            "valid": valid,
            "bottom_y": np.zeros(width, dtype=np.float32),
            "x_min": 0,
            "x_max": -1,
        }

    xs_all = np.arange(width, dtype=np.float32)
    xs_valid = xs_all[valid]
    bottom_valid = bottom_raw[valid]
    bottom_interp = np.interp(xs_all, xs_valid, bottom_valid).astype(np.float32)

    return {
        "valid": valid,
        "bottom_y": bottom_interp,
        "x_min": int(xs_valid[0]),
        "x_max": int(xs_valid[-1]),
    }


def _remove_common_with_pleura(
    top3_mask: np.ndarray,
    pleura_mask: np.ndarray,
    dilate_px: int = COMMON_PLEURA_DILATE_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sterge din masca candidat partea comuna cu pleura deja detectata.

    common_mask = candidat ∩ pleura_detectata.
    Daca dilate_px > 0, se sterge si o mica zona din jurul pleurei, dar implicit
    dilate_px = 0 pentru ca aici vrem strict partea comuna.
    """
    top3 = _as_binary_mask(top3_mask, pleura_mask.shape[:2])
    pleura = _as_binary_mask(pleura_mask, top3.shape[:2])

    if not REMOVE_COMMON_WITH_DETECTED_PLEURA:
        empty = np.zeros_like(top3, dtype=np.uint8)
        return top3.copy(), empty

    pleura_zone = pleura.copy()
    if dilate_px > 0 and np.count_nonzero(pleura_zone > 0) > 0:
        size = 2 * int(dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        pleura_zone = cv2.dilate(pleura_zone, kernel, iterations=1)
        pleura_zone[pleura_zone > 0] = 255

    common = np.zeros_like(top3, dtype=np.uint8)
    common[(top3 > 0) & (pleura_zone > 0)] = 255

    cleaned = top3.copy()
    cleaned[pleura_zone > 0] = 0
    cleaned[cleaned > 0] = 255

    return cleaned, common


def _build_under_pleura_mask(
    top3_mask: np.ndarray,
    pleura_mask: np.ndarray,
    start_offset_px: int = UNDER_PLEURA_START_OFFSET_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pastreaza din masca candidat doar pixelii aflati sub marginea inferioara a pleurei.

    under_mask[x, y] = candidat[x, y] daca y >= bottom_y[x] + start_offset_px.
    Nu aplica alte filtre.
    """
    top3 = _as_binary_mask(top3_mask, pleura_mask.shape[:2])
    profile = _build_pleura_bottom_profile(pleura_mask)
    bottom_y = np.asarray(profile["bottom_y"], dtype=np.float32)
    valid = np.asarray(profile["valid"], dtype=bool)

    height, width = top3.shape[:2]
    under = np.zeros_like(top3, dtype=np.uint8)
    search_band = np.zeros_like(top3, dtype=np.uint8)

    for x in range(width):
        if not bool(valid[x]):
            continue

        y_start = int(round(float(bottom_y[x]))) + int(start_offset_px)
        y_start = max(0, min(height - 1, y_start))

        search_band[y_start:height, x] = 255
        under[y_start:height, x] = top3[y_start:height, x]

    under[under > 0] = 255
    return under, search_band


def _build_contact_band_under_pleura(
    shape: Tuple[int, int],
    pleura_mask: np.ndarray,
    start_offset_px: int = UNDER_PLEURA_START_OFFSET_PX,
    max_gap_px: int = CONTACT_WITH_PLEURA_MAX_GAP_PX,
) -> np.ndarray:
    """Construieste banda de contact imediat sub pleura.

    O componenta candidat de sub pleura este pastrata doar daca intersecteaza
    aceasta banda. Astfel eliminam insulele care plutesc mai jos si nu pornesc
    din zona pleurala.
    """
    height, width = shape[:2]
    profile = _build_pleura_bottom_profile(pleura_mask)
    bottom_y = np.asarray(profile["bottom_y"], dtype=np.float32)
    valid = np.asarray(profile["valid"], dtype=bool)

    band = np.zeros((height, width), dtype=np.uint8)

    for x in range(width):
        if not bool(valid[x]):
            continue

        y1 = int(round(float(bottom_y[x]))) + int(start_offset_px)
        y2 = int(round(float(bottom_y[x]))) + int(max_gap_px)
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))

        if y2 >= y1:
            band[y1 : y2 + 1, x] = 255

    return band


def _remove_island_components(
    under_mask: np.ndarray,
    contact_band: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Elimina componentele insulare din masca candidat sub pleura.

    Pastram o componenta doar daca are cel putin un pixel in banda de contact
    de sub pleura. Componentele complet separate, aflate mai jos, sunt mutate
    in removed_mask.
    """
    under = _as_binary_mask(under_mask)
    contact = _as_binary_mask(contact_band, under.shape[:2])

    kept = np.zeros_like(under, dtype=np.uint8)
    removed = np.zeros_like(under, dtype=np.uint8)

    if np.count_nonzero(under > 0) == 0:
        return kept, removed

    num_labels, labels, _stats, _ = cv2.connectedComponentsWithStats(
        under,
        connectivity=8,
    )

    for label in range(1, num_labels):
        component = labels == label
        if np.count_nonzero(component) == 0:
            continue

        touches_contact = np.count_nonzero(component & (contact > 0)) > 0

        if touches_contact:
            kept[component] = 255
        else:
            removed[component] = 255

    return kept, removed


def _peel_top_rows_from_components(
    mask: np.ndarray,
    rows_to_remove: int = TOP_PEEL_ROWS_FROM_COMPONENTS_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sterge primele randuri/pixeli de sus din fiecare componenta.

    Nu stergem un dreptunghi global, ci facem un "peel" pe fiecare coloana
    a fiecarei componente: pentru fiecare coloana ocupata de componenta, se
    elimina primii N pixeli foreground vazuti de sus in jos.

    Motivul este practic: daca mai multe coborari sunt unite printr-o banda
    subtire de sus, eliminarea primilor 1-2 pixeli poate rupe puntea si poate
    transforma o componenta mare in mai multe componente separate.
    """
    binary = _as_binary_mask(mask)

    if rows_to_remove <= 0 or np.count_nonzero(binary > 0) == 0:
        return binary.copy(), np.zeros_like(binary, dtype=np.uint8)

    peeled = binary.copy()
    removed = np.zeros_like(binary, dtype=np.uint8)

    num_labels, labels, _stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    for label in range(1, num_labels):
        component = labels == label
        ys, xs = np.where(component)

        if len(xs) == 0:
            continue

        for x in np.unique(xs):
            col_ys = ys[xs == x]
            if len(col_ys) == 0:
                continue

            col_ys_sorted = np.sort(col_ys)
            remove_count = min(int(rows_to_remove), len(col_ys_sorted))
            remove_ys = col_ys_sorted[:remove_count]

            peeled[remove_ys, x] = 0
            removed[remove_ys, x] = 255

    peeled[peeled > 0] = 255
    return peeled, removed


def _break_thin_bridges_by_vertical_thickness(
    mask: np.ndarray,
    min_vertical_thickness_px: int = THIN_BRIDGE_MIN_VERTICAL_THICKNESS_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sparge bridge-urile subtiri dintre componente.

    Criteriul este simplu si controlabil:
      - pentru fiecare coloana x, cautam segmente verticale continue de pixeli;
      - daca un segment are grosimea verticala < min_vertical_thickness_px,
        il stergem;
      - astfel, puntile orizontale subtiri se rup, iar turturii verticali,
        care au inaltime mai mare, raman.

    Cu min_vertical_thickness_px = 5:
      - segmentele cu grosime 1, 2, 3 sau 4 px sunt sterse;
      - segmentele cu grosime 5 px sau mai mult sunt pastrate.
    """
    binary = _as_binary_mask(mask)
    kept = binary.copy()
    removed = np.zeros_like(binary, dtype=np.uint8)

    if (
        not BREAK_THIN_BRIDGES
        or min_vertical_thickness_px <= 1
        or np.count_nonzero(binary > 0) == 0
    ):
        return kept, removed

    height, width = binary.shape[:2]

    for x in range(width):
        ys = np.flatnonzero(binary[:, x] > 0)
        if len(ys) == 0:
            continue

        run_start = int(ys[0])
        prev_y = int(ys[0])

        for y_value in ys[1:]:
            y = int(y_value)
            if y == prev_y + 1:
                prev_y = y
                continue

            run_end = prev_y
            run_len = run_end - run_start + 1
            if run_len < int(min_vertical_thickness_px):
                kept[run_start : run_end + 1, x] = 0
                removed[run_start : run_end + 1, x] = 255

            run_start = y
            prev_y = y

        run_end = prev_y
        run_len = run_end - run_start + 1
        if run_len < int(min_vertical_thickness_px):
            kept[run_start : run_end + 1, x] = 0
            removed[run_start : run_end + 1, x] = 255

    kept[kept > 0] = 255
    return kept, removed


def _filter_components_by_min_size(
    mask: np.ndarray,
    min_area_px: int = MIN_COMPONENT_AREA_PX,
    min_width_px: int = MIN_COMPONENT_WIDTH_PX,
    min_height_px: int = MIN_COMPONENT_HEIGHT_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pastreaza doar componentele care trec filtrul de marime minima.

    Filtrul este intentionat simplu si separat de filtrul de marime maxima,
    pe care il vom adauga ulterior. Momentan eliminam doar fragmentele foarte
    mici care nu ne intereseaza.
    """
    binary = _as_binary_mask(mask)
    kept = np.zeros_like(binary, dtype=np.uint8)
    removed = np.zeros_like(binary, dtype=np.uint8)

    if not FILTER_MIN_COMPONENT_SIZE or np.count_nonzero(binary > 0) == 0:
        return binary.copy(), removed

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        is_valid = True
        if area < int(min_area_px):
            is_valid = False
        elif width < int(min_width_px):
            is_valid = False
        elif height < int(min_height_px):
            is_valid = False

        if is_valid:
            kept[component] = 255
        else:
            removed[component] = 255

    return kept, removed


def _filter_components_by_max_size(
    mask: np.ndarray,
    max_area_px: int = MAX_COMPONENT_AREA_PX,
    max_width_px: int = MAX_COMPONENT_WIDTH_PX,
    max_height_px: int = MAX_COMPONENT_HEIGHT_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Elimina componentele care depasesc filtrul de marime maxima.

    Componenta este pastrata doar daca respecta toate limitele maxime active:
      - area <= MAX_COMPONENT_AREA_PX;
      - width <= MAX_COMPONENT_WIDTH_PX;
      - height <= MAX_COMPONENT_HEIGHT_PX.

    O limita pusa pe 0 este ignorata, ca sa poti dezactiva separat aria,
    latimea sau inaltimea maxima.
    """
    binary = _as_binary_mask(mask)
    kept = np.zeros_like(binary, dtype=np.uint8)
    removed = np.zeros_like(binary, dtype=np.uint8)

    if not FILTER_MAX_COMPONENT_SIZE or np.count_nonzero(binary > 0) == 0:
        return binary.copy(), removed

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        is_valid = True
        if int(max_area_px) > 0 and area > int(max_area_px):
            is_valid = False
        elif int(max_width_px) > 0 and width > int(max_width_px):
            is_valid = False
        elif int(max_height_px) > 0 and height > int(max_height_px):
            is_valid = False

        if is_valid:
            kept[component] = 255
        else:
            removed[component] = 255

    return kept, removed


def _compute_component_average_row_width(component: np.ndarray) -> float:
    """Calculeaza latimea medie reala a unei componente.

    Pentru fiecare rand y pe care apare componenta, numaram cati pixeli are
    componenta pe acel rand. Media acestor valori reprezinta latimea medie.

    Exemplu:
      - un turture vertical subtire are, de obicei, latime medie mica;
      - un artefact/bloc lat are latime medie mare, chiar daca inaltimea e mare.
    """
    ys = np.flatnonzero(np.any(component, axis=1))
    if ys.size == 0:
        return 0.0

    row_widths = []
    for y in ys:
        row_widths.append(float(np.count_nonzero(component[int(y), :])))

    if len(row_widths) == 0:
        return 0.0

    return float(np.mean(np.asarray(row_widths, dtype=np.float32)))


def _filter_components_by_average_turture_width(
    mask: np.ndarray,
    min_average_width_px: float = MIN_AVERAGE_TURTURE_WIDTH_PX,
    max_average_width_px: float = MAX_AVERAGE_TURTURE_WIDTH_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filtreaza componentele dupa latimea medie a turturelui.

    Spre deosebire de bounding-box width, latimea medie se calculeaza din forma
    reala a componentei. Pentru fiecare rand ocupat de componenta, se numara
    pixelii foreground, apoi se face media.

    Componenta ramane daca respecta limitele active:
      - average_width >= MIN_AVERAGE_TURTURE_WIDTH_PX, daca minimul > 0;
      - average_width <= MAX_AVERAGE_TURTURE_WIDTH_PX, daca maximul > 0.
    """
    binary = _as_binary_mask(mask)
    kept = np.zeros_like(binary, dtype=np.uint8)
    removed = np.zeros_like(binary, dtype=np.uint8)

    if not FILTER_BY_AVERAGE_TURTURE_WIDTH or np.count_nonzero(binary > 0) == 0:
        return binary.copy(), removed

    num_labels, labels, _stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    for label in range(1, num_labels):
        component = labels == label
        average_width = _compute_component_average_row_width(component)

        is_valid = True
        if float(min_average_width_px) > 0 and average_width < float(
            min_average_width_px
        ):
            is_valid = False
        elif float(max_average_width_px) > 0 and average_width > float(
            max_average_width_px
        ):
            is_valid = False

        if is_valid:
            kept[component] = 255
        else:
            removed[component] = 255

    return kept, removed


def _filter_components_by_top2_turture_confirmation(
    mask: np.ndarray,
    top2_support_mask: np.ndarray | None,
    x_padding_px: int = TOP2_CONFIRM_X_PADDING_PX,
    y_padding_px: int = TOP2_CONFIRM_Y_PADDING_PX,
    min_overlap_px: int = TOP2_CONFIRM_MIN_OVERLAP_PX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pastreaza doar componentele TOP3 care apar si in TOP2.

    Logica:
      - componenta candidat este extrasa din TOP3;
      - construim o zona de verificare in jurul bounding-box-ului componentei;
      - daca in acea zona exista pixeli TOP2, consideram ca turturele are suport
        si in TOP2;
      - daca nu exista suport TOP2, componenta este eliminata.

    top2_support_mask trebuie sa fie deja curatat ca masca de sub pleura, ca sa
    nu confundam banda pleurala cu turturele.
    """
    binary = _as_binary_mask(mask)
    kept = np.zeros_like(binary, dtype=np.uint8)
    removed = np.zeros_like(binary, dtype=np.uint8)
    support_zone_mask = np.zeros_like(binary, dtype=np.uint8)

    if not FILTER_BY_TOP2_TURTURE_CONFIRMATION or np.count_nonzero(binary > 0) == 0:
        return binary.copy(), removed, support_zone_mask

    if top2_support_mask is None:
        # Daca TOP2 nu este disponibil, nu stergem agresiv componentele.
        return binary.copy(), removed, support_zone_mask

    top2_support = _as_binary_mask(top2_support_mask, binary.shape[:2])

    height, width = binary.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    for label in range(1, num_labels):
        component = labels == label

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        x1 = max(0, x - int(x_padding_px))
        x2 = min(width - 1, x + w - 1 + int(x_padding_px))
        y1 = max(0, y - int(y_padding_px))
        y2 = min(height - 1, y + h - 1 + int(y_padding_px))

        zone = top2_support[y1 : y2 + 1, x1 : x2 + 1] > 0
        overlap_px = int(np.count_nonzero(zone))

        if overlap_px >= int(min_overlap_px):
            kept[component] = 255
            support_zone_mask[y1 : y2 + 1, x1 : x2 + 1] = np.where(
                zone,
                255,
                support_zone_mask[y1 : y2 + 1, x1 : x2 + 1],
            ).astype(np.uint8)
        else:
            removed[component] = 255

    return kept, removed, support_zone_mask


def _filter_components_above_interruption(
    mask: np.ndarray,
    interruption_mask: np.ndarray | None,
    x_padding_px: int = INTERRUPTION_X_PADDING_PX,
    vertical_tolerance_px: int = INTERRUPTION_ABOVE_VERTICAL_TOLERANCE_PX,
    min_overlap_columns_px: int = INTERRUPTION_MIN_OVERLAP_COLUMNS_PX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elimina componentele care au deasupra o intrerupere pleurala.

    Regula medicala folosita aici este simpla:
      - unde pleura este intrerupta, nu clasificam structura de sub acea zona ca nodul;
      - pentru fiecare componenta candidat, luam intervalul ei pe X, extins putin;
      - cautam pixeli din interruption_mask deasupra componentei;
      - daca intreruperea se suprapune pe X cu componenta, componenta este eliminata.

    vertical_tolerance_px permite ca intreruperea sa fie foarte aproape de partea
    de sus a componentei, nu doar strict deasupra y_min.
    """
    binary = _as_binary_mask(mask)
    kept = np.zeros_like(binary, dtype=np.uint8)
    removed = np.zeros_like(binary, dtype=np.uint8)
    interruption_above_zone = np.zeros_like(binary, dtype=np.uint8)

    if (
        not FILTER_COMPONENTS_ABOVE_INTERRUPTION
        or interruption_mask is None
        or np.count_nonzero(binary > 0) == 0
    ):
        return binary.copy(), removed, interruption_above_zone

    interruption = _as_binary_mask(interruption_mask, binary.shape[:2])
    if np.count_nonzero(interruption > 0) == 0:
        return binary.copy(), removed, interruption_above_zone

    height, width = binary.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    for label in range(1, num_labels):
        component = labels == label

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])

        x1 = max(0, x - int(x_padding_px))
        x2 = min(width - 1, x + w - 1 + int(x_padding_px))
        y2 = min(height - 1, y + int(vertical_tolerance_px))

        if x2 < x1 or y2 < 0:
            kept[component] = 255
            continue

        # Zona verificata: toate intreruperile aflate deasupra componentei
        # sau foarte aproape de partea ei superioara.
        zone = interruption[0 : y2 + 1, x1 : x2 + 1] > 0
        if np.count_nonzero(zone) == 0:
            kept[component] = 255
            continue

        overlap_by_column = np.any(zone, axis=0)
        overlap_columns = int(np.count_nonzero(overlap_by_column))

        if overlap_columns >= int(min_overlap_columns_px):
            removed[component] = 255
            interruption_above_zone[0 : y2 + 1, x1 : x2 + 1] = np.where(
                zone,
                255,
                interruption_above_zone[0 : y2 + 1, x1 : x2 + 1],
            ).astype(np.uint8)
        else:
            kept[component] = 255

    return kept, removed, interruption_above_zone


def _choose_pleura_thickening_source(
    binary_top2: np.ndarray | None,
    pleura_mask: np.ndarray,
) -> np.ndarray:
    """Alege masca folosita pentru masurarea ingrosarii pleurale.

    Preferam binary_top2, fiindca acolo se vede mai bine banda pleurala.
    Daca binary_top2 lipseste sau este gol, folosim pleura_mask.
    """
    pleura = _as_binary_mask(pleura_mask)

    if PLEURA_THICKENING_SOURCE == "binary_top2" and binary_top2 is not None:
        top2 = _as_binary_mask(binary_top2, pleura.shape[:2])
        if np.count_nonzero(top2 > 0) > 0:
            return top2

    return pleura.copy()


def _build_pleura_thickness_profile(
    source_mask: np.ndarray,
    detected_pleura_mask: np.ndarray,
    band_up_px: int = PLEURA_THICKENING_BAND_UP_PX,
    band_down_px: int = PLEURA_THICKENING_BAND_DOWN_PX,
) -> Dict[str, object]:
    """Calculeaza un profil 1D al grosimii pleurale pe fiecare coloana.

    Pentru fiecare coloana x:
      - luam marginea inferioara a pleurei detectate, bottom_y[x];
      - construim o fereastra verticala [bottom_y - band_up, bottom_y + band_down];
      - numaram cati pixeli din source_mask exista in acea fereastra.

    Rezultatul thickness[x] poate fi folosit pentru a vedea daca deasupra unei
    componente candidat exista o ingrosare locala a pleurei.
    """
    source = _as_binary_mask(source_mask, detected_pleura_mask.shape[:2])
    pleura = _as_binary_mask(detected_pleura_mask, source.shape[:2])

    height, width = source.shape[:2]
    profile = _build_pleura_bottom_profile(pleura)
    bottom_y = np.asarray(profile["bottom_y"], dtype=np.float32)
    valid = np.asarray(profile["valid"], dtype=bool)

    thickness = np.zeros(width, dtype=np.float32)
    band_mask = np.zeros_like(source, dtype=np.uint8)
    source_in_band = np.zeros_like(source, dtype=np.uint8)

    for x in range(width):
        if not bool(valid[x]):
            continue

        y_center = int(round(float(bottom_y[x])))
        y1 = y_center - int(band_up_px)
        y2 = y_center + int(band_down_px)
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))

        if y2 < y1:
            continue

        band_mask[y1 : y2 + 1, x] = 255
        col = source[y1 : y2 + 1, x] > 0
        thickness[x] = float(np.count_nonzero(col))
        source_in_band[y1 : y2 + 1, x] = np.where(col, 255, 0).astype(np.uint8)

    return {
        "thickness": thickness,
        "valid": valid,
        "band_mask": band_mask,
        "source_in_band": source_in_band,
    }


def _safe_percentile(
    values: np.ndarray, percentile: float, fallback: float = 0.0
) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(fallback)
    return float(np.percentile(values, percentile))


def _filter_components_by_pleura_thickening(
    mask: np.ndarray,
    detected_pleura_mask: np.ndarray,
    pleura_thickening_source_mask: np.ndarray,
    x_padding_px: int = PLEURA_THICKENING_X_PADDING_PX,
    side_reference_px: int = PLEURA_THICKENING_REFERENCE_SIDE_PX,
    local_percentile: float = PLEURA_THICKENING_LOCAL_PERCENTILE,
    reference_percentile: float = PLEURA_THICKENING_REFERENCE_PERCENTILE,
    global_fallback_percentile: float = PLEURA_THICKENING_GLOBAL_FALLBACK_PERCENTILE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pastreaza doar componentele aflate sub o pleura local ingrosata.

    Varianta adaptiva, fara praguri fixe in pixeli pentru grosime.

    Pentru fiecare componenta ramasa:
      - luam intervalul ei pe axa X, extins cu x_padding_px;
      - local_value = percentila local_percentile a grosimii pleurei deasupra ei;
      - reference_value = percentila reference_percentile a grosimii din stanga/dreapta;
      - daca nu exista suficienta referinta laterala, folosim o referinta globala
        calculata din aceeasi pleura.

    Conditia de pastrare este simpla si adaptiva:
      local_value >= reference_value

    Astfel, un nodul de pe o pleura subtire nu este eliminat doar pentru ca nu
    atinge un prag absolut fix, de exemplu 3 px.
    """
    binary = _as_binary_mask(mask)
    kept = np.zeros_like(binary, dtype=np.uint8)
    removed = np.zeros_like(binary, dtype=np.uint8)

    thickness_data = _build_pleura_thickness_profile(
        pleura_thickening_source_mask,
        detected_pleura_mask,
        band_up_px=PLEURA_THICKENING_BAND_UP_PX,
        band_down_px=PLEURA_THICKENING_BAND_DOWN_PX,
    )
    thickness = np.asarray(thickness_data["thickness"], dtype=np.float32)
    valid = np.asarray(thickness_data["valid"], dtype=bool)
    band_mask = np.asarray(thickness_data["band_mask"], dtype=np.uint8)
    source_in_band = np.asarray(thickness_data["source_in_band"], dtype=np.uint8)

    if (
        not FILTER_BY_PLEURA_THICKENING
        or np.count_nonzero(binary > 0) == 0
        or np.count_nonzero(valid) == 0
    ):
        return binary.copy(), removed, band_mask, source_in_band

    valid_values = thickness[valid]
    usable_global_values = valid_values[valid_values > 0]
    if usable_global_values.size == 0:
        usable_global_values = valid_values

    global_reference = _safe_percentile(
        usable_global_values,
        float(global_fallback_percentile),
        fallback=0.0,
    )
    global_profile_peak = _safe_percentile(usable_global_values, 90, fallback=0.0)

    # Daca masca folosita pentru grosime este practic o linie subtire, nu avem
    # informatie reala despre ingrosare. In cazul asta nu aplicam filtrul.
    if global_profile_peak <= float(PLEURA_THICKENING_PROFILE_MIN_USABLE_PX):
        return binary.copy(), removed, band_mask, source_in_band

    height, width = binary.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    for label in range(1, num_labels):
        component = labels == label

        x = int(stats[label, cv2.CC_STAT_LEFT])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        x1 = max(0, x - int(x_padding_px))
        x2 = min(width - 1, x + w - 1 + int(x_padding_px))

        local_cols = np.arange(x1, x2 + 1)
        local_cols = local_cols[valid[local_cols]]

        if local_cols.size == 0:
            # Nu avem informatie de grosime exact deasupra componentei.
            # Nu o stergem agresiv, ca sa nu pierdem noduli buni din cauza profilului incomplet.
            kept[component] = 255
            continue

        local_values = thickness[local_cols]
        usable_local_values = local_values[local_values > 0]
        if usable_local_values.size == 0:
            usable_local_values = local_values

        local_value = _safe_percentile(
            usable_local_values,
            float(local_percentile),
            fallback=0.0,
        )

        left1 = max(0, x1 - int(side_reference_px))
        left2 = max(0, x1 - 1)
        right1 = min(width - 1, x2 + 1)
        right2 = min(width - 1, x2 + int(side_reference_px))

        reference_values: List[float] = []

        if left2 >= left1:
            left_cols = np.arange(left1, left2 + 1)
            left_cols = left_cols[valid[left_cols]]
            if left_cols.size > 0:
                vals = thickness[left_cols]
                vals = vals[vals > 0]
                reference_values.extend([float(v) for v in vals])

        if right2 >= right1:
            right_cols = np.arange(right1, right2 + 1)
            right_cols = right_cols[valid[right_cols]]
            if right_cols.size > 0:
                vals = thickness[right_cols]
                vals = vals[vals > 0]
                reference_values.extend([float(v) for v in vals])

        if len(reference_values) > 0:
            reference = _safe_percentile(
                np.asarray(reference_values, dtype=np.float32),
                float(reference_percentile),
                fallback=global_reference,
            )
        else:
            reference = global_reference

        # Conditie adaptiva: nu cerem un prag fix de pixeli.
        # Componenta este pastrata daca pleura deasupra ei este cel putin la
        # nivelul referintei locale a aceleiasi pleure.
        is_valid = bool(local_value >= float(reference))

        if is_valid:
            kept[component] = 255
        else:
            removed[component] = 255

    return kept, removed, band_mask, source_in_band


# ============================================================
# Debug drawing
# ============================================================
def draw_nodule_marking(
    base_bgr: np.ndarray,
    pleura_mask: np.ndarray,
    nodule_mask: np.ndarray,
    title_text: str | None = None,
) -> np.ndarray:
    result = _to_bgr(base_bgr)
    result = _draw_contours(result, pleura_mask, COLOR_PLEURA, thickness=1)
    result = _draw_contours(result, nodule_mask, COLOR_CANDIDATE, thickness=1)

    if title_text is None:
        title_text = "TOP3 SUB PLEURA | verde=pleura | rosu=noduli candidati"

    return _put_title(result, title_text)


def draw_nodule_candidate_debug(
    base_bgr: np.ndarray,
    pleura_mask: np.ndarray,
    top3_mask: np.ndarray,
    title_text: str | None = None,
) -> np.ndarray:
    result = _to_bgr(base_bgr)
    result = _draw_contours(result, top3_mask, COLOR_CANDIDATE, thickness=1)
    result = _draw_contours(result, pleura_mask, COLOR_PLEURA, thickness=1)

    if title_text is None:
        title_text = "DEBUG TOP3 CANDIDATI | rosu=TOP3 pastrat"

    return _put_title(result, title_text)


def _empty_result(base: np.ndarray, pleura: np.ndarray) -> Dict[str, object]:
    empty = np.zeros_like(pleura, dtype=np.uint8)
    debug = draw_nodule_candidate_debug(base, pleura, empty)

    result: Dict[str, object] = {
        "nodule_mask": empty,
        "nodule_core_mask": empty,
        "nodule_box_mask": empty,
        "candidate_mask": empty,
        "rejected_mask": empty,
        "under_structure_mask": empty,
        "under_structure_rejected_mask": empty,
        "working_mask": empty,
        "excluded_mask": empty,
        "search_band_mask": empty,
        "contact_zone_mask": empty,
        "interruption_exclusion_mask": empty,
        "nodule_image": draw_nodule_marking(base, pleura, empty),
        "candidate_debug_image": debug,
        "nodule_infos": [],
        "nodule_count": 0,
        "stage4_removed_thin_bridge_mask": empty.copy(),
        "stage5_removed_small_mask": empty.copy(),
        "stage5_removed_large_mask": empty.copy(),
        "stage5_removed_bad_average_width_mask": empty.copy(),
        "stage5_removed_no_top2_support_mask": empty.copy(),
        "stage5_removed_above_interruption_mask": empty.copy(),
        "stage5_removed_no_thickening_mask": empty.copy(),
        "interruption_above_zone_mask": empty.copy(),
        "top2_turture_support_zone_mask": empty.copy(),
        "pleura_thickening_band_mask": empty.copy(),
        "pleura_thickening_source_in_band_mask": empty.copy(),
    }

    for stage in range(0, 6):
        result[f"stage{stage}_mask"] = empty.copy()
        result[f"stage{stage}_rejected_mask"] = empty.copy()
        result[f"stage{stage}_debug_image"] = debug.copy()

    return result


# ============================================================
# Main API - compatibil cu main.py
# ============================================================
def detect_pleural_nodules(
    base_bgr: np.ndarray,
    pleura_mask: np.ndarray,
    bridge_mask: np.ndarray | None = None,
    interruption_mask: np.ndarray | None = None,
    binary_top3: np.ndarray | None = None,
    binary_top4: np.ndarray | None = None,
    binary_top2: np.ndarray | None = None,
) -> Dict[str, object]:
    _ = bridge_mask
    _ = binary_top4
    base = _to_bgr(base_bgr)
    pleura = _as_binary_mask(pleura_mask, base.shape[:2])

    # Varianta 80: revenim la TOP3 ca sursa principala de candidati.
    # TOP2 este folosit separat doar ca masca de confirmare pentru turturi.
    candidate_source = (
        binary_top3 if NODULE_CANDIDATE_SOURCE == "binary_top3" else binary_top2
    )
    if candidate_source is None and binary_top3 is not None:
        candidate_source = binary_top3

    if not NODULE_ENABLE or candidate_source is None:
        return _empty_result(base, pleura)

    top3 = _as_binary_mask(candidate_source, pleura.shape[:2])
    interruption = _as_binary_mask(interruption_mask, pleura.shape[:2])
    top3_without_pleura, common_with_pleura = _remove_common_with_pleura(
        top3,
        pleura,
        dilate_px=COMMON_PLEURA_DILATE_PX,
    )
    top3_under_pleura, search_band = _build_under_pleura_mask(
        top3_without_pleura,
        pleura,
        start_offset_px=UNDER_PLEURA_START_OFFSET_PX,
    )

    top2_support_under_pleura = None
    if binary_top2 is not None:
        top2_raw = _as_binary_mask(binary_top2, pleura.shape[:2])
        top2_without_pleura, _common_top2_with_pleura = _remove_common_with_pleura(
            top2_raw,
            pleura,
            dilate_px=COMMON_PLEURA_DILATE_PX,
        )
        top2_support_under_pleura, _top2_support_band = _build_under_pleura_mask(
            top2_without_pleura,
            pleura,
            start_offset_px=UNDER_PLEURA_START_OFFSET_PX,
        )

    contact_band = _build_contact_band_under_pleura(
        top3.shape[:2],
        pleura,
        start_offset_px=UNDER_PLEURA_START_OFFSET_PX,
        max_gap_px=CONTACT_WITH_PLEURA_MAX_GAP_PX,
    )
    # IMPORTANT:
    # Operatiile care pot sparge o componenta mare in bucati se fac INAINTE
    # de eliminarea insulelor. Altfel, insulele aparute dupa bridge-breaking
    # nu mai sunt curatate.
    top3_top_peeled, removed_top_rows = _peel_top_rows_from_components(
        top3_under_pleura,
        rows_to_remove=TOP_PEEL_ROWS_FROM_COMPONENTS_PX,
    )
    top3_bridge_broken, removed_thin_bridges = (
        _break_thin_bridges_by_vertical_thickness(
            top3_top_peeled,
            min_vertical_thickness_px=THIN_BRIDGE_MIN_VERTICAL_THICKNESS_PX,
        )
    )
    top3_no_islands, removed_islands = _remove_island_components(
        top3_bridge_broken,
        contact_band,
    )
    top3_min_size, removed_too_small = _filter_components_by_min_size(
        top3_no_islands,
        min_area_px=MIN_COMPONENT_AREA_PX,
        min_width_px=MIN_COMPONENT_WIDTH_PX,
        min_height_px=MIN_COMPONENT_HEIGHT_PX,
    )
    top3_max_size, removed_too_large = _filter_components_by_max_size(
        top3_min_size,
        max_area_px=MAX_COMPONENT_AREA_PX,
        max_width_px=MAX_COMPONENT_WIDTH_PX,
        max_height_px=MAX_COMPONENT_HEIGHT_PX,
    )
    top3_avg_width, removed_bad_average_width = (
        _filter_components_by_average_turture_width(
            top3_max_size,
            min_average_width_px=MIN_AVERAGE_TURTURE_WIDTH_PX,
            max_average_width_px=MAX_AVERAGE_TURTURE_WIDTH_PX,
        )
    )
    top3_confirmed_by_top2, removed_no_top2_support, top2_turture_support_zone = (
        _filter_components_by_top2_turture_confirmation(
            top3_avg_width,
            top2_support_mask=top2_support_under_pleura,
            x_padding_px=TOP2_CONFIRM_X_PADDING_PX,
            y_padding_px=TOP2_CONFIRM_Y_PADDING_PX,
            min_overlap_px=TOP2_CONFIRM_MIN_OVERLAP_PX,
        )
    )
    top3_no_interruption, removed_above_interruption, interruption_above_zone = (
        _filter_components_above_interruption(
            top3_confirmed_by_top2,
            interruption_mask=interruption,
            x_padding_px=INTERRUPTION_X_PADDING_PX,
            vertical_tolerance_px=INTERRUPTION_ABOVE_VERTICAL_TOLERANCE_PX,
            min_overlap_columns_px=INTERRUPTION_MIN_OVERLAP_COLUMNS_PX,
        )
    )
    pleura_thickening_source = _choose_pleura_thickening_source(
        binary_top2,
        pleura,
    )
    (
        top3_after_thickening,
        removed_no_thickening,
        pleura_thickening_band,
        pleura_source_in_band,
    ) = _filter_components_by_pleura_thickening(
        top3_no_interruption,
        detected_pleura_mask=pleura,
        pleura_thickening_source_mask=pleura_thickening_source,
        x_padding_px=PLEURA_THICKENING_X_PADDING_PX,
        side_reference_px=PLEURA_THICKENING_REFERENCE_SIDE_PX,
        local_percentile=PLEURA_THICKENING_LOCAL_PERCENTILE,
        reference_percentile=PLEURA_THICKENING_REFERENCE_PERCENTILE,
        global_fallback_percentile=PLEURA_THICKENING_GLOBAL_FALLBACK_PERCENTILE,
    )

    # Stage 0 = TOP3 dupa stergerea partii comune cu pleura detectata.
    # Stage 1 = TOP3 curatat, pastrat doar sub pleura.
    # Stage 2 = sterge primele randuri/pixeli de sus din fiecare componenta.
    # Stage 3 = sparge bridge-urile subtiri.
    # Stage 4 = elimina componentele insulare aparute/ramase dupa spargeri.
    # Stage 5 = elimina componentele prea mici/prea mari, componentele prea late mediu,
    # componentele fara suport TOP2, componentele sub intreruperi si cele fara ingrosare pleurala adaptiva.
    stage0_mask = top3_without_pleura.copy()
    stage1_mask = top3_under_pleura.copy()
    stage2_mask = top3_top_peeled.copy()
    stage3_mask = top3_bridge_broken.copy()
    stage4_mask = top3_no_islands.copy()
    stage5_mask = top3_after_thickening.copy()

    empty = np.zeros_like(top3, dtype=np.uint8)
    removed_total = np.zeros_like(top3, dtype=np.uint8)
    removed_total[common_with_pleura > 0] = 255
    removed_total[removed_islands > 0] = 255
    removed_total[removed_top_rows > 0] = 255
    removed_total[removed_thin_bridges > 0] = 255
    removed_total[removed_too_small > 0] = 255
    removed_total[removed_too_large > 0] = 255
    removed_total[removed_bad_average_width > 0] = 255
    removed_total[removed_no_top2_support > 0] = 255
    removed_total[removed_above_interruption > 0] = 255
    removed_total[removed_no_thickening > 0] = 255

    nodule_infos = _build_component_infos(stage5_mask)

    stage_masks = {
        0: stage0_mask,
        1: stage1_mask,
        2: stage2_mask,
        3: stage3_mask,
        4: stage4_mask,
        5: stage5_mask,
    }

    stage_titles = {
        0: "STAGE 0: TOP3 fara partea comuna cu pleura",
        1: "STAGE 1: TOP3 fara pleura, doar sub pleura",
        2: "STAGE 2: sterge primii pixeli de sus din componente",
        3: f"STAGE 3: sparge bridge-urile subtiri sub {THIN_BRIDGE_MIN_VERTICAL_THICKNESS_PX} px",
        4: "STAGE 4: elimina insulele dupa spargeri",
        5: "STAGE 5: final - TOP3 confirmat in TOP2 + fara intreruperi + ingrosare",
    }

    stage_debug_images: Dict[int, np.ndarray] = {}
    for stage, mask in stage_masks.items():
        stage_debug_images[stage] = draw_nodule_candidate_debug(
            base_bgr=base,
            pleura_mask=pleura,
            top3_mask=mask,
            title_text=stage_titles[stage],
        )

    nodule_image = draw_nodule_marking(
        base_bgr=base,
        pleura_mask=pleura,
        nodule_mask=stage5_mask,
        title_text="NODULI CANDIDATI DIN TOP3 CONFIRMATI IN TOP2 | fara intreruperi + ingrosare",
    )

    candidate_debug_image = draw_nodule_candidate_debug(
        base_bgr=base,
        pleura_mask=pleura,
        top3_mask=top3_without_pleura,
        title_text="DEBUG TOP3 FARA ZONA COMUNA CU PLEURA",
    )

    result: Dict[str, object] = {
        "nodule_mask": stage5_mask,
        "nodule_core_mask": stage5_mask,
        "nodule_box_mask": stage5_mask,
        "candidate_mask": stage5_mask,
        "rejected_mask": removed_total,
        "under_structure_mask": stage5_mask,
        "under_structure_rejected_mask": removed_total,
        "working_mask": stage5_mask,
        "excluded_mask": removed_total,
        "search_band_mask": search_band,
        "contact_zone_mask": contact_band,
        "interruption_exclusion_mask": empty,
        "nodule_image": nodule_image,
        "candidate_debug_image": candidate_debug_image,
        "nodule_infos": nodule_infos,
        "nodule_count": len(nodule_infos),
        "stage4_removed_thin_bridge_mask": removed_thin_bridges,
        "stage5_removed_small_mask": removed_too_small,
        "stage5_removed_large_mask": removed_too_large,
        "stage5_removed_bad_average_width_mask": removed_bad_average_width,
        "stage5_removed_no_top2_support_mask": removed_no_top2_support,
        "stage5_removed_above_interruption_mask": removed_above_interruption,
        "stage5_removed_no_thickening_mask": removed_no_thickening,
        "interruption_above_zone_mask": interruption_above_zone,
        "top2_turture_support_zone_mask": top2_turture_support_zone,
        "pleura_thickening_band_mask": pleura_thickening_band,
        "pleura_thickening_source_in_band_mask": pleura_source_in_band,
    }

    for stage, mask in stage_masks.items():
        result[f"stage{stage}_mask"] = mask
        if stage == 0:
            result[f"stage{stage}_rejected_mask"] = common_with_pleura.copy()
        elif stage == 1:
            result[f"stage{stage}_rejected_mask"] = common_with_pleura.copy()
        elif stage == 2:
            stage2_removed = np.zeros_like(top3, dtype=np.uint8)
            stage2_removed[common_with_pleura > 0] = 255
            stage2_removed[removed_top_rows > 0] = 255
            result[f"stage{stage}_rejected_mask"] = stage2_removed
        elif stage == 3:
            stage3_removed = np.zeros_like(top3, dtype=np.uint8)
            stage3_removed[common_with_pleura > 0] = 255
            stage3_removed[removed_top_rows > 0] = 255
            stage3_removed[removed_thin_bridges > 0] = 255
            result[f"stage{stage}_rejected_mask"] = stage3_removed
        elif stage == 4:
            stage4_removed = np.zeros_like(top3, dtype=np.uint8)
            stage4_removed[common_with_pleura > 0] = 255
            stage4_removed[removed_top_rows > 0] = 255
            stage4_removed[removed_thin_bridges > 0] = 255
            stage4_removed[removed_islands > 0] = 255
            result[f"stage{stage}_rejected_mask"] = stage4_removed
        elif stage >= 5:
            result[f"stage{stage}_rejected_mask"] = removed_total.copy()
        else:
            result[f"stage{stage}_rejected_mask"] = empty.copy()
        result[f"stage{stage}_debug_image"] = stage_debug_images[stage]

    result["reject_table"] = _build_reject_table_from_infos(nodule_infos)

    return result


# ============================================================
# CSV DEBUG API - compatibil cu main.py
# ============================================================
# main.py importa save_debug si append_reject_summary. In varianta asta
# curata nu exista selectie de noduli, deci CSV-ul noteaza doar componentele
# ramase dupa filtre, ca sa nu crape pipeline-ul.
# ============================================================

REJECT_CSV_COLUMNS = [
    "zone_id",
    "x_start",
    "x_end",
    "width",
    "height",
    "area",
    "accepted",
    "reason",
]

NODULE_CSV_COLUMNS = [
    "index",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "width",
    "height",
    "area",
    "average_width",
    "filter_stage",
    "reason",
]


def _build_reject_table_from_infos(
    nodule_infos: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for info in nodule_infos:
        rows.append(
            {
                "zone_id": int(info.get("index", 0)),
                "x_start": int(info.get("x_min", 0)),
                "x_end": int(info.get("x_max", 0)),
                "width": int(info.get("width", 0)),
                "height": int(info.get("height", 0)),
                "area": int(info.get("area", 0)),
                "accepted": True,
                "reason": str(
                    info.get(
                        "reason",
                        "top2_bridge_broken_no_islands_min_max_avg_width_no_interruption_with_pleural_thickening_component",
                    )
                ),
            }
        )

    return rows


def _write_csv(
    path: str,
    rows: List[Dict[str, object]],
    columns: List[str],
) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_debug(
    debug_images: Dict[str, object],
    out_dir: str,
    nodule_infos: List[Dict[str, object]] | None = None,
    prefix: str = "",
    save_stage_images: bool = True,
) -> Dict[str, object]:
    """Compatibilitate cu main.py.

    In variantele vechi, primul argument era un dictionar de debug.
    In main.py actual, se trimite direct nodule_result. Functia accepta ambele.
    """
    os.makedirs(out_dir, exist_ok=True)
    pre = f"{prefix}_" if prefix else ""
    written: Dict[str, object] = {"stage_images": []}

    data = debug_images

    if save_stage_images:
        for stage in range(0, 6):
            key = f"stage{stage}_debug_image"
            image = data.get(key)
            if isinstance(image, np.ndarray):
                path = os.path.join(out_dir, f"{pre}{key}.png")
                cv2.imwrite(path, image)
                written["stage_images"].append(path)

        for key in ["nodule_image", "candidate_debug_image"]:
            image = data.get(key)
            if isinstance(image, np.ndarray):
                path = os.path.join(out_dir, f"{pre}{key}.png")
                cv2.imwrite(path, image)
                written["stage_images"].append(path)

    infos = nodule_infos
    if infos is None:
        infos = data.get("nodule_infos", [])

    reject_table = data.get("reject_table")
    if reject_table is None:
        reject_table = _build_reject_table_from_infos(infos)

    reject_path = os.path.join(out_dir, f"{pre}reject_table.csv")
    _write_csv(reject_path, reject_table, REJECT_CSV_COLUMNS)
    written["reject_table"] = reject_path

    nodule_path = os.path.join(out_dir, f"{pre}nodule_infos.csv")
    _write_csv(nodule_path, infos, NODULE_CSV_COLUMNS)
    written["nodule_infos"] = nodule_path

    return written


def append_reject_summary(
    csv_path: str,
    debug_images: Dict[str, object],
    prefix: str = "",
) -> str:
    """Adauga reject_table intr-un CSV cumulativ.

    Pentru varianta TOP2, fiecare componenta ramasa este trecuta ca acceptata,
    fiindca momentan nu facem selectie reala de noduli.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    rows = debug_images.get("reject_table")
    if rows is None:
        rows = _build_reject_table_from_infos(debug_images.get("nodule_infos", []))

    columns = ["image"] + REJECT_CSV_COLUMNS
    exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            out = dict(row)
            out["image"] = prefix
            writer.writerow(out)

    return csv_path
