import cv2
import numpy as np

PALETTE_METHOD = "clahe_kmeans"

PALETTE_PERCENTILE_LOW = 2
PALETTE_PERCENTILE_HIGH = 99

PALETTE_CLAHE_CLIP_LIMIT = 2.0
PALETTE_CLAHE_TILE_SIZE = 8

PALETTE_KMEANS_ATTEMPTS = 3
PALETTE_KMEANS_MAX_ITER = 35

PALETTE_VALID_LOW_PERCENTILE = 1
PALETTE_VALID_HIGH_PERCENTILE = 99.7

BINARY_KEEP_TOP1_LEVELS = 1
BINARY_KEEP_TOP2_LEVELS = 2
BINARY_KEEP_TOP3_LEVELS = 3
BINARY_KEEP_TOP4_LEVELS = 4


def to_gray(image_bgr):
    if image_bgr.ndim == 2:
        return image_bgr.copy()

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def reduce_palette_7_percentile_linear(gray, colors=7):
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray = gray.astype(np.uint8)

    low = np.percentile(gray, PALETTE_PERCENTILE_LOW)
    high = np.percentile(gray, PALETTE_PERCENTILE_HIGH)

    if high <= low:
        return gray.copy()

    normalized = np.clip(
        (gray.astype(np.float32) - low) / (high - low),
        0,
        1,
    )

    quantized = np.floor(normalized * colors)
    quantized[quantized >= colors] = colors - 1

    return (quantized / (colors - 1) * 255).astype(np.uint8)


def reduce_palette_7_clahe_kmeans(gray, colors=7):
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray = gray.astype(np.uint8)
    tile_size = max(2, int(PALETTE_CLAHE_TILE_SIZE))

    clahe = cv2.createCLAHE(
        clipLimit=float(PALETTE_CLAHE_CLIP_LIMIT),
        tileGridSize=(tile_size, tile_size),
    )

    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    valid_low = np.percentile(enhanced, PALETTE_VALID_LOW_PERCENTILE)
    valid_high = np.percentile(enhanced, PALETTE_VALID_HIGH_PERCENTILE)

    valid_mask = (enhanced >= valid_low) & (enhanced <= valid_high)
    samples = enhanced[valid_mask].reshape(-1, 1).astype(np.float32)

    if samples.shape[0] < colors * 20:
        return reduce_palette_7_percentile_linear(gray, colors=colors)

    cv2.setRNGSeed(12345)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        int(PALETTE_KMEANS_MAX_ITER),
        0.4,
    )

    _, _, centers = cv2.kmeans(
        samples,
        int(colors),
        None,
        criteria,
        int(PALETTE_KMEANS_ATTEMPTS),
        cv2.KMEANS_PP_CENTERS,
    )

    centers = centers.reshape(-1).astype(np.float32)
    centers_sorted = centers[np.argsort(centers)]

    output_levels = np.linspace(0, 255, int(colors)).astype(np.uint8)

    flat = enhanced.reshape(-1).astype(np.float32)
    distances = np.abs(flat[:, None] - centers_sorted[None, :])
    nearest = np.argmin(distances, axis=1)

    result = output_levels[nearest].reshape(enhanced.shape).astype(np.uint8)

    dark_cut = max(3, np.percentile(gray, 0.5))
    result[gray <= dark_cut] = 0

    return result


def reduce_palette_7(gray, colors=7):
    if PALETTE_METHOD == "clahe_kmeans":
        return reduce_palette_7_clahe_kmeans(gray, colors=colors)

    return reduce_palette_7_percentile_linear(gray, colors=colors)


def binarize_palette_7(palette_gray, keep_top_levels):
    if palette_gray.ndim == 3:
        palette_gray = cv2.cvtColor(palette_gray, cv2.COLOR_BGR2GRAY)

    palette_gray = palette_gray.astype(np.uint8)
    values = np.sort(np.unique(palette_gray))

    if len(values) < 2:
        binary = np.zeros_like(palette_gray, dtype=np.uint8)
        return binary, 0

    keep_top_levels = max(1, min(int(keep_top_levels), len(values)))
    threshold = int(values[-keep_top_levels])

    binary = (palette_gray >= threshold).astype(np.uint8) * 255

    return binary, threshold


def preprocess_crop(crop_bgr):
    gray = to_gray(crop_bgr)
    palette_7 = reduce_palette_7(gray, colors=7)

    binary_top1, threshold_top1 = binarize_palette_7(
        palette_7,
        keep_top_levels=BINARY_KEEP_TOP1_LEVELS,
    )

    binary_top2, threshold_top2 = binarize_palette_7(
        palette_7,
        keep_top_levels=BINARY_KEEP_TOP2_LEVELS,
    )

    binary_top3, threshold_top3 = binarize_palette_7(
        palette_7,
        keep_top_levels=BINARY_KEEP_TOP3_LEVELS,
    )

    binary_top4, threshold_top4 = binarize_palette_7(
        palette_7,
        keep_top_levels=BINARY_KEEP_TOP4_LEVELS,
    )

    return {
        "gray": gray,
        "palette_7": palette_7,
        "binary_top1": binary_top1,
        "binary_top2": binary_top2,
        "binary_top3": binary_top3,
        "binary_top4": binary_top4,
        "threshold_top1": threshold_top1,
        "threshold_top2": threshold_top2,
        "threshold_top3": threshold_top3,
        "threshold_top4": threshold_top4,
    }
