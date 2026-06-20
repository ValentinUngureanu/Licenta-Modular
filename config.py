from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "ORIGINAL_IMAGES"
RESULTS_DIR = BASE_DIR / "RESULTS"

READ_TEST_DIR = RESULTS_DIR / "00_READ_TEST"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

RESET_RESULTS_ON_RUN = True

RUN_SINGLE_IMAGE = False
SINGLE_IMAGE_IDX = 15

START_IDX = 0
END_IDX = 61