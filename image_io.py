import shutil
from pathlib import Path

import cv2

import config


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_results_dir() -> None:
    if config.RESULTS_DIR.exists():
        shutil.rmtree(config.RESULTS_DIR)

    ensure_dir(config.RESULTS_DIR)
    ensure_dir(config.READ_TEST_DIR)


def prepare_results_dir() -> None:
    if config.RESET_RESULTS_ON_RUN:
        reset_results_dir()
    else:
        ensure_dir(config.RESULTS_DIR)
        ensure_dir(config.READ_TEST_DIR)


def find_image_path(index: int) -> Path | None:
    for extension in config.IMAGE_EXTENSIONS:
        image_path = config.INPUT_DIR / f"{index}{extension}"

        if image_path.exists():
            return image_path

    return None


def get_indices_to_process() -> list[int]:
    if config.RUN_SINGLE_IMAGE:
        image_path = find_image_path(config.SINGLE_IMAGE_IDX)

        if image_path is None:
            return []

        return [config.SINGLE_IMAGE_IDX]

    indices = []

    for index in range(config.START_IDX, config.END_IDX):
        image_path = find_image_path(index)

        if image_path is not None:
            indices.append(index)

    return indices


def read_image_bgr(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Nu pot citi imaginea: {image_path}")

    return image


def save_image(output_path: Path, image) -> None:
    ensure_dir(output_path.parent)

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise ValueError(f"Nu pot salva imaginea: {output_path}")


def make_output_name(index: int, suffix: str, extension: str = ".png") -> str:
    return f"{index:02d}_{suffix}{extension}"