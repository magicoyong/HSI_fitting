from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import scipy.io
import torch
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _normalize_band_to_unit_interval(band: np.ndarray) -> np.ndarray:
    band = band.astype(np.float32)
    band_min = float(band.min())
    band_max = float(band.max())
    if band_max > band_min:
        return (band - band_min) / (band_max - band_min)
    return np.zeros_like(band, dtype=np.float32)


def _normalize_cube(cube_hwc: np.ndarray) -> np.ndarray:
    cube_hwc = np.asarray(cube_hwc, dtype=np.float32)
    normalized = np.empty_like(cube_hwc, dtype=np.float32)
    for channel in range(cube_hwc.shape[-1]):
        normalized[..., channel] = _normalize_band_to_unit_interval(cube_hwc[..., channel])
    return normalized


def _infer_channel_axis(shape: Iterable[int]) -> int:
    dims = list(shape)
    if len(dims) != 3:
        raise ValueError(f"Expected 3D HSI cube, got shape {tuple(dims)}")
    return int(np.argmin(np.asarray(dims)))


def ensure_hwc_cube(array: np.ndarray, channel_axis: Optional[int] = None) -> np.ndarray:
    cube = np.asarray(array)
    if cube.ndim == 4 and cube.shape[0] == 1:
        cube = cube.squeeze(0)

    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D HSI cube, got shape {cube.shape}")

    spectral_axis = _infer_channel_axis(cube.shape) if channel_axis is None else int(channel_axis)
    cube = np.moveaxis(cube, spectral_axis, -1)
    return np.ascontiguousarray(cube.astype(np.float32))


def load_hsi_from_mat(mat_path: Path, mat_key: Optional[str] = None, channel_axis: Optional[int] = None) -> np.ndarray:
    mat = scipy.io.loadmat(str(mat_path))
    candidate_keys = [
        key
        for key, value in mat.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim >= 3
    ]
    if mat_key is None:
        if not candidate_keys:
            raise ValueError(f"No 3D array found in {mat_path}")
        if len(candidate_keys) == 1:
            mat_key = candidate_keys[0]
        else:
            ranked = sorted(candidate_keys, key=lambda key: mat[key].ndim, reverse=True)
            mat_key = ranked[0]

    if mat_key not in mat:
        raise KeyError(f"Key '{mat_key}' not found in {mat_path}")

    cube = ensure_hwc_cube(mat[mat_key], channel_axis=channel_axis)
    return _normalize_cube(cube)


def load_hsi_from_npy(npy_path: Path, channel_axis: Optional[int] = None) -> np.ndarray:
    cube = ensure_hwc_cube(np.load(str(npy_path)), channel_axis=channel_axis)
    return _normalize_cube(cube)


def _band_sort_key(path: Path):
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return (0, int(digits), path.name) if digits else (1, stem.lower(), path.name)


def load_hsi_from_band_dir(directory: Path) -> np.ndarray:
    band_paths = sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=_band_sort_key,
    )
    if not band_paths:
        raise ValueError(f"No band images found in directory: {directory}")

    bands = []
    expected_shape = None
    for band_path in band_paths:
        band = np.asarray(Image.open(band_path))
        if band.ndim == 3:
            band = band[..., 0]
        if band.ndim != 2:
            raise ValueError(f"Expected single-band image at {band_path}, got shape {band.shape}")
        if expected_shape is None:
            expected_shape = band.shape
        elif band.shape != expected_shape:
            raise ValueError(
                f"Band size mismatch in {directory}: expected {expected_shape}, got {band.shape} from {band_path.name}"
            )
        bands.append(_normalize_band_to_unit_interval(band))

    return np.stack(bands, axis=-1).astype(np.float32)


def load_hsi_cube(source: str, mat_key: Optional[str] = None, channel_axis: Optional[int] = None) -> np.ndarray:
    source_path = Path(source).expanduser()
    if source_path.is_dir():
        return load_hsi_from_band_dir(source_path)

    suffix = source_path.suffix.lower()
    if suffix == ".mat":
        return load_hsi_from_mat(source_path, mat_key=mat_key, channel_axis=channel_axis)
    if suffix == ".npy":
        return load_hsi_from_npy(source_path, channel_axis=channel_axis)

    raise ValueError(
        "Unsupported HSI source. Expected a .mat file, .npy file, or a directory of per-band images. "
        f"Got: {source}"
    )


def cube_to_tensor(cube_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(cube_hwc, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    return tensor.contiguous().to(device)


def compute_sam(gt_hwc: np.ndarray, pred_hwc: np.ndarray) -> float:
    dot = np.sum(gt_hwc * pred_hwc, axis=-1)
    gt_norm = np.linalg.norm(gt_hwc, axis=-1)
    pred_norm = np.linalg.norm(pred_hwc, axis=-1)
    cosine = dot / (gt_norm * pred_norm + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.mean(np.arccos(cosine) * 180.0 / np.pi))


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    diff_h = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean()
    diff_w = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w


def spectral_smoothness_loss(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] <= 1:
        return image.new_zeros(())
    return (image[:, 1:, :, :] - image[:, :-1, :, :]).abs().mean()
