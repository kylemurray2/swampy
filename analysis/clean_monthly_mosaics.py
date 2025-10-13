#!/usr/bin/env python3
"""Clean monthly DSWE mosaics using statistics-derived masks and heuristics."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import rasterio
from scipy.ndimage import label

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config_module() -> Optional[object]:
    config_path = REPO_ROOT / "config.py"
    if not config_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("swampy_config", config_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG_MODULE = load_config_module()

WATER_CLASSES = np.array([1, 2, 3, 4], dtype=np.uint8)
CLOUD_CLASS = 9
NODATA_CLASS = 255

CONFIDENCE_CODES = {
    "unchanged": 0,
    "ocean": 1,
    "persistent_water": 2,
    "stable_land": 3,
    "temporal_fill": 4,
    "fallback_mode": 5,
}

PRODUCT_FILENAMES = {
    "intr": "mosaic_intr.tif",
    "inwam": "mosaic_inwam.tif",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean monthly DSWE INTR mosaics")
    parser.add_argument("--config-dir", type=Path, default=Path("."))
    parser.add_argument("--data-dir", type=Path, help="Directory with mosaics/YYYYMM")
    parser.add_argument("--stats-dir", type=Path, help="Directory containing statistics rasters")
    parser.add_argument("--output-dir", type=Path, help="Directory for cleaned mosaics")
    parser.add_argument("--start", help="First month YYYYMM", default=None)
    parser.add_argument("--end", help="Last month YYYYMM", default=None)
    parser.add_argument("--max-fill-months", type=int, default=3, help="Search radius for temporal fill")
    parser.add_argument("--min-water-pixels", type=int, default=9, help="Minimum water component size (pixels)")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    if CONFIG_MODULE is not None:
        ps = CONFIG_MODULE.getPS(str(args.config_dir))
        data_dir = Path(args.data_dir or getattr(ps, "dataDir_usgs", "")).resolve()
        stats_dir = Path(args.stats_dir or (data_dir / "stats")).resolve()
        output_dir = Path(args.output_dir or (data_dir / "cleaned")).resolve()
    else:
        data_dir = Path(args.data_dir).resolve()
        stats_dir = Path(args.stats_dir).resolve()
        output_dir = Path(args.output_dir).resolve()
    return {"data_dir": data_dir, "stats_dir": stats_dir, "output_dir": output_dir}


def list_month_dirs(data_dir: Path, start: Optional[str], end: Optional[str]) -> List[Path]:
    mosaic_root = data_dir / "mosaics"
    months = sorted(p for p in mosaic_root.iterdir() if p.is_dir() and p.name.isdigit())
    if start:
        months = [p for p in months if p.name >= start]
    if end:
        months = [p for p in months if p.name <= end]
    return months


def load_mask(stats_dir: Path, name: str, dtype=np.uint8) -> np.ndarray:
    path = stats_dir / f"{name}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Required statistics raster not found: {path}")
    with rasterio.open(path) as src:
        return src.read(1).astype(dtype)


def load_optional_mask(stats_dir: Path, name: str, dtype=np.uint8) -> Optional[np.ndarray]:
    path = stats_dir / f"{name}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1).astype(dtype)


def load_mode(stats_dir: Path) -> np.ndarray:
    return load_mask(stats_dir, "mode_class", dtype=np.uint8)


def load_stack(month_dirs: List[Path]) -> np.ndarray:
    stack = []
    for month_dir in month_dirs:
        path = month_dir / PRODUCT_FILENAMES["intr"]
        if not path.exists():
            raise FileNotFoundError(path)
        with rasterio.open(path) as src:
            stack.append(src.read(1))
    return np.stack(stack, axis=0)


def load_inwam_stack(month_dirs: List[Path]) -> Optional[np.ndarray]:
    arrays = []
    available = False
    for month_dir in month_dirs:
        path = month_dir / PRODUCT_FILENAMES["inwam"]
        if path.exists():
            available = True
            with rasterio.open(path) as src:
                arrays.append(src.read(1))
        else:
            arrays.append(None)
    if not available:
        return None
    template = next(arr for arr in arrays if arr is not None)
    filled = []
    for arr in arrays:
        if arr is None:
            filled.append(np.full_like(template, NODATA_CLASS))
        else:
            filled.append(arr)
    return np.stack(filled, axis=0)


def apply_masks(cleaned: np.ndarray, confidence: np.ndarray, masks: Dict[str, np.ndarray]) -> None:
    ocean_mask = masks.get("ocean_mask")
    if ocean_mask is not None:
        mask = ocean_mask.astype(bool)
        change = mask & ~np.isin(cleaned, WATER_CLASSES)
        cleaned[mask] = 2
        confidence[change] = np.maximum(confidence[change], CONFIDENCE_CODES["ocean"])

    persistent = masks.get("persistent_water_mask")
    if persistent is not None:
        mask = persistent.astype(bool)
        change = mask & ~np.isin(cleaned, WATER_CLASSES)
        cleaned[mask] = 2
        confidence[change] = np.maximum(confidence[change], CONFIDENCE_CODES["persistent_water"])

    stable_land = masks.get("stable_land_mask")
    if stable_land is not None:
        mask = stable_land.astype(bool)
        water_mask = np.isin(cleaned, WATER_CLASSES)
        change = mask & water_mask
        cleaned[change] = 0
        confidence[change] = np.maximum(confidence[change], CONFIDENCE_CODES["stable_land"])


def temporal_fill(index: int, stack: np.ndarray, inwam_stack: Optional[np.ndarray], cleaned: np.ndarray, confidence: np.ndarray, mode_class: np.ndarray, args: argparse.Namespace) -> None:
    cloud_mask = (cleaned == CLOUD_CLASS) | (cleaned == NODATA_CLASS)
    if not cloud_mask.any():
        return

    n_months = stack.shape[0]
    fill_mask = cloud_mask.copy()
    for offset in range(1, args.max_fill_months + 1):
        if not fill_mask.any():
            break
        for neighbor_idx in (index - offset, index + offset):
            if neighbor_idx < 0 or neighbor_idx >= n_months:
                continue
            neighbor = stack[neighbor_idx]
            valid_neighbor = (neighbor != CLOUD_CLASS) & (neighbor != NODATA_CLASS)
            if inwam_stack is not None:
                inwam = inwam_stack[neighbor_idx]
                high_conf = np.isin(inwam, [1, 2])
                valid_neighbor &= high_conf
            replace = fill_mask & valid_neighbor
            cleaned[replace] = neighbor[replace]
            confidence[replace] = np.maximum(confidence[replace], CONFIDENCE_CODES["temporal_fill"])
            fill_mask[replace] = False

    if fill_mask.any():
        cleaned[fill_mask] = mode_class[fill_mask]
        confidence[fill_mask] = np.maximum(confidence[fill_mask], CONFIDENCE_CODES["fallback_mode"])


def remove_small_water(cleaned: np.ndarray, min_pixels: int) -> None:
    water_mask = np.isin(cleaned, WATER_CLASSES)
    structure = np.ones((3, 3), dtype=bool)
    labeled, num = label(water_mask, structure)
    if num == 0:
        return
    counts = np.bincount(labeled.ravel())
    small = counts < min_pixels
    small[0] = False
    remove_mask = small[labeled]
    cleaned[remove_mask] = 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args)
    data_dir = paths["data_dir"]
    stats_dir = paths["stats_dir"]
    output_dir = paths["output_dir"]

    month_dirs = list_month_dirs(data_dir, args.start, args.end)
    if not month_dirs:
        print("No monthly mosaics found.")
        return 1

    stack = load_stack(month_dirs)
    inwam_stack = load_inwam_stack(month_dirs)

    masks = {
        "persistent_water_mask": load_mask(stats_dir, "persistent_water_mask"),
        "stable_land_mask": load_mask(stats_dir, "stable_land_mask"),
        "ephemeral_water_mask": load_optional_mask(stats_dir, "ephemeral_water_mask"),
        "ocean_mask": load_optional_mask(stats_dir, "ocean_mask"),
    }
    mode_class = load_mode(stats_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, month_dir in enumerate(month_dirs):
        original = stack[idx].copy()
        cleaned = original.copy()
        confidence = np.zeros_like(cleaned, dtype=np.uint8)

        apply_masks(cleaned, confidence, masks)
        temporal_fill(idx, stack, inwam_stack, cleaned, confidence, mode_class, args)

        remove_small_water(cleaned, args.min_water_pixels)

        month_output = output_dir / month_dir.name
        month_output.mkdir(parents=True, exist_ok=True)

        with rasterio.open(month_dir / PRODUCT_FILENAMES["intr"]) as src:
            profile = src.profile.copy()
            profile.update({"compress": "lzw", "dtype": "uint8"})
            with rasterio.open(month_output / "mosaic_intr_clean.tif", "w", **profile) as dst:
                dst.write(cleaned, 1)

            conf_profile = profile.copy()
            with rasterio.open(month_output / "mosaic_intr_confidence.tif", "w", **conf_profile) as dst_conf:
                dst_conf.write(confidence, 1)

        if args.debug:
            print(f"Cleaned month {month_dir.name} -> {month_output}")

    print(f"Cleaned mosaics written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
