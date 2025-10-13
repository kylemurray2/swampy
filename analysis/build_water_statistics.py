#!/usr/bin/env python3
"""Compute per-pixel statistics and masks from monthly DSWE mosaics.

Outputs include water probability, cloud fraction, valid observation count,
mode class, and masks for persistent water, ephemeral water, stable land,
and ocean/shore buffers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import rasterio
from rasterio.features import rasterize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


WATER_CLASSES = {1, 2, 3, 4}
CLOUD_CLASS = 9
NODATA_CLASS = 255

PRODUCT_FILENAMES = {
    "intr": "mosaic_intr.tif",
    "inwam": "mosaic_inwam.tif",
}

DEFAULT_OCEAN_THRESHOLD = 0.6


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute DSWE monthly statistics and masks")
    parser.add_argument("--config-dir", type=Path, default=Path("."))
    parser.add_argument("--data-dir", type=Path, help="Directory containing mosaics/YYYYMM")
    parser.add_argument("--output-dir", type=Path, help="Directory to write statistics")
    parser.add_argument("--start", help="First month YYYYMM", default=None)
    parser.add_argument("--end", help="Last month YYYYMM", default=None)
    parser.add_argument("--include-inwam", action="store_true", help="Use INWAM rasters where available")
    parser.add_argument("--coast-geojson", type=Path, help="Coastline polygons (EPSG:4326) for ocean mask")
    parser.add_argument("--shore-buffer-km", type=float, default=3.0)
    parser.add_argument("--persistent-threshold", type=float, default=0.95)
    parser.add_argument("--ephemeral-min", type=float, default=0.1)
    parser.add_argument("--stable-land-max", type=float, default=0.05)
    parser.add_argument("--ocean-threshold", type=float, default=DEFAULT_OCEAN_THRESHOLD)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


CONFIG_MODULE = load_config_module()


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    data_dir = None
    output_dir = None
    if CONFIG_MODULE is not None:
        ps = CONFIG_MODULE.getPS(str(args.config_dir))
        if args.data_dir:
            data_dir = Path(args.data_dir).resolve()
        else:
            data_dir = Path(getattr(ps, "dataDir_usgs", "")).resolve()
        if args.output_dir:
            output_dir = Path(args.output_dir).resolve()
        else:
            output_dir = Path(data_dir / "stats").resolve()
    else:
        data_dir = Path(args.data_dir).resolve()
        output_dir = Path(args.output_dir or (data_dir / "stats")).resolve()
    return {"data_dir": data_dir, "output_dir": output_dir}


def list_month_dirs(data_dir: Path, start: Optional[str], end: Optional[str]) -> List[Path]:
    mosaic_root = data_dir / "mosaics"
    months = sorted(p for p in mosaic_root.iterdir() if p.is_dir() and p.name.isdigit())
    if start:
        months = [p for p in months if p.name >= start]
    if end:
        months = [p for p in months if p.name <= end]
    return months


def load_month_raster(month_dir: Path, kind: str) -> Optional[np.ndarray]:
    path = month_dir / PRODUCT_FILENAMES[kind]
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        data = src.read(1)
    return data


def stack_product(month_dirs: List[Path], kind: str) -> Optional[np.ndarray]:
    arrays = []
    template = None
    for month_dir in month_dirs:
        data = load_month_raster(month_dir, kind)
        if data is None:
            arrays.append(None)
        else:
            if template is None:
                template = data
            arrays.append(data)
    if template is None:
        return None
    filled = []
    for arr in arrays:
        if arr is None:
            filled.append(np.full_like(template, NODATA_CLASS))
        else:
            filled.append(arr)
    return np.stack(filled, axis=0)


def stack_intr(month_dirs: List[Path]) -> np.ndarray:
    stack = []
    for month_dir in month_dirs:
        data = load_month_raster(month_dir, "intr")
        if data is not None:
            stack.append(data)
    if not stack:
        raise RuntimeError("No INTR mosaics found")
    return np.stack(stack, axis=0)


def compute_counts(stack: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    valid = stack != NODATA_CLASS
    water = np.isin(stack, list(WATER_CLASSES))
    cloud = stack == CLOUD_CLASS
    if weights is not None:
        w = np.where(weights == NODATA_CLASS, 0.0, weights / 4.0)
        return {
            "valid": np.sum(valid * w, axis=0),
            "water": np.sum(water * w, axis=0),
            "cloud": np.sum(cloud * w, axis=0),
        }
    return {
        "valid": valid.sum(axis=0),
        "water": water.sum(axis=0),
        "cloud": cloud.sum(axis=0),
    }


def compute_probabilities(counts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    valid = counts["valid"].astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        water_prob = np.where(valid > 0, counts["water"] / valid, 0.0)
        cloud_frac = np.where(valid > 0, counts["cloud"] / valid, 0.0)
    return {"water_prob": water_prob, "cloud_frac": cloud_frac}


def compute_mode(stack: np.ndarray) -> np.ndarray:
    height, width = stack.shape[1:]
    mode = np.full((height, width), NODATA_CLASS, dtype=np.uint8)
    for y in range(height):
        column = stack[:, y, :]
        for x in range(width):
            vals = column[:, x]
            vals = vals[vals != NODATA_CLASS]
            if vals.size == 0:
                continue
            unique, counts = np.unique(vals, return_counts=True)
            mode[y, x] = unique[np.argmax(counts)]
    return mode


def derive_masks(probs: Dict[str, np.ndarray], counts: Dict[str, np.ndarray], args: argparse.Namespace) -> Dict[str, np.ndarray]:
    water_prob = probs["water_prob"]
    persistent = (water_prob >= args.persistent_threshold).astype(np.uint8)
    stable_land = (water_prob <= args.stable_land_max).astype(np.uint8)
    ephemeral = ((water_prob >= args.ephemeral_min) & (water_prob < args.persistent_threshold)).astype(np.uint8)
    return {
        "persistent_water_mask": persistent,
        "stable_land_mask": stable_land,
        "ephemeral_water_mask": ephemeral,
    }


def read_template(month_dir: Path) -> rasterio.io.DatasetReader:
    return rasterio.open(month_dir / PRODUCT_FILENAMES["intr"])


def write_raster(output_path: Path, template: rasterio.io.DatasetReader, data: np.ndarray, dtype: str, nodata) -> None:
    profile = template.profile.copy()
    profile.update({
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": dtype,
        "compress": "lzw",
    })
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype(profile["dtype"]), 1)


def load_coast_geometries(geojson_path: Path) -> List[Dict]:
    with open(geojson_path) as f:
        data = json.load(f)
    return [feature["geometry"] for feature in data.get("features", [])]


def rasterize_coast(template: rasterio.io.DatasetReader, coast_geoms: List[Dict]) -> np.ndarray:
    if not coast_geoms:
        return np.zeros((template.height, template.width), dtype=np.uint8)
    transform = template.transform
    return rasterize(
        ((geom, 1) for geom in coast_geoms),
        out_shape=(template.height, template.width),
        transform=transform,
        all_touched=True,
        fill=0,
        dtype=np.uint8,
    )


def create_ocean_masks(template: rasterio.io.DatasetReader, water_prob: np.ndarray, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    ocean_mask = np.zeros_like(water_prob, dtype=np.uint8)
    shore_buffer = np.zeros_like(water_prob, dtype=np.uint8)

    if not args.coast_geojson or not args.coast_geojson.exists():
        return {"ocean_mask": ocean_mask, "shore_buffer": shore_buffer}

    coast_geoms = load_coast_geometries(args.coast_geojson)
    coast_raster = rasterize_coast(template, coast_geoms)
    ocean_mask[(coast_raster == 1) & (water_prob >= args.ocean_threshold)] = 1

    if args.shore_buffer_km > 0:
        # Rough pixel-based buffer using distance transform
        pixel_size = abs(template.transform.a)
        buffer_pixels = int((args.shore_buffer_km * 1000) / pixel_size)
        from scipy.ndimage import distance_transform_edt

        dist = distance_transform_edt(coast_raster == 0)
        shore_buffer[dist <= buffer_pixels] = 1

    return {"ocean_mask": ocean_mask, "shore_buffer": shore_buffer}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args)
    data_dir, output_dir = paths["data_dir"], paths["output_dir"]

    month_dirs = list_month_dirs(data_dir, args.start, args.end)
    if not month_dirs:
        print("No monthly mosaics found.")
        return 1

    stack = stack_intr(month_dirs)
    inwam_stack = stack_product(month_dirs, "inwam") if args.include_inwam else None
    counts = compute_counts(stack, inwam_stack if inwam_stack is not None else None)
    probs = compute_probabilities(counts)
    mode = compute_mode(stack)
    masks = derive_masks(probs, counts, args)

    with read_template(month_dirs[0]) as template:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_raster(output_dir / "water_prob.tif", template, probs["water_prob"], "float32", 0.0)
        write_raster(output_dir / "cloud_frac.tif", template, probs["cloud_frac"], "float32", 0.0)
        write_raster(output_dir / "valid_count.tif", template, counts["valid"].astype(np.float32), "float32", 0.0)
        write_raster(output_dir / "mode_class.tif", template, mode, "uint8", NODATA_CLASS)

        for name, data in masks.items():
            write_raster(output_dir / f"{name}.tif", template, data, "uint8", 0)

        ocean_outputs = create_ocean_masks(template, probs["water_prob"], args)
        for name, data in ocean_outputs.items():
            write_raster(output_dir / f"{name}.tif", template, data, "uint8", 0)

    print(f"Wrote statistics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
