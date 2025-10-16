#!/usr/bin/env python3
"""Compute per-pixel statistics and masks from daily stitched DSWE mosaics.

This script reads raw daily mosaics from stitched_dates/ to compute unbiased
statistics that can guide downstream monthly aggregation and cleaning.

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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

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
CLOUD_CLASS = 4
NODATA_CLASS = 255

PRODUCT_FILENAMES = {
    "intr": "mosaic_intr.tif",
    "inwam": "mosaic_inwam.tif",
}

DEFAULT_OCEAN_THRESHOLD = 0.6


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute DSWE statistics from daily stitched mosaics")
    parser.add_argument("--config-dir", type=Path, default=Path("."))
    parser.add_argument("--data-dir", type=Path, help="Directory containing stitched_dates/YYYYMMDD")
    parser.add_argument("--output-dir", type=Path, help="Directory to write statistics")
    parser.add_argument("--start-date", help="First date YYYYMMDD", default=None)
    parser.add_argument("--end-date", help="Last date YYYYMMDD", default=None)
    parser.add_argument("--include-inwam", action="store_true", help="Use INWAM rasters where available")
    parser.add_argument("--coast-geojson", type=Path, help="Coastline polygons (EPSG:4326) for ocean mask")
    parser.add_argument("--shore-buffer-km", type=float, default=3.0)
    parser.add_argument("--persistent-threshold", type=float, default=0.81)
    parser.add_argument("--ephemeral-min", type=float, default=0.1)
    parser.add_argument("--stable-land-max", type=float, default=0.4)
    parser.add_argument("--ocean-threshold", type=float, default=DEFAULT_OCEAN_THRESHOLD)
    parser.add_argument("--workers", type=int, default=19, help="Parallel I/O workers")
    parser.add_argument("--chunk-size", type=int, default=100, help="Process dates in chunks of this size")
    parser.add_argument("--verbose", action="store_true", help="Print progress information")
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


def list_date_dirs(data_dir: Path, start: Optional[str], end: Optional[str]) -> List[Path]:
    stitched_root = data_dir / "stitched_dates"
    if not stitched_root.exists():
        return []
    dates = sorted(p for p in stitched_root.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 8)
    if start:
        dates = [p for p in dates if p.name >= start]
    if end:
        dates = [p for p in dates if p.name <= end]
    return dates


def load_and_align_raster(
    path: Path,
    reference_profile: Dict,
    kind: str,
) -> Optional[np.ndarray]:
    """Load a raster and align it to the reference grid if needed."""
    if not path.exists():
        return None
    
    try:
        with rasterio.open(path) as src:
            needs_warp = (
                src.width != reference_profile["width"]
                or src.height != reference_profile["height"]
                or src.transform != reference_profile["transform"]
                or src.crs != reference_profile.get("crs")
            )
            
            if needs_warp:
                with WarpedVRT(
                    src,
                    crs=reference_profile.get("crs"),
                    transform=reference_profile["transform"],
                    width=int(reference_profile["width"]),
                    height=int(reference_profile["height"]),
                    resampling=Resampling.nearest,
                ) as vrt:
                    return vrt.read(1).astype(np.uint8)
            else:
                return src.read(1).astype(np.uint8)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None


def establish_reference_grid(date_dirs: List[Path], log_fn=lambda msg: None) -> Dict:
    """Determine a reference grid from available INTR rasters.

    Selects the smallest-area grid (width*height) to honor cropped mosaics
    while logging any mismatched grids for transparency.
    """
    candidates: List[Tuple[int, Path, Dict]] = []
    grid_summary: Dict[Tuple[int, int], int] = {}

    for date_dir in date_dirs:
        intr_path = date_dir / PRODUCT_FILENAMES["intr"]
        if not intr_path.exists():
            continue
        try:
            with rasterio.open(intr_path) as src:
                width = src.width
                height = src.height
                area = width * height
                profile = src.profile.copy()
                candidates.append(
                    (
                        area,
                        date_dir,
                        {
                            "width": width,
                            "height": height,
                            "transform": src.transform,
                            "crs": src.crs,
                            "profile": profile,
                        },
                    )
                )
                grid_summary[(width, height)] = grid_summary.get((width, height), 0) + 1
        except Exception as exc:
            log_fn(f"Warning: failed to inspect {intr_path}: {exc}")

    if not candidates:
        raise RuntimeError("No valid INTR rasters found to establish reference grid")

    if len(grid_summary) > 1:
        log_fn(
            "Detected multiple grid sizes in stitched data: "
            + ", ".join(
                f"{w}x{h} ({count} dates)" for (w, h), count in sorted(grid_summary.items())
            )
        )

    candidates.sort(key=lambda item: item[0])  # smallest area first
    _, ref_dir, ref_profile = candidates[0]
    log_fn(
        f"Using {ref_dir.name} as reference grid ({ref_profile['width']}x{ref_profile['height']})"
    )
    return ref_profile


def load_date_chunk(
    date_dirs: List[Path],
    reference_profile: Dict,
    kind: str,
    workers: int,
) -> np.ndarray:
    """Load a chunk of dates in parallel and return stacked array."""
    height = int(reference_profile["height"])
    width = int(reference_profile["width"])
    stack = np.full((len(date_dirs), height, width), NODATA_CLASS, dtype=np.uint8)
    
    def load_one(idx: int, date_dir: Path) -> tuple[int, Optional[np.ndarray]]:
        path = date_dir / PRODUCT_FILENAMES[kind]
        data = load_and_align_raster(path, reference_profile, kind)
        return idx, data
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(load_one, idx, date_dir): idx
            for idx, date_dir in enumerate(date_dirs)
        }
        
        for future in as_completed(futures):
            idx, data = future.result()
            if data is not None:
                stack[idx] = data
    
    return stack


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
    mask_valid = stack != NODATA_CLASS
    default_result = np.full(stack.shape[1:], NODATA_CLASS, dtype=np.uint8)

    if not np.any(mask_valid):
        return default_result

    modes = default_result.copy()
    current_max = np.zeros(stack.shape[1:], dtype=np.int32)

    for value in range(256):
        mask_value = (stack == value) & mask_valid
        if not np.any(mask_value):
            continue
        count_value = mask_value.sum(axis=0)
        update_mask = count_value > current_max
        modes[update_mask] = value
        current_max[update_mask] = count_value[update_mask]

    modes[~mask_valid.any(axis=0)] = NODATA_CLASS
    return modes


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


def read_template(reference_profile: Dict) -> rasterio.io.DatasetReader:
    """Create a memory dataset from the reference profile for writing outputs."""
    from rasterio.io import MemoryFile
    with MemoryFile() as memfile:
        with memfile.open(**reference_profile) as dataset:
            return dataset


def write_raster(output_path: Path, reference_profile: Dict, data: np.ndarray, dtype: str, nodata) -> None:
    profile = reference_profile.copy()
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


def rasterize_coast(reference_profile: Dict, coast_geoms: List[Dict]) -> np.ndarray:
    height = int(reference_profile["height"])
    width = int(reference_profile["width"])
    if not coast_geoms:
        return np.zeros((height, width), dtype=np.uint8)
    transform = reference_profile["transform"]
    return rasterize(
        ((geom, 1) for geom in coast_geoms),
        out_shape=(height, width),
        transform=transform,
        all_touched=True,
        fill=0,
        dtype=np.uint8,
    )


def create_ocean_masks(reference_profile: Dict, water_prob: np.ndarray, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    ocean_mask = np.zeros_like(water_prob, dtype=np.uint8)
    shore_buffer = np.zeros_like(water_prob, dtype=np.uint8)

    if not args.coast_geojson or not args.coast_geojson.exists():
        return {"ocean_mask": ocean_mask, "shore_buffer": shore_buffer}

    coast_geoms = load_coast_geometries(args.coast_geojson)
    coast_raster = rasterize_coast(reference_profile, coast_geoms)
    ocean_mask[(coast_raster == 1) & (water_prob >= args.ocean_threshold)] = 1

    if args.shore_buffer_km > 0:
        # Rough pixel-based buffer using distance transform
        pixel_size = abs(reference_profile["transform"].a)
        buffer_pixels = int((args.shore_buffer_km * 1000) / pixel_size)
        from scipy.ndimage import distance_transform_edt

        dist = distance_transform_edt(coast_raster == 0)
        shore_buffer[dist <= buffer_pixels] = 1

    return {"ocean_mask": ocean_mask, "shore_buffer": shore_buffer}


def main(argv: Optional[Sequence[str]] = None) -> int:
    start_time = perf_counter()
    args = parse_args(argv)
    paths = resolve_paths(args)
    data_dir, output_dir = paths["data_dir"], paths["output_dir"]

    def log(message: str) -> None:
        if args.verbose or args.debug:
            print(message)

    log(f"Scanning daily stitched mosaics under {data_dir}/stitched_dates")
    date_dirs = list_date_dirs(data_dir, args.start_date, args.end_date)
    if not date_dirs:
        print("No daily stitched mosaics found.")
        return 1

    log(f"Found {len(date_dirs)} date(s). Establishing reference grid...")
    reference_profile = establish_reference_grid(date_dirs, log_fn=log)
    height = int(reference_profile["height"])
    width = int(reference_profile["width"])
    
    log(f"Grid: {width}x{height} pixels")
    
    # Initialize accumulators
    valid_count = np.zeros((height, width), dtype=np.int32)
    water_count = np.zeros((height, width), dtype=np.int32)
    cloud_count = np.zeros((height, width), dtype=np.int32)
    class_counts = {cls: np.zeros((height, width), dtype=np.int32) for cls in range(256)}
    
    # Process in chunks to manage memory
    chunk_size = args.chunk_size
    n_chunks = (len(date_dirs) + chunk_size - 1) // chunk_size
    
    log(f"Processing {n_chunks} chunk(s) of up to {chunk_size} dates each...")
    
    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, len(date_dirs))
        chunk_dates = date_dirs[chunk_start:chunk_end]
        
        log(f"  Chunk {chunk_idx + 1}/{n_chunks}: loading {len(chunk_dates)} dates...")
        chunk_load_start = perf_counter()
        
        intr_chunk = load_date_chunk(chunk_dates, reference_profile, "intr", args.workers)
        inwam_chunk = None
        if args.include_inwam:
            inwam_chunk = load_date_chunk(chunk_dates, reference_profile, "inwam", args.workers)
        
        log(f"    Loaded in {perf_counter() - chunk_load_start:.2f}s, computing stats...")
        
        # Accumulate counts
        valid_mask = intr_chunk != NODATA_CLASS
        water_mask = np.isin(intr_chunk, list(WATER_CLASSES))
        cloud_mask = intr_chunk == CLOUD_CLASS
        
        valid_count += valid_mask.sum(axis=0)
        water_count += water_mask.sum(axis=0)
        cloud_count += cloud_mask.sum(axis=0)
        
        # Accumulate class counts for mode
        for value in range(256):
            value_mask = (intr_chunk == value) & valid_mask
            if np.any(value_mask):
                class_counts[value] += value_mask.sum(axis=0)
        
        log(f"    Chunk {chunk_idx + 1} complete")
    
    log("Computing final statistics...")
    
    # Compute probabilities
    valid_count_float = valid_count.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        water_prob = np.where(valid_count_float > 0, water_count / valid_count_float, 0.0)
        cloud_frac = np.where(valid_count_float > 0, cloud_count / valid_count_float, 0.0)
    
    # Compute mode
    log("Computing mode class...")
    mode_start = perf_counter()
    mode = np.full((height, width), NODATA_CLASS, dtype=np.uint8)
    max_count = np.zeros((height, width), dtype=np.int32)
    for value in range(256):
        if value == NODATA_CLASS:
            continue
        count_arr = class_counts[value]
        if not np.any(count_arr):
            continue
        update_mask = count_arr > max_count
        mode[update_mask] = value
        max_count[update_mask] = count_arr[update_mask]
    log(f"Mode computed in {perf_counter() - mode_start:.2f}s")
    
    # Derive masks
    probs = {"water_prob": water_prob, "cloud_frac": cloud_frac}
    counts_dict = {"valid": valid_count, "water": water_count, "cloud": cloud_count}
    masks = derive_masks(probs, counts_dict, args)
    
    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    log("Writing output rasters...")
    
    write_raster(output_dir / "water_prob.tif", reference_profile, water_prob, "float32", None)
    write_raster(output_dir / "cloud_frac.tif", reference_profile, cloud_frac, "float32", None)
    write_raster(output_dir / "valid_count.tif", reference_profile, valid_count.astype(np.float32), "float32", None)
    write_raster(output_dir / "mode_class.tif", reference_profile, mode, "uint8", NODATA_CLASS)

    for name, data in masks.items():
        write_raster(output_dir / f"{name}.tif", reference_profile, data.astype(np.uint8), "uint8", None)

    ocean_outputs = create_ocean_masks(reference_profile, water_prob, args)
    for name, data in ocean_outputs.items():
        write_raster(output_dir / f"{name}.tif", reference_profile, data.astype(np.uint8), "uint8", None)
    
    elapsed = perf_counter() - start_time
    print(f"Wrote statistics to {output_dir} in {elapsed:.2f}s")
    print(f"  Processed {len(date_dirs)} daily observations")
    print(f"  Water probability range: [{water_prob.min():.3f}, {water_prob.max():.3f}]")
    print(f"  Persistent water pixels: {masks['persistent_water_mask'].sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
