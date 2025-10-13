#!/usr/bin/env python3
"""Utility for stitching USGS DSWE products into date mosaics.

The script scans a data directory (defaults to ``dataDir_usgs`` from ``params.yaml``)
for scene folders named like ``LT05_CU_002009_19930213_20210424_02_DSWE``.  It
collects ``*_INTR.TIF`` rasters, groups them by acquisition date, and builds a
mosaic for each date.  Raw scene folders are relocated into ``<dataDir_usgs>/raw`` and stitched
outputs are written to ``<dataDir_usgs>/stitched_dates/<YYYYMMDD>`` in a layout
similar to ``stitch_dsw.py`` for OPERA data.

Example:
    python -m usgs.stitch_dsw_usgs --year 2003 --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from fractions import Fraction

try:  # Optional dependency that ships with the project
    import config  # type: ignore
except ImportError:  # pragma: no cover - config is part of the repo
    config = None

RAW_SCENE_PATTERN = re.compile(r"_DSWE$", re.IGNORECASE)


@dataclass(frozen=True)
class ProductConfig:
    key: str
    pattern: str
    output_name: str
    priority: Tuple[int, ...]
    nodata: int = 255


@dataclass
class ProcessOutcome:
    status: str
    output: Optional[str]
    reason: Optional[str] = None
    debug: Optional[str] = None


PRODUCTS: Dict[str, ProductConfig] = {
    "INTR": ProductConfig(
        key="INTR",
        pattern="*_INTR.TIF",
        output_name="mosaic_intr.tif",
        priority=(1, 2, 3, 4, 0, 9),
    ),
    "INWAM": ProductConfig(
        key="INWAM",
        pattern="*_INWAM.TIF",
        output_name="mosaic_inwam.tif",
        priority=(1, 2, 3, 4, 0, 9),
    ),
}

LANDSAT7_SLC_OFF_DATE = "20030531"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch USGS DSWE scenes by date")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override data directory (defaults to dataDir_usgs from params.yaml)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("."),
        help="Directory that holds params.yaml (default: current directory)",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        choices=sorted(PRODUCTS.keys()),
        default=["INTR", "INWAM"],
        help="Product keys to stitch (default: INTR INWAM)",
    )
    parser.add_argument("--date", dest="dates", nargs="+", help="Process specific YYYYMMDD date(s)")
    parser.add_argument("--year", type=int, help="Only process dates within this calendar year")
    parser.add_argument("--start-date", help="Process dates >= YYYYMMDD")
    parser.add_argument("--end-date", help="Process dates <= YYYYMMDD")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mosaics instead of skipping",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print unique value counts for each output mosaic",
    )
    try:
        default_workers = max(1, mp.cpu_count() - 1)
    except NotImplementedError:  # pragma: no cover - unusual platforms
        default_workers = 1
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Maximum parallel workers (default: cpu_count - 1)",
    )
    return parser.parse_args(argv)


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        return args.data_dir.expanduser().resolve()

    if config is not None:
        try:
            ps = config.getPS(str(args.config_dir))
            data_dir = getattr(ps, "dataDir_usgs", None)
            if data_dir:
                return Path(data_dir).expanduser().resolve()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: could not load dataDir_usgs from config ({exc})")

    # Fallback to repository default
    return (Path(args.config_dir) / "data" / "usgs_dsw").resolve()


DATE_REGEX = re.compile(r"_(\d{8})_")


def extract_date(identifier: str) -> Optional[str]:
    match = DATE_REGEX.search(identifier)
    if match:
        return match.group(1)
    return None


def extract_satellite_number(identifier: str) -> Optional[int]:
    try:
        root = identifier.split("_")[0]
    except Exception:
        return None

    digits = "".join(ch for ch in root if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            return None
    return None


def to_float(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float(Fraction(value))
    return float(value)


def collect_product_files(data_root: Path, product: ProductConfig, skip_log: Dict[str, int]) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {}

    if not data_root.exists():
        return grouped

    for scene_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        date = extract_date(scene_dir.name)
        if not date:
            continue
            
        satellite = extract_satellite_number(scene_dir.name)
        if satellite == 7 and date >= LANDSAT7_SLC_OFF_DATE:
            skip_log["landsat7_slc_off"] = skip_log.get("landsat7_slc_off", 0) + 1
            continue

        candidates = sorted(scene_dir.glob(product.pattern))
        if not candidates:
            candidates = sorted(scene_dir.glob(product.pattern.lower()))
        if not candidates:
            continue
        
        grouped.setdefault(date, []).extend(candidates)

    return grouped


def determine_bounds_and_resolution(sources: Sequence[rasterio.DatasetReader]) -> Tuple[Tuple[float, float, float, float], Tuple[float, float]]:
    bounds: Optional[Tuple[float, float, float, float]] = None
    res_x, res_y = None, None

    for src in sources:
        left = to_float(src.bounds.left)
        bottom = to_float(src.bounds.bottom)
        right = to_float(src.bounds.right)
        top = to_float(src.bounds.top)
        if bounds is None:
            bounds = (left, bottom, right, top)
        else:
            bounds = (
                min(bounds[0], left),
                min(bounds[1], bottom),
                max(bounds[2], right),
                max(bounds[3], top),
            )

        sx, sy = src.res
        sx = to_float(sx)
        sy = to_float(sy)
        res_x = sx if res_x is None else min(res_x, sx)
        res_y = sy if res_y is None else min(res_y, sy)

    if bounds is None or res_x is None or res_y is None:
        raise ValueError("Could not determine mosaic extent or resolution from inputs")

    return bounds, (res_x, res_y)


def merge_arrays(dest: np.ndarray, new_data: np.ndarray, priority: Tuple[int, ...], nodata: int) -> None:
    if priority:
        higher_values = set()
        for value in priority:
            if value == nodata:
                higher_values.add(value)
                continue

            mask = new_data == value
            if higher_values:
                mask &= ~np.isin(dest, list(higher_values))

            if mask.any():
                dest[mask] = value
            higher_values.add(value)

        # Any remaining values not in priority list: fill where dest still nodata
        other_mask = (new_data != nodata) & ~np.isin(new_data, priority)
        if other_mask.any():
            update_mask = other_mask & (dest == nodata)
            if update_mask.any():
                dest[update_mask] = new_data[update_mask]
    else:
        update_mask = (new_data != nodata) & (dest == nodata)
        if update_mask.any():
            dest[update_mask] = new_data[update_mask]


def custom_merge(
    file_list: Sequence[Path], nodata: int, priority: Tuple[int, ...]
) -> Tuple[np.ndarray, rasterio.Affine]:
    if not file_list:
        raise ValueError("custom_merge received an empty file list")

    paths = [Path(p) for p in file_list]

    bounds: Optional[Tuple[float, float, float, float]] = None
    res_x: Optional[float] = None
    res_y: Optional[float] = None
    dtype_name: Optional[str] = None
    dest_crs = None

    for path in paths:
        with rasterio.open(path) as src:
            if dest_crs is None:
                dest_crs = src.crs
            elif src.crs != dest_crs:
                raise ValueError("All inputs must share the same CRS for mosaic creation")

            dtype_name = dtype_name or src.dtypes[0]

            left = to_float(src.bounds.left)
            bottom = to_float(src.bounds.bottom)
            right = to_float(src.bounds.right)
            top = to_float(src.bounds.top)
            if bounds is None:
                bounds = (left, bottom, right, top)
            else:
                bounds = (
                    min(bounds[0], left),
                    min(bounds[1], bottom),
                    max(bounds[2], right),
                    max(bounds[3], top),
                )

            sx, sy = src.res
            sx = to_float(sx)
            sy = to_float(sy)
            res_x = sx if res_x is None else min(res_x, sx)
            res_y = sy if res_y is None else min(res_y, sy)

    if bounds is None or res_x is None or res_y is None or dest_crs is None or dtype_name is None:
        raise ValueError("Unable to determine mosaic extent, resolution, or CRS")

    left, bottom, right, top = map(to_float, bounds)
    res_x = to_float(res_x)
    res_y = to_float(res_y)
    width = int(round((right - left) / res_x))
    height = int(round((top - bottom) / res_y))
    if width <= 0 or height <= 0:
        raise ValueError("Computed non-positive mosaic dimensions")

    transform = from_bounds(left, bottom, right, top, width, height)
    dest = np.full((1, height, width), nodata, dtype=np.dtype(dtype_name))

    for path in paths:
        with rasterio.open(path) as src:
            with WarpedVRT(
                src,
                crs=dest_crs,
                transform=transform,
                width=width,
                height=height,
                resampling=Resampling.nearest,
                nodata=nodata,
            ) as vrt:
                window = rasterio.windows.from_bounds(*src.bounds, transform)
                window = window.round_lengths().round_offsets()
                
                row_start = max(0, int(window.row_off))
                row_stop = min(height, int(window.row_off + window.height))
                col_start = max(0, int(window.col_off))
                col_stop = min(width, int(window.col_off + window.width))
                if row_start >= row_stop or col_start >= col_stop:
                    continue
                
                read_window = Window(
                    col_off=col_start,
                    row_off=row_start,
                    width=col_stop - col_start,
                    height=row_stop - row_start,
                )

                out_shape = (int(read_window.height), int(read_window.width))
                if out_shape[0] == 0 or out_shape[1] == 0:
                    continue

                data = vrt.read(1, window=read_window, out_shape=out_shape, masked=False)
                if data.size == 0:
                    continue

                curr = dest[0, row_start:row_stop, col_start:col_stop]
                merge_arrays(curr, data, priority, nodata)
                dest[0, row_start:row_stop, col_start:col_stop] = curr

    return dest, transform


def examine_values(file_path: Path, label: str = "") -> str:
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
        unique_vals, counts = np.unique(data, return_counts=True)
        header = f"{label} " if label else ""
        lines = [f"{header}unique values: {unique_vals.tolist()}"]
        for value, count in zip(unique_vals, counts):
            pct = (count / data.size) * 100
            lines.append(f"  Value {int(value)}: {int(count)} pixels ({pct:.2f}%)")
        return "\n".join(lines)
    except Exception as exc:  # pragma: no cover - diagnostics helper
        return f"Warning: failed examining {file_path}: {exc}"


def ensure_output_dir(base_dir: Path, date: str) -> Path:
    date_dir = base_dir / date
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir


def process_date(
    date: str,
    files: Sequence[Path],
    output_root: Path,
    product: ProductConfig,
    overwrite: bool,
    debug: bool,
) -> ProcessOutcome:
    file_paths = [Path(f) for f in files]
    output_root = Path(output_root)  # Ensure Path in case string was passed via multiprocessing
    if not file_paths:
        return ProcessOutcome(status="error", output=None, reason="No input files supplied")

    output_dir = ensure_output_dir(output_root, date)
    output_path = output_dir / product.output_name

    if output_path.exists() and not overwrite:
        return ProcessOutcome(status="skipped", output=str(output_path), reason="Existing mosaic")

    try:
        merged_data, transform = custom_merge(file_paths, product.nodata, product.priority)
    except Exception as exc:
        return ProcessOutcome(status="error", output=str(output_path), reason=str(exc))

    with rasterio.open(file_paths[0]) as src:
        profile = src.profile.copy()
    try:
        colormap = src.colormap(1)
    except Exception:
        colormap = None

    profile.update(
        {
            "height": merged_data.shape[1],
            "width": merged_data.shape[2],
            "transform": transform,
            "count": 1,
            "nodata": product.nodata,
            "dtype": merged_data.dtype,
        }
    )

    try:
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(merged_data)
            if colormap:
                dst.write_colormap(1, colormap)
    except Exception as exc:
        return ProcessOutcome(status="error", output=str(output_path), reason=str(exc))

    debug_text = examine_values(output_path, f"{date} {product.key}") if debug else None

    return ProcessOutcome(status="created", output=str(output_path), debug=debug_text)


def filter_dates(all_dates: Iterable[str], args: argparse.Namespace) -> List[str]:
    selected = sorted(all_dates)

    if args.year:
        prefix = str(args.year)
        selected = [d for d in selected if d.startswith(prefix)]

    if args.start_date:
        selected = [d for d in selected if d >= args.start_date]

    if args.end_date:
        selected = [d for d in selected if d <= args.end_date]

    if args.dates:
        requested = set(args.dates)
        selected = [d for d in selected if d in requested]

    return selected


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data_dir = resolve_data_dir(args)
    workers = max(1, args.workers)

    if not data_dir.exists():
        print(f"Data directory does not exist: {data_dir}")
        return 1

    raw_dir = data_dir / "raw"
    stitched_dir = data_dir / "stitched_dates"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stitched_dir.mkdir(parents=True, exist_ok=True)

    skip_log: Dict[str, int] = {}

    def move_scene_directories() -> None:
        for entry in list(data_dir.iterdir()):
            if entry in (raw_dir, stitched_dir):
                continue
            if entry.is_dir() and RAW_SCENE_PATTERN.search(entry.name):
                dest = raw_dir / entry.name
                if dest.exists():
                    continue
                entry.rename(dest)

    move_scene_directories()

    any_processed = False

    for product_key in args.products:
        product = PRODUCTS[product_key]
        files_by_date = collect_product_files(raw_dir, product, skip_log)
        if not files_by_date:
            print(f"No {product.key} files discovered under {raw_dir}")
            continue

        dates = filter_dates(files_by_date.keys(), args)
        if not dates:
            print(f"No dates selected for {product.key}")
            continue

        print(
            f"Processing {len(dates)} date(s) for product {product.key} with {workers} worker(s)"
        )

        def handle_result(date: str, outcome: ProcessOutcome) -> None:
            nonlocal any_processed
            if outcome.status == "created":
                print(f"Created {product.key} mosaic for {date}: {outcome.output}")
                if outcome.debug:
                    print(outcome.debug)
                any_processed = True
            elif outcome.status == "skipped":
                reason = outcome.reason or "already exists"
                print(f"Skipping {date} {product.key}: {reason}")
            else:
                reason = outcome.reason or "unknown error"
                print(f"Error merging {product.key} for {date}: {reason}")
                if outcome.debug:
                    print(outcome.debug)

        if workers == 1 or len(dates) == 1:
            for date in dates:
                outcome = process_date(
                    date,
                    files_by_date[date],
                    stitched_dir,
                    product,
                    args.overwrite,
                    args.debug,
                )
                handle_result(date, outcome)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_date,
                        date,
                        [str(p) for p in files_by_date[date]],
                        str(stitched_dir),
                        product,
                        args.overwrite,
                        args.debug,
                    ): date
                    for date in dates
                }

                for future in as_completed(futures):
                    date = futures[future]
                    try:
                        outcome = future.result()
                    except Exception as exc:  # pragma: no cover - unexpected worker crash
                        outcome = ProcessOutcome(status="error", output=None, reason=str(exc))
                    handle_result(date, outcome)

    if not any_processed:
        print("No mosaics created. Adjust filters or ensure data is available.")
        return 1

    if skip_log.get("landsat7_slc_off"):
        print(
            "Skipped Landsat 7 acquisitions on/after 2003-05-31 due to SLC-off striping impact."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
