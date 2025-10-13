#!/usr/bin/env python3
"""Monthly mosaic builder for USGS DSWE stitched outputs.

This script walks a directory tree produced by ``usgs/stitch_dsw_usgs.py``
(typically ``dataDir_usgs`` from ``params.yaml``) where each acquisition date
has a folder containing mosaicked GeoTIFFs (e.g., ``mosaic_intr.tif``). It
groups mosaics by month and merges them into monthly composites using the same
priority-aware merge logic as the daily stitcher. Outputs are written to a
``mosaics/<YYYYMM>`` directory adjacent to the data root unless overridden.

Example
-------
    python monthly_files_merge.py --config-dir . --products INTR --year 2003
"""

from __future__ import annotations

import argparse
import calendar
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio

try:
    import config  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    config = None

try:
    from usgs.stitch_dsw_usgs import custom_merge, LANDSAT7_SLC_OFF_DATE
    from usgs.stitch_dsw_usgs import extract_satellite_number  # type: ignore
except ImportError:  # pragma: no cover - allow standalone execution
    custom_merge = None
    LANDSAT7_SLC_OFF_DATE = "20030531"

if custom_merge is None:
    raise ImportError(
        "monthly_files_merge requires usgs.stitch_dsw_usgs.custom_merge to be importable"
    )


@dataclass(frozen=True)
class ProductConfig:
    key: str
    filename: str
    nodata: int
    priority: Tuple[int, ...]


PRODUCTS: Dict[str, ProductConfig] = {
    "INTR": ProductConfig(
        key="INTR",
        filename="mosaic_intr.tif",
        nodata=255,
        priority=(1, 2, 3, 4, 0, 9),
    ),
}


DATE_REGEX = re.compile(r"^(\d{8})$")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build monthly mosaics from stitched USGS DSWE data")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("."),
        help="Directory containing params.yaml (default: current directory)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Root directory containing date folders with mosaics (defaults to dataDir_usgs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write monthly mosaics (default: <data-dir>/mosaics)",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        choices=sorted(PRODUCTS.keys()),
        default=["INTR"],
        help="Product keys to process (default: INTR)",
    )
    parser.add_argument("--year", type=int, help="Only process months within this calendar year")
    parser.add_argument("--start", help="Process months >= YYYYMM")
    parser.add_argument("--end", help="Process months <= YYYYMM")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing monthly mosaics",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print unique values for generated mosaics",
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
            print(f"Warning: failed to load dataDir_usgs from params ({exc})")

    return (Path(args.config_dir) / "data" / "usgs_dsw").resolve()


def ensure_output_dir(base_output: Path, month: str) -> Path:
    month_dir = base_output / month
    month_dir.mkdir(parents=True, exist_ok=True)
    return month_dir


def month_key(date_str: str) -> str:
    return date_str[:6]


def is_slc_off_directory(path: Path) -> bool:
    satellite = extract_satellite_number(path.name) if 'extract_satellite_number' in globals() else None
    date = path.name
    is_landsat7 = satellite == 7
    return bool(is_landsat7 and date >= LANDSAT7_SLC_OFF_DATE)


def collect_monthly_files(data_dir: Path, product: ProductConfig) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    if not data_dir.exists():
        return grouped

    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if not DATE_REGEX.match(child.name):
            continue
        if is_slc_off_directory(child):
            print(
                f"Skipping month contribution from Landsat 7 SLC-off directory: {child.name}"
            )
            continue

        target = child / product.filename
        if target.exists():
            grouped[month_key(child.name)].append(target)

    return grouped


def filter_months(months: Iterable[str], args: argparse.Namespace) -> List[str]:
    months_sorted = sorted(months)

    if args.year:
        prefix = str(args.year)
        months_sorted = [m for m in months_sorted if m.startswith(prefix)]

    if args.start:
        months_sorted = [m for m in months_sorted if m >= args.start]

    if args.end:
        months_sorted = [m for m in months_sorted if m <= args.end]

    return months_sorted


def examine_values(file_path: Path) -> str:
    with rasterio.open(file_path) as src:
        data = src.read(1)
    values, counts = np.unique(data, return_counts=True)
    lines = [f"unique values: {values.tolist()}"]
    for value, count in zip(values, counts):
        pct = (count / data.size) * 100
        lines.append(f"  {int(value)}: {int(count)} pixels ({pct:.2f}%)")
    return "\n".join(lines)


def process_month(
    month: str,
    files: Sequence[Path],
    product: ProductConfig,
    output_root: Path,
    overwrite: bool,
    debug: bool,
) -> None:
    if not files:
        return

    output_dir = ensure_output_dir(output_root, month)
    output_file = output_dir / product.filename

    if output_file.exists() and not overwrite:
        print(f"Skipping {product.key} {month}: {output_file.name} exists")
        return

    merged, transform = custom_merge(files, product.nodata, product.priority)

    with rasterio.open(files[0]) as src:
        profile = src.profile.copy()
        try:
            colormap = src.colormap(1)
        except Exception:
            colormap = None

    profile.update(
        {
            "height": merged.shape[1],
            "width": merged.shape[2],
            "transform": transform,
            "count": 1,
            "nodata": product.nodata,
            "dtype": merged.dtype,
        }
    )

    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(merged)
        if colormap:
            dst.write_colormap(1, colormap)

    print(f"Created monthly mosaic for {product.key} {month}: {output_file}")
    if debug:
        print(examine_values(output_file))


def month_range_string(month: str) -> str:
    year = int(month[:4])
    mon = int(month[4:])
    name = calendar.month_abbr[mon]
    return f"{name} {year}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data_dir = resolve_data_dir(args)

    if not data_dir.exists():
        print(f"Data directory does not exist: {data_dir}")
        return 1

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (data_dir / "mosaics")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    any_processed = False

    for product_key in args.products:
        product = PRODUCTS[product_key]
        files_by_month = collect_monthly_files(data_dir, product)
        if not files_by_month:
            print(f"No {product.filename} files discovered under {data_dir}")
            continue

        months = filter_months(files_by_month.keys(), args)
        if not months:
            print(f"No months selected for product {product.key}")
            continue

        print(
            f"Processing {len(months)} month(s) for product {product.key} spanning "
            f"{month_range_string(months[0])} to {month_range_string(months[-1])}"
        )

        for month in months:
            process_month(month, files_by_month[month], product, output_dir, args.overwrite, args.debug)
            any_processed = True

    if not any_processed:
        print("No monthly mosaics created. Check filters and data availability.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
