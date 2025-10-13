#!/usr/bin/env python3
"""Build aligned monthly mosaics for DSWE INTR and INWAM products.

This script mirrors ``monthly_files_merge.py`` but writes both the interpreted
(INTR) and aggregated (INWAM) water classification mosaics for each month. It
expects a directory layout produced by ``usgs/stitch_dsw_usgs.py`` where each
acquisition date folder contains mosaics such as ``mosaic_intr.tif`` and
``mosaic_inwam.tif``.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import rasterio

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

from usgs.stitch_dsw_usgs import custom_merge, LANDSAT7_SLC_OFF_DATE, extract_satellite_number


DATE_REGEX = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class ProductConfig:
    key: str
    filename: str
    nodata: int
    priority: Tuple[int, ...]


PRODUCTS: Dict[str, ProductConfig] = {
    "INTR": ProductConfig("INTR", "mosaic_intr.tif", 255, (1, 2, 3, 4, 0, 9)),
    "INWAM": ProductConfig("INWAM", "mosaic_inwam.tif", 255, (1, 2, 3, 4, 0, 9)),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build monthly mosaics for DSWE INTR/INWAM products")
    parser.add_argument("--config-dir", type=Path, default=Path("."), help="Directory containing params.yaml")
    parser.add_argument("--data-dir", type=Path, help="Root directory containing date folders")
    parser.add_argument("--output-dir", type=Path, help="Directory to write monthly mosaics")
    parser.add_argument("--products", nargs="+", choices=["INTR", "INWAM"], default=["INTR", "INWAM"])
    parser.add_argument("--year", type=int)
    parser.add_argument("--start", help="Process months >= YYYYMM")
    parser.add_argument("--end", help="Process months <= YYYYMM")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        return args.data_dir.expanduser().resolve()

    if CONFIG_MODULE is not None:
        ps = CONFIG_MODULE.getPS(str(args.config_dir))
        data_dir = getattr(ps, "dataDir_usgs", None)
        if data_dir:
            return Path(data_dir).expanduser().resolve()

    return (Path(args.config_dir) / "data" / "usgs_dsw").resolve()


def resolve_output_dir(args: argparse.Namespace, data_dir: Path) -> Path:
    if args.output_dir:
        return args.output_dir.expanduser().resolve()
    return (data_dir / "mosaics").resolve()


def is_slc_off(date_dir: Path) -> bool:
    sat = extract_satellite_number(date_dir.name)
    return bool(sat == 7 and date_dir.name >= LANDSAT7_SLC_OFF_DATE)


def collect_products(data_dir: Path, product: ProductConfig) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir() or not DATE_REGEX.match(child.name):
            continue
        if is_slc_off(child):
            continue
        target = child / product.filename
        if target.exists():
            grouped[child.name[:6]].append(target)
    return grouped


def filter_months(months: Iterable[str], args: argparse.Namespace) -> List[str]:
    selected = sorted(months)
    if args.year:
        selected = [m for m in selected if m.startswith(str(args.year))]
    if args.start:
        selected = [m for m in selected if m >= args.start]
    if args.end:
        selected = [m for m in selected if m <= args.end]
    return selected


def ensure_output(output_root: Path, month: str) -> Path:
    dest = output_root / month
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def process_month(month: str, files: Sequence[Path], product: ProductConfig, output_root: Path, overwrite: bool, debug: bool) -> None:
    if not files:
        return
    output_dir = ensure_output(output_root, month)
    output_path = output_dir / product.filename
    if output_path.exists() and not overwrite:
        return

    merged, transform = custom_merge(files, product.nodata, product.priority)
    with rasterio.open(files[0]) as src:
        profile = src.profile.copy()
        try:
            cmap = src.colormap(1)
        except Exception:
            cmap = None
    profile.update({
        "height": merged.shape[1],
        "width": merged.shape[2],
        "transform": transform,
        "count": 1,
        "nodata": product.nodata,
        "dtype": merged.dtype,
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(merged)
        if cmap:
            dst.write_colormap(1, cmap)
    if debug:
        print(f"Wrote {product.key} monthly mosaic {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data_dir = resolve_data_dir(args)
    stitched_dir = data_dir / "stitched_dates"
    if not stitched_dir.exists():
        print(f"Stitched data directory does not exist: {stitched_dir}")
        return 1

    output_dir = resolve_output_dir(args, data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    any_processed = False
    for product_key in args.products:
        product = PRODUCTS[product_key]
        files_by_month = collect_products(stitched_dir, product)
        if not files_by_month:
            print(f"No {product.filename} files found in {stitched_dir}")
            continue
        months = filter_months(files_by_month.keys(), args)
        if not months:
            continue
        print(f"Processing {len(months)} months for product {product.key}...")
        for month in months:
            process_month(
                month,
                files_by_month[month],
                product,
                output_dir,
                args.overwrite,
                args.debug,
            )
            any_processed = True

    if not any_processed:
        print("No monthly products were generated.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
