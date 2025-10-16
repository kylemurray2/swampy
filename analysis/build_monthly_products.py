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

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject

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

from usgs.stitch_dsw_usgs import LANDSAT7_SLC_OFF_DATE, extract_satellite_number


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


WATER_CLASSES = np.array([1, 2, 3, 4], dtype=np.uint8)
LAND_CLASS = np.uint8(0)
CLOUD_CLASS = np.uint8(9)
NODATA_CLASS = np.uint8(255)

# Confidence-weighting parameters
INWAM_WEIGHTS = np.array([0.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
INTR_WEIGHT_FALLBACK: Dict[int, float] = {1: 3.5, 2: 3.0, 3: 2.0, 4: 1.0}
LAND_WEIGHT = 3.0
HIGH_WEIGHT_THRESHOLD = 3.0
MOD_WEIGHT_THRESHOLD = 2.0
MIN_VALID_OBSERVATIONS = 3
MIN_HIGH_COUNT = 2
MIN_HIGH_FRACTION = 0.5
MIN_MODERATE_COUNT = 3
MIN_MODERATE_FRACTION = 0.6
MIN_SCORE_ADVANTAGE = 2.0
MIN_ACCEPTABLE_WEIGHT = 2.0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build monthly mosaics for DSWE INTR/INWAM products")
    parser.add_argument("--config-dir", type=Path, default=Path("."), help="Directory containing params.yaml")
    parser.add_argument("--data-dir", type=Path, help="Root directory containing date folders")
    parser.add_argument("--output-dir", type=Path, help="Directory to write monthly mosaics")
    parser.add_argument("--stats-dir", type=Path, help="Directory containing water statistics (default: <data-dir>/stats)")
    parser.add_argument("--products", nargs="+", choices=["INTR", "INWAM"], default=["INTR", "INWAM"])
    parser.add_argument("--year", type=int)
    parser.add_argument("--start", help="Process months >= YYYYMM")
    parser.add_argument("--end", help="Process months <= YYYYMM")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--max-daily-cloud",
        type=float,
        default=0.6,
        help="Skip daily mosaics where the cloud/nodata fraction exceeds this value (0-1).",
    )
    parser.add_argument(
        "--min-daily-valid",
        type=float,
        default=0.2,
        help="Require at least this fraction of valid pixels in a daily mosaic to include it (0-1).",
    )
    parser.add_argument(
        "--min-daily-water",
        type=float,
        default=0.0,
        help="Optional minimum fraction of water pixels for a daily mosaic to be used (0-1).",
    )
    parser.add_argument(
        "--water-prob-threshold",
        type=float,
        default=0.55,
        help="Water probability threshold used to promote water when votes are ambiguous (0-1).",
    )
    parser.add_argument(
        "--max-stats-mismatch",
        type=float,
        default=0.3,
        help="Maximum fraction of pixels allowed to disagree with statistics before dropping a daily mosaic (0-1).",
    )
    parser.add_argument(
        "--water-fraction-threshold",
        type=float,
        default=0.6,
        help="Minimum fraction of water (classes 1-2) across valid days to classify a pixel as water.",
    )
    parser.add_argument(
        "--partial-fraction-threshold",
        type=float,
        default=0.3,
        help="Minimum fraction of partial water (class 3) across valid days to classify a pixel as partial water when water is not dominant.",
    )
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


def resolve_output_dir(args: argparse.Namespace, data_dir: Path) -> Tuple[Path, Path]:
    if args.output_dir:
        out_dir = args.output_dir.expanduser().resolve()
    else:
        out_dir = (data_dir / "mosaics").resolve()

    if args.stats_dir:
        stats_dir = args.stats_dir.expanduser().resolve()
    else:
        stats_dir = (data_dir / "stats").resolve()

    return out_dir, stats_dir


def load_statistics(stats_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    required = [
        ("water_prob", "water_prob.tif"),
        ("cloud_frac", "cloud_frac.tif"),
        ("persistent_water_mask", "persistent_water_mask.tif"),
        ("stable_land_mask", "stable_land_mask.tif"),
        ("ephemeral_water_mask", "ephemeral_water_mask.tif"),
    ]
    stats: Dict[str, np.ndarray] = {}
    profile: Optional[Dict[str, object]] = None

    for key, filename in required:
        path = stats_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required statistics file not found: {path}")
        with rasterio.open(path) as src:
            data = src.read(1)
            stats[key] = data
            if profile is None:
                profile = src.profile.copy()

    if profile is None:
        raise RuntimeError("Failed to load statistics profile")

    return stats, profile


def reproject_statistics(
    stats: Dict[str, np.ndarray],
    src_profile: Dict[str, object],
    dst_profile: Dict[str, object],
) -> Dict[str, np.ndarray]:
    aligned: Dict[str, np.ndarray] = {}
    for key, array in stats.items():
        dst = np.empty((dst_profile["height"], dst_profile["width"]), dtype=array.dtype)
        reproject(
            source=array,
            destination=dst,
            src_transform=src_profile["transform"],
            src_crs=src_profile["crs"],
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            resampling=Resampling.nearest,
        )
        aligned[key] = dst
    return aligned


def get_profile_key(profile: Dict[str, object]) -> Tuple[int, int, Tuple[float, ...]]:
    transform = profile["transform"]
    return (
        profile["width"],
        profile["height"],
        (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f),
    )


def reproject_array(array: np.ndarray, src_profile: Dict[str, object], dst_profile: Dict[str, object]) -> np.ndarray:
    dst = np.empty((dst_profile["height"], dst_profile["width"]), dtype=array.dtype)
    reproject(
        source=array,
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        resampling=Resampling.nearest,
    )
    return dst


def is_slc_off(date_dir: Path) -> bool:
    sat = extract_satellite_number(date_dir.name)
    return bool(sat == 7 and date_dir.name >= LANDSAT7_SLC_OFF_DATE)


def collect_monthly_observations(stitched_dir: Path) -> Dict[str, Dict[str, Dict[str, Path]]]:
    """Return mapping month -> date -> product key -> file path."""

    months: Dict[str, Dict[str, Dict[str, Path]]] = defaultdict(lambda: defaultdict(dict))
    if not stitched_dir.exists():
        return {}

    for date_dir in sorted(stitched_dir.iterdir()):
        if not date_dir.is_dir() or not DATE_REGEX.match(date_dir.name):
            continue
        if is_slc_off(date_dir):
            continue

        month_key = date_dir.name[:6]
        for product in PRODUCTS.values():
            candidate = date_dir / product.filename
            if candidate.exists():
                months[month_key][date_dir.name][product.key] = candidate

    return months


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
    month_dir = output_root / month
    month_dir.mkdir(parents=True, exist_ok=True)
    return month_dir


def _ensure_same_grid(
    dataset: rasterio.io.DatasetReader,
    reference_profile: Dict[str, object],
    path: str,
) -> None:
    if (
        dataset.width != reference_profile["width"]
        or dataset.height != reference_profile["height"]
        or dataset.transform != reference_profile["transform"]
        or dataset.crs != reference_profile.get("crs")
    ):
        raise ValueError(f"Raster {path} does not share the common grid")


def read_stack(paths: Sequence[Path]) -> Tuple[np.ndarray, Dict[str, object], Optional[Dict[int, Tuple[int, int, int, int]]]]:
    arrays: List[np.ndarray] = []
    profile: Optional[Dict[str, object]] = None
    colormap: Optional[Dict[int, Tuple[int, int, int, int]]] = None

    for path in paths:
        with rasterio.open(path) as src:
            resampling_needed = False
            if profile is None:
                profile = src.profile.copy()
                try:
                    colormap = src.colormap(1)
                except Exception:
                    colormap = None
            else:
                if (
                    src.width != profile["width"]
                    or src.height != profile["height"]
                    or src.transform != profile["transform"]
                    or src.crs != profile.get("crs")
                ):
                    resampling_needed = True

            if profile is not None and resampling_needed:
                with WarpedVRT(
                    src,
                    crs=profile.get("crs"),
                    transform=profile["transform"],
                    width=int(profile["width"]),
                    height=int(profile["height"]),
                    resampling=Resampling.nearest,
                ) as vrt:
                    arrays.append(vrt.read(1).astype(np.uint8))
            else:
                arrays.append(src.read(1).astype(np.uint8))

    if profile is None or not arrays:
        raise ValueError("No rasters available for stacking")

    stack = np.stack(arrays, axis=0)
    return stack, profile, colormap


def read_inwam_stack(
    paths: Sequence[Optional[Path]],
    reference_profile: Dict[str, object],
) -> Tuple[Optional[np.ndarray], Optional[Dict[int, Tuple[int, int, int, int]]]]:
    height = int(reference_profile["height"])
    width = int(reference_profile["width"])
    stack = np.full((len(paths), height, width), fill_value=-1, dtype=np.int16)
    colormap: Optional[Dict[int, Tuple[int, int, int, int]]] = None
    has_any = False

    for idx, path in enumerate(paths):
        if path is None:
            continue
        candidate = Path(path)
        if not candidate.exists():
            continue
        with rasterio.open(candidate) as src:
            resample = False
            if (
                src.width != reference_profile["width"]
                or src.height != reference_profile["height"]
                or src.transform != reference_profile["transform"]
                or src.crs != reference_profile.get("crs")
            ):
                resample = True

            if resample:
                with WarpedVRT(
                    src,
                    crs=reference_profile.get("crs"),
                    transform=reference_profile["transform"],
                    width=int(reference_profile["width"]),
                    height=int(reference_profile["height"]),
                    resampling=Resampling.nearest,
                ) as vrt:
                    stack[idx] = vrt.read(1).astype(np.int16)
            else:
                stack[idx] = src.read(1).astype(np.int16)
            has_any = True
            if colormap is None:
                try:
                    colormap = src.colormap(1)
                except Exception:
                    colormap = None

    if not has_any:
        return None, None

    return stack, colormap


def examine_values(array: np.ndarray, label: str) -> str:
    values, counts = np.unique(array, return_counts=True)
    lines = [f"{label} unique values: {values.tolist()}"]
    total = array.size
    for value, count in zip(values, counts):
        pct = (count / total) * 100 if total else 0
        lines.append(f"  {int(value)}: {int(count)} pixels ({pct:.2f}%)")
    return "\n".join(lines)


def aggregate_month(
    intr_stack: np.ndarray,
    inwam_stack: Optional[np.ndarray],
    stats: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n_obs, height, width = intr_stack.shape

    valid_mask = (intr_stack != CLOUD_CLASS) & (intr_stack != NODATA_CLASS)
    nodata_mask = np.all(intr_stack == NODATA_CLASS, axis=0)
    valid_count = valid_mask.sum(axis=0)
    no_valid = (valid_count == 0) & ~nodata_mask

    water_mask_daily = np.isin(intr_stack, [1, 2]) & valid_mask
    partial_mask_daily = (intr_stack == 3) & valid_mask
    water_sum = water_mask_daily.sum(axis=0)
    partial_sum = partial_mask_daily.sum(axis=0)

    water_mask = np.isin(intr_stack, WATER_CLASSES) & valid_mask
    land_mask = (intr_stack == LAND_CLASS) & valid_mask

    weights = np.zeros((n_obs, height, width), dtype=np.float32)

    has_inwam = inwam_stack is not None and np.any(inwam_stack >= 0)
    if has_inwam and inwam_stack is not None:
        inwam_valid = (inwam_stack >= 1) & (inwam_stack <= 4)
        clipped = np.clip(inwam_stack, 0, 4).astype(np.int16)
        weights = np.where(
            water_mask & inwam_valid,
            INWAM_WEIGHTS[clipped],
            weights,
        )
        missing_weight_mask = water_mask & ~inwam_valid
    else:
        missing_weight_mask = water_mask

    for cls, weight in INTR_WEIGHT_FALLBACK.items():
        cls_mask = (intr_stack == cls) & missing_weight_mask
        if np.any(cls_mask):
            weights = np.where(cls_mask, weight, weights)

    weights = np.where(land_mask, LAND_WEIGHT, weights)

    high_conf_mask = water_mask & (weights >= HIGH_WEIGHT_THRESHOLD)
    moderate_conf_mask = water_mask & (weights >= MOD_WEIGHT_THRESHOLD)

    high_count = high_conf_mask.sum(axis=0)
    moderate_count = moderate_conf_mask.sum(axis=0)
    water_weight_sum = np.sum(np.where(water_mask, weights, 0.0), axis=0)
    land_weight_sum = np.sum(np.where(land_mask, weights, 0.0), axis=0)
    max_water_weight = np.max(np.where(water_mask, weights, 0.0), axis=0)

    valid_count_float = valid_count.astype(np.float32)
    high_fraction = np.divide(
        high_count.astype(np.float32),
        valid_count_float,
        out=np.zeros_like(valid_count_float, dtype=np.float32),
        where=valid_count_float > 0,
    )
    moderate_fraction = np.divide(
        moderate_count.astype(np.float32),
        valid_count_float,
        out=np.zeros_like(valid_count_float, dtype=np.float32),
        where=valid_count_float > 0,
    )

    class_scores = []
    for cls in WATER_CLASSES:
        class_scores.append(
            np.sum(np.where(intr_stack == cls, weights, 0.0), axis=0)
        )

    if class_scores:
        class_scores_arr = np.stack(class_scores, axis=0)
        best_idx = np.argmax(class_scores_arr, axis=0)
        best_scores = np.take_along_axis(
            class_scores_arr, best_idx[np.newaxis, ...], axis=0
        )[0]
        best_classes = np.take(WATER_CLASSES, best_idx)
        
        # Key insight: partial water (class 3) represents uncertain/narrow/ephemeral water
        # If ANY daily observation shows class 3, promote the pixel to class 3 in the monthly
        # This overrides the weighted vote to preserve ephemeral features
        has_class3 = (intr_stack == 3).any(axis=0) & valid_mask.any(axis=0)
        best_classes = np.where(has_class3, 3, best_classes)
    else:
        best_scores = np.zeros((height, width), dtype=np.float32)
        best_classes = np.full((height, width), LAND_CLASS, dtype=np.uint8)

    valid_enough = valid_count >= MIN_VALID_OBSERVATIONS

    water_vote = (
        (high_count >= MIN_HIGH_COUNT) & (high_fraction >= MIN_HIGH_FRACTION)
    )
    water_vote |= (
        (moderate_count >= MIN_MODERATE_COUNT)
        & (moderate_fraction >= MIN_MODERATE_FRACTION)
        & (water_weight_sum >= land_weight_sum + MIN_SCORE_ADVANTAGE)
    )
    water_vote |= (
        (high_count >= 1)
        & (water_weight_sum >= land_weight_sum + MIN_SCORE_ADVANTAGE)
        & (max_water_weight >= HIGH_WEIGHT_THRESHOLD)
    )

    water_vote &= valid_enough
    water_vote &= best_scores > 0
    water_vote &= max_water_weight >= MIN_ACCEPTABLE_WEIGHT

    monthly_intr = np.full((height, width), LAND_CLASS, dtype=np.uint8)
    monthly_intr[nodata_mask] = NODATA_CLASS
    monthly_intr[no_valid] = CLOUD_CLASS
    monthly_intr[water_vote] = best_classes[water_vote]

    # Downgrade sparse water observations to partial water (class 3)
    # If a pixel was classified as class 1-2 but:
    # - has valid_count < 5 (sparse coverage), AND
    # - has partial_sum >= 1 (showed partial water on at least one day)
    # Then it's probably an ephemeral/narrow feature that should be class 3, not class 1
    sparse_water = (valid_count < 5) & (np.isin(monthly_intr, [1, 2])) & (partial_sum >= 1)
    monthly_intr[sparse_water] = 3

    # Statistics guidance
    persistent_mask = stats.get("persistent_water_mask", np.zeros((height, width), dtype=bool)).astype(bool)
    stable_mask = stats.get("stable_land_mask", np.zeros((height, width), dtype=bool)).astype(bool)
    ephemeral_mask = stats.get("ephemeral_water_mask", np.zeros((height, width), dtype=bool)).astype(bool)
    water_prob = stats.get("water_prob", np.zeros((height, width), dtype=np.float32))

    monthly_intr[persistent_mask] = 1
    monthly_intr[stable_mask & ~persistent_mask] = LAND_CLASS

    updatable = ~(persistent_mask | stable_mask)

    valid_count_float = valid_count.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        water_fraction = np.divide(
            water_sum.astype(np.float32),
            valid_count_float,
            out=np.zeros_like(valid_count_float, dtype=np.float32),
            where=valid_count_float > 0,
        )
        partial_fraction = np.divide(
            partial_sum.astype(np.float32),
            valid_count_float,
            out=np.zeros_like(valid_count_float, dtype=np.float32),
            where=valid_count_float > 0,
        )

    set_water = water_fraction >= args.water_fraction_threshold
    set_partial = partial_fraction >= args.partial_fraction_threshold

    conflict = set_water & set_partial
    conflict_water = conflict & (water_fraction >= partial_fraction)
    conflict_partial = conflict & ~conflict_water

    mask = updatable & conflict_water
    monthly_intr[mask] = 1

    mask = updatable & conflict_partial
    monthly_intr[mask] = 3

    mask = updatable & ~conflict & set_water
    monthly_intr[mask] = 1

    mask = updatable & ~conflict & ~set_water & set_partial
    monthly_intr[mask] = 3

    # For any pixel with water evidence that hasn't been classified yet, promote to partial water
    # This captures all small water features (rivers, ponds, etc.) even with sparse daily coverage
    # Key insight: if a pixel showed water/partial water on ANY day, it should be class 3 (not land)
    has_any_water_evidence = (water_sum > 0) | (partial_sum > 0)
    still_land = monthly_intr == LAND_CLASS
    mask = updatable & has_any_water_evidence & still_land
    monthly_intr[mask] = 3

    # Fallback using statistics-based probability for high water_prob areas
    # IMPORTANT: Only use statistics if the pixel has water or partial water evidence in daily data
    # This prevents hallucinating water in pixels that show ONLY land + nodata/clouds
    has_water_or_partial = (water_sum > 0) | (partial_sum > 0)
    mask = updatable & (monthly_intr == LAND_CLASS) & has_water_or_partial & (water_prob >= args.water_prob_threshold)
    monthly_intr[mask] = 1

    # FINAL PASS: Promote pixels with partial water evidence to class 3
    # 1. If a pixel voted as water but has significant partial evidence (>50% partial), demote to class 3
    # 2. If a pixel voted as land but shows partial on ANY day, promote to class 3
    #    (These are river edges/margins that are mostly land but have partial water evidence)
    
    water_plus_partial = water_sum + partial_sum
    is_water_from_voting = np.isin(monthly_intr, [1, 2])
    is_land_from_voting = monthly_intr == LAND_CLASS
    
    # Case 1: Water → Partial only if ALSO has land evidence
    # Don't demote pure-water pixels; only demote mixed water+land+partial pixels
    # Compute which pixels show land on ANY day
    has_land_evidence = (land_mask.any(axis=0)).astype(bool)
    
    # Only demote to partial if: voted as water + significant partial evidence + has land mixed in
    with np.errstate(divide="ignore", invalid="ignore"):
        partial_ratio = np.divide(
            partial_sum.astype(np.float32),
            water_plus_partial.astype(np.float32),
            out=np.zeros_like(valid_count_float, dtype=np.float32),
            where=water_plus_partial > 0,
        )
    has_significant_partial = (water_plus_partial > 0) & (partial_ratio >= 0.5)
    mask = is_water_from_voting & has_significant_partial & has_land_evidence
    monthly_intr[mask] = 3
    
    # Case 2: Land → Partial if partial water evidence BUT NOT much actual water
    # Only promote to partial if partial is more abundant than water
    # This captures river margins/edges with mixed land+partial, but preserves pure-water areas
    has_any_partial = partial_sum > 0
    partial_dominates_water = partial_sum > water_sum
    mask = is_land_from_voting & has_any_partial & partial_dominates_water
    monthly_intr[mask] = 3
    
    # Also promote land to water if there's significant water evidence (>1 day)
    has_water = water_sum > 1
    mask = is_land_from_voting & has_water & ~partial_dominates_water
    monthly_intr[mask] = 1

    monthly_inwam: Optional[np.ndarray] = None
    if has_inwam:
        monthly_inwam = monthly_intr.copy()

    return monthly_intr, monthly_inwam


def daily_matches_statistics(
    arr: np.ndarray,
    profile: Dict[str, object],
    stats: Dict[str, np.ndarray],
    stats_profile: Dict[str, object],
    args: argparse.Namespace,
    cache: Dict[Tuple[int, int, Tuple[float, ...]], np.ndarray],
) -> Tuple[bool, float]:
    valid_mask = (arr != NODATA_CLASS) & (arr != CLOUD_CLASS)
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return False, 1.0

    key = get_profile_key(profile)
    water_prob = cache.get(key)
    if water_prob is None:
        water_prob = reproject_array(stats["water_prob"], stats_profile, profile)
        cache[key] = water_prob

    expected_water = water_prob >= args.water_prob_threshold
    water_mask = np.isin(arr, WATER_CLASSES)

    mismatches = ((expected_water & ~water_mask) | (~expected_water & water_mask)) & valid_mask
    mismatch_frac = mismatches.sum() / valid_count

    return mismatch_frac <= args.max_stats_mismatch, mismatch_frac


def process_month(
    month: str,
    per_date: Dict[str, Dict[str, Path]],
    output_root: Path,
    requested_products: Sequence[str],
    overwrite: bool,
    debug: bool,
    stats: Dict[str, np.ndarray],
    stats_profile: Dict[str, object],
    cached_stats: Dict[str, np.ndarray],
    stats_cache: Dict[Tuple[int, int, Tuple[float, ...]], np.ndarray],
    args: argparse.Namespace,
) -> bool:
    dates = sorted(per_date.keys())
    intr_paths: List[Path] = []
    inwam_paths: List[Optional[Path]] = []

    for date in dates:
        intr_path = per_date[date].get("INTR")
        if intr_path is None:
            continue
        with rasterio.open(intr_path) as src:
            arr = src.read(1)
            profile = src.profile.copy()
            total = arr.size
            valid = np.count_nonzero((arr != NODATA_CLASS) & (arr != CLOUD_CLASS))
            cloud = np.count_nonzero(arr == CLOUD_CLASS)
            water = np.count_nonzero(np.isin(arr, WATER_CLASSES))
            valid_frac = valid / total
            cloud_frac = cloud / total
            water_frac = water / total

        if cloud_frac > args.max_daily_cloud:
            if debug:
                print(
                    f"Skipping {date} due to high cloud fraction {cloud_frac:.2f} (> {args.max_daily_cloud})"
                )
            continue
        if valid_frac < args.min_daily_valid:
            if debug:
                print(
                    f"Skipping {date} due to low valid fraction {valid_frac:.2f} (< {args.min_daily_valid})"
                )
            continue
        if args.min_daily_water > 0 and water_frac < args.min_daily_water:
            if debug:
                print(
                    f"Skipping {date} due to low water fraction {water_frac:.2f} (< {args.min_daily_water})"
                )
            continue

        ok, mismatch_frac = daily_matches_statistics(
            arr,
            profile,
            stats,
            stats_profile,
            args,
            stats_cache,
        )
        if not ok:
            if debug:
                print(
                    f"Skipping {date} due to high mismatch with stats ({mismatch_frac:.2f} > {args.max_stats_mismatch})"
                )
            continue

        intr_paths.append(Path(intr_path))
        inwam_paths.append(per_date[date].get("INWAM"))

    if not intr_paths:
        if debug:
            print(f"No daily mosaics passed QC for month {month}")
        return False

    intr_stack, profile, intr_cmap = read_stack(intr_paths)
    inwam_stack, inwam_cmap = read_inwam_stack(inwam_paths, profile)

    if not cached_stats or cached_stats.get("__profile_id__") != profile["transform"]:
        aligned = reproject_statistics(stats, stats_profile, profile)
        aligned["__profile_id__"] = profile["transform"]
        cached_stats.clear()
        cached_stats.update(aligned)

    monthly_intr, monthly_inwam = aggregate_month(
        intr_stack,
        inwam_stack,
        cached_stats,
        args,
    )

    month_dir = ensure_output(output_root, month)
    processed = False

    if "INTR" in requested_products:
        intr_output = month_dir / PRODUCTS["INTR"].filename
        if overwrite or not intr_output.exists():
            intr_profile = profile.copy()
            intr_profile.update(
                {
                    "count": 1,
                    "dtype": "uint8",
                    "nodata": int(NODATA_CLASS),
                }
            )
            with rasterio.open(intr_output, "w", **intr_profile) as dst:
                dst.write(monthly_intr[np.newaxis, ...])
                if intr_cmap:
                    dst.write_colormap(1, intr_cmap)
            processed = True
            if debug:
                print(examine_values(monthly_intr, f"{month} INTR"))

    if "INWAM" in requested_products and monthly_inwam is not None:
        inwam_output = month_dir / PRODUCTS["INWAM"].filename
        if overwrite or not inwam_output.exists():
            inwam_profile = profile.copy()
            inwam_profile.update(
                {
                    "count": 1,
                    "dtype": "uint8",
                    "nodata": int(NODATA_CLASS),
                }
            )
            with rasterio.open(inwam_output, "w", **inwam_profile) as dst:
                dst.write(monthly_inwam[np.newaxis, ...])
                if inwam_cmap:
                    dst.write_colormap(1, inwam_cmap)
            processed = True
            if debug:
                print(examine_values(monthly_inwam, f"{month} INWAM"))

    return processed


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data_dir = resolve_data_dir(args)
    stitched_dir = data_dir / "stitched_dates"
    if not stitched_dir.exists():
        print(f"Stitched data directory does not exist: {stitched_dir}")
        return 1

    output_dir, stats_dir = resolve_output_dir(args, data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats, stats_profile = load_statistics(stats_dir)
    cached_stats: Dict[str, np.ndarray] = {}

    monthly_observations = collect_monthly_observations(stitched_dir)
    if not monthly_observations:
        print(f"No stitched mosaics discovered under {stitched_dir}")
        return 1

    months = filter_months(monthly_observations.keys(), args)
    if not months:
        print("No months matched the requested filters.")
        return 1

    # Determine statistics reprojected to mosaic grid once we know the profile
    reprojected_stats: Optional[Dict[str, np.ndarray]] = None

    any_processed = False
    for month in months:
        processed = process_month(
            month,
            monthly_observations[month],
            output_dir,
            args.products,
            args.overwrite,
            args.debug,
            stats,
            stats_profile,
            cached_stats,
            {}, # stats_cache is not used in this function, but passed as a placeholder
            args,
        )
        if processed:
            any_processed = True

    if not any_processed:
        print("No monthly products were generated.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
