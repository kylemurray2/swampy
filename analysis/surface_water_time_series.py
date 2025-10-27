#!/usr/bin/env python3
"""Comprehensive surface water time series analysis and visualization.

This script analyzes long-term surface water changes from monthly DSWE mosaics,
extracting physically meaningful metrics and performing robust statistical analyses.
Outputs are designed for comparison with climate indices (ENSO, PDO, etc.) and
other geophysical time series.

Scientific Approach:
1. Define water bodies using multi-temporal statistics
2. Extract spatial and temporal metrics from monthly mosaics
3. Perform trend analysis, seasonal decomposition, and anomaly detection
4. Generate publication-quality visualizations
5. Export analysis-ready data formats (CSV, NetCDF)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
import rasterio
from rasterio.features import shapes
from scipy import stats as scipy_stats
from scipy.ndimage import label, binary_erosion, binary_dilation
from scipy.signal import detrend
from skimage.measure import regionprops
from statsmodels.tsa.seasonal import STL
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Classification constants
WATER_CLASSES = {1, 2, 3, 4}
HIGH_CONF_WATER = {1, 2}
PARTIAL_WATER = {3}
LOW_CONF_WATER = {4}
LAND_CLASS = 0
CLOUD_CLASS = 9
NODATA_CLASS = 255

# Analysis thresholds
PERMANENT_WATER_THRESHOLD = 0.80
SEASONAL_WATER_MIN = 0.15
SEASONAL_WATER_MAX = 0.80
MIN_WATER_BODY_SIZE = 10  # pixels


@dataclass
class WaterBodyMetrics:
    """Container for water body spatial metrics."""
    total_area_km2: float
    permanent_area_km2: float
    seasonal_area_km2: float
    ephemeral_area_km2: float
    high_conf_area_km2: float
    partial_area_km2: float
    num_water_bodies: int
    mean_body_size_km2: float
    max_body_size_km2: float
    fragmentation_index: float
    centroid_x: float
    centroid_y: float


@dataclass
class TimeSeriesMetrics:
    """Container for all time series metrics."""
    date: str
    year: int
    month: int
    metrics: WaterBodyMetrics
    valid_pixels: int
    cloud_fraction: float


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze surface water time series from monthly mosaics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing cleaned monthly mosaics (cleaned/YYYYMM/)",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        required=True,
        help="Directory containing water statistics (water_prob.tif, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for analysis outputs",
    )
    parser.add_argument(
        "--start-date",
        help="Start date YYYYMM (default: earliest available)",
    )
    parser.add_argument(
        "--end-date",
        help="End date YYYYMM (default: latest available)",
    )
    parser.add_argument(
        "--permanent-threshold",
        type=float,
        default=PERMANENT_WATER_THRESHOLD,
        help="Water probability threshold for permanent water",
    )
    parser.add_argument(
        "--seasonal-min",
        type=float,
        default=SEASONAL_WATER_MIN,
        help="Minimum water probability for seasonal water",
    )
    parser.add_argument(
        "--seasonal-max",
        type=float,
        default=SEASONAL_WATER_MAX,
        help="Maximum water probability for seasonal water",
    )
    parser.add_argument(
        "--min-body-size",
        type=int,
        default=MIN_WATER_BODY_SIZE,
        help="Minimum water body size in pixels",
    )
    parser.add_argument(
        "--trend-method",
        choices=["linear", "mann-kendall", "both"],
        default="both",
        help="Method for trend analysis",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=12,
        help="Seasonal period in months for decomposition",
    )
    parser.add_argument(
        "--anomaly-baseline-start",
        help="Start date YYYYMM for anomaly baseline (default: use all data)",
    )
    parser.add_argument(
        "--anomaly-baseline-end",
        help="End date YYYYMM for anomaly baseline (default: use all data)",
    )
    parser.add_argument(
        "--export-netcdf",
        action="store_true",
        help="Export spatial-temporal data as NetCDF",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information",
    )
    return parser.parse_args(argv)


def load_statistics(stats_dir: Path, verbose: bool = False) -> Dict[str, np.ndarray]:
    """Load pre-computed water statistics."""
    stats = {}
    required_files = {
        "water_prob": "water_prob.tif",
        "persistent_mask": "persistent_water_mask.tif",
        "stable_land_mask": "stable_land_mask.tif",
        "ephemeral_mask": "ephemeral_water_mask.tif",
    }
    
    for key, filename in required_files.items():
        path = stats_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required statistics file not found: {path}")
        with rasterio.open(path) as src:
            stats[key] = src.read(1)
            if key == "water_prob":
                stats["profile"] = src.profile.copy()
                stats["transform"] = src.transform
                stats["crs"] = src.crs
                # Calculate pixel area in km²
                pixel_width = abs(src.transform.a)
                pixel_height = abs(src.transform.e)
                # Approximate: works well for small areas in UTM or similar projections
                stats["pixel_area_km2"] = (pixel_width * pixel_height) / 1e6
    
    if verbose:
        print(f"Loaded statistics from {stats_dir}")
        print(f"  Water probability range: [{stats['water_prob'].min():.3f}, {stats['water_prob'].max():.3f}]")
        print(f"  Pixel area: {stats['pixel_area_km2']:.6f} km²")
    
    return stats


def create_water_body_masks(
    water_prob: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    """Create masks defining different water body types."""
    permanent_mask = water_prob >= args.permanent_threshold
    seasonal_mask = (water_prob >= args.seasonal_min) & (water_prob < args.seasonal_max)
    ephemeral_mask = (water_prob > 0) & (water_prob < args.seasonal_min)
    
    return {
        "permanent": permanent_mask,
        "seasonal": seasonal_mask,
        "ephemeral": ephemeral_mask,
        "any_water": water_prob > 0,
    }


def find_monthly_mosaics(data_dir: Path, start: Optional[str], end: Optional[str]) -> List[Tuple[str, Path]]:
    """Find all monthly mosaic files within date range."""
    mosaics = []
    
    # Check both cleaned and mosaics directories
    for subdir in ["cleaned", "mosaics"]:
        mosaic_root = data_dir / subdir
        if not mosaic_root.exists():
            continue
        
        for month_dir in sorted(mosaic_root.iterdir()):
            if not month_dir.is_dir():
                continue
            month_str = month_dir.name
            if len(month_str) != 6 or not month_str.isdigit():
                continue
            
            if start and month_str < start:
                continue
            if end and month_str > end:
                continue
            
            # Look for mosaic files
            for filename in ["mosaic_intr_clean.tif", "mosaic_intr.tif"]:
                mosaic_path = month_dir / filename
                if mosaic_path.exists():
                    mosaics.append((month_str, mosaic_path))
                    break
    
    return sorted(mosaics, key=lambda x: x[0])


def calculate_water_body_metrics(
    monthly_data: np.ndarray,
    water_masks: Dict[str, np.ndarray],
    pixel_area_km2: float,
    transform: rasterio.Affine,
    min_size: int,
) -> WaterBodyMetrics:
    """Calculate comprehensive water body metrics for a single month."""
    
    # Handle grid mismatch - align water_masks to monthly_data if needed
    if monthly_data.shape != water_masks["permanent"].shape:
        # Reproject water masks to match monthly data grid
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        shape_ratio_y = monthly_data.shape[0] / water_masks["permanent"].shape[0]
        shape_ratio_x = monthly_data.shape[1] / water_masks["permanent"].shape[1]
        
        # Resample masks to match monthly data shape
        aligned_masks = {}
        for key, mask in water_masks.items():
            if isinstance(mask, np.ndarray):
                if mask.ndim == 2:
                    # Use nearest neighbor for masks
                    aligned_masks[key] = zoom(
                        mask.astype(float),
                        (shape_ratio_y, shape_ratio_x),
                        order=0
                    ).astype(mask.dtype)
                else:
                    aligned_masks[key] = mask
            else:
                aligned_masks[key] = mask
        water_masks = aligned_masks
    
    # Basic area calculations
    water_mask = np.isin(monthly_data, list(WATER_CLASSES))
    high_conf_mask = np.isin(monthly_data, list(HIGH_CONF_WATER))
    partial_mask = np.isin(monthly_data, list(PARTIAL_WATER))
    
    total_area = np.sum(water_mask) * pixel_area_km2
    high_conf_area = np.sum(high_conf_mask) * pixel_area_km2
    partial_area = np.sum(partial_mask) * pixel_area_km2
    
    # Area by water body type
    permanent_water = water_mask & water_masks["permanent"]
    seasonal_water = water_mask & water_masks["seasonal"]
    ephemeral_water = water_mask & water_masks["ephemeral"]
    
    permanent_area = np.sum(permanent_water) * pixel_area_km2
    seasonal_area = np.sum(seasonal_water) * pixel_area_km2
    ephemeral_area = np.sum(ephemeral_water) * pixel_area_km2
    
    # Connected component analysis for water bodies
    labeled_array, num_features = label(water_mask)
    
    # Filter small water bodies
    if num_features > 0:
        regions = regionprops(labeled_array)
        valid_bodies = [r for r in regions if r.area >= min_size]
        num_water_bodies = len(valid_bodies)
        
        if valid_bodies:
            body_sizes_km2 = [r.area * pixel_area_km2 for r in valid_bodies]
            mean_body_size = np.mean(body_sizes_km2)
            max_body_size = np.max(body_sizes_km2)
            
            # Calculate fragmentation index (perimeter/area ratio)
            # Higher values = more fragmented
            total_perimeter = sum(r.perimeter for r in valid_bodies)
            fragmentation = total_perimeter / (np.sum(water_mask) + 1e-10)
            
            # Calculate centroid in geographic coordinates
            if water_mask.sum() > 0:
                y_indices, x_indices = np.where(water_mask)
                centroid_y = np.mean(y_indices)
                centroid_x = np.mean(x_indices)
                # Convert to geographic coordinates
                geo_x, geo_y = transform * (centroid_x, centroid_y)
            else:
                geo_x, geo_y = 0, 0
        else:
            num_water_bodies = 0
            mean_body_size = 0
            max_body_size = 0
            fragmentation = 0
            geo_x, geo_y = 0, 0
    else:
        num_water_bodies = 0
        mean_body_size = 0
        max_body_size = 0
        fragmentation = 0
        geo_x, geo_y = 0, 0
    
    return WaterBodyMetrics(
        total_area_km2=total_area,
        permanent_area_km2=permanent_area,
        seasonal_area_km2=seasonal_area,
        ephemeral_area_km2=ephemeral_area,
        high_conf_area_km2=high_conf_area,
        partial_area_km2=partial_area,
        num_water_bodies=num_water_bodies,
        mean_body_size_km2=mean_body_size,
        max_body_size_km2=max_body_size,
        fragmentation_index=fragmentation,
        centroid_x=geo_x,
        centroid_y=geo_y,
    )


def extract_time_series(
    mosaics: List[Tuple[str, Path]],
    stats: Dict[str, np.ndarray],
    water_masks: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> List[TimeSeriesMetrics]:
    """Extract time series metrics from monthly mosaics."""
    time_series = []
    
    for month_str, mosaic_path in mosaics:
        if args.verbose:
            print(f"Processing {month_str}...")
        
        with rasterio.open(mosaic_path) as src:
            data = src.read(1)
            
            # Calculate cloud/nodata fraction
            valid_mask = (data != NODATA_CLASS)
            cloud_mask = (data == CLOUD_CLASS)
            total_pixels = data.size
            valid_pixels = np.sum(valid_mask)
            cloud_fraction = np.sum(cloud_mask) / total_pixels if total_pixels > 0 else 0
            
            # Calculate metrics
            metrics = calculate_water_body_metrics(
                data,
                water_masks,
                stats["pixel_area_km2"],
                stats["transform"],
                args.min_body_size,
            )
            
            year = int(month_str[:4])
            month = int(month_str[4:6])
            
            time_series.append(TimeSeriesMetrics(
                date=month_str,
                year=year,
                month=month,
                metrics=metrics,
                valid_pixels=valid_pixels,
                cloud_fraction=cloud_fraction,
            ))
    
    return time_series


def time_series_to_dataframe(time_series: List[TimeSeriesMetrics]) -> pd.DataFrame:
    """Convert time series metrics to a pandas DataFrame."""
    records = []
    
    for ts in time_series:
        record = {
            "date": ts.date,
            "datetime": pd.to_datetime(ts.date, format="%Y%m"),
            "year": ts.year,
            "month": ts.month,
            "total_area_km2": ts.metrics.total_area_km2,
            "permanent_area_km2": ts.metrics.permanent_area_km2,
            "seasonal_area_km2": ts.metrics.seasonal_area_km2,
            "ephemeral_area_km2": ts.metrics.ephemeral_area_km2,
            "high_conf_area_km2": ts.metrics.high_conf_area_km2,
            "partial_area_km2": ts.metrics.partial_area_km2,
            "num_water_bodies": ts.metrics.num_water_bodies,
            "mean_body_size_km2": ts.metrics.mean_body_size_km2,
            "max_body_size_km2": ts.metrics.max_body_size_km2,
            "fragmentation_index": ts.metrics.fragmentation_index,
            "centroid_x": ts.metrics.centroid_x,
            "centroid_y": ts.metrics.centroid_y,
            "valid_pixels": ts.valid_pixels,
            "cloud_fraction": ts.cloud_fraction,
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def mann_kendall_test(data: np.ndarray) -> Tuple[float, float, str]:
    """Perform Mann-Kendall trend test."""
    n = len(data)
    s = 0
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(data[j] - data[i])
    
    # Calculate variance
    var_s = n * (n - 1) * (2 * n + 5) / 18
    
    # Calculate Z-statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    # Calculate p-value
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
    
    # Determine trend
    if p_value < 0.05:
        if z > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
    else:
        trend = "no significant trend"
    
    return z, p_value, trend


def sens_slope(data: np.ndarray, time: np.ndarray) -> float:
    """Calculate Sen's slope estimator."""
    slopes = []
    n = len(data)
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            if time[j] != time[i]:
                slope = (data[j] - data[i]) / (time[j] - time[i])
                slopes.append(slope)
    
    return np.median(slopes) if slopes else 0


def perform_trend_analysis(df: pd.DataFrame, column: str, method: str = "both") -> Dict:
    """Perform trend analysis on a time series."""
    data = df[column].values
    time = np.arange(len(data))
    
    results = {}
    
    # Linear regression
    if method in ["linear", "both"]:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(time, data)
        results["linear"] = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_error": std_err,
        }
    
    # Mann-Kendall test
    if method in ["mann-kendall", "both"]:
        mk_z, mk_p, mk_trend = mann_kendall_test(data)
        mk_slope = sens_slope(data, time)
        results["mann_kendall"] = {
            "z_statistic": mk_z,
            "p_value": mk_p,
            "trend": mk_trend,
            "sens_slope": mk_slope,
        }
    
    return results


def seasonal_decomposition(df: pd.DataFrame, column: str, period: int = 12) -> pd.DataFrame:
    """Perform STL seasonal decomposition."""
    if len(df) < 2 * period:
        print(f"Warning: Not enough data for seasonal decomposition (need at least {2*period} points)")
        return df
    
    try:
        stl = STL(df[column].values, period=period, seasonal=13)
        result = stl.fit()
        
        df[f"{column}_trend"] = result.trend
        df[f"{column}_seasonal"] = result.seasonal
        df[f"{column}_residual"] = result.resid
    except Exception as e:
        print(f"Warning: Seasonal decomposition failed for {column}: {e}")
    
    return df


def calculate_anomalies(
    df: pd.DataFrame,
    column: str,
    baseline_start: Optional[str] = None,
    baseline_end: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate anomalies relative to a baseline period."""
    if baseline_start or baseline_end:
        baseline_mask = pd.Series(True, index=df.index)
        if baseline_start:
            baseline_mask &= df["date"] >= baseline_start
        if baseline_end:
            baseline_mask &= df["date"] <= baseline_end
        baseline_data = df.loc[baseline_mask, column]
    else:
        baseline_data = df[column]
    
    # Calculate climatology (monthly means)
    df["month_of_year"] = df["datetime"].dt.month
    monthly_climatology = baseline_data.groupby(df.loc[baseline_data.index, "month_of_year"]).mean()
    
    # Calculate anomalies
    df[f"{column}_anomaly"] = df.apply(
        lambda row: row[column] - monthly_climatology.get(row["month_of_year"], 0),
        axis=1
    )
    
    # Calculate standardized anomalies
    monthly_std = baseline_data.groupby(df.loc[baseline_data.index, "month_of_year"]).std()
    df[f"{column}_anomaly_std"] = df.apply(
        lambda row: (row[column] - monthly_climatology.get(row["month_of_year"], 0)) / 
                    (monthly_std.get(row["month_of_year"], 1) + 1e-10),
        axis=1
    )
    
    return df


def create_summary_statistics(df: pd.DataFrame, output_dir: Path):
    """Create and save summary statistics."""
    summary = {
        "time_period": {
            "start": df["date"].min(),
            "end": df["date"].max(),
            "n_months": len(df),
            "n_years": df["year"].nunique(),
        },
        "water_area_km2": {
            "mean": float(df["total_area_km2"].mean()),
            "std": float(df["total_area_km2"].std()),
            "min": float(df["total_area_km2"].min()),
            "max": float(df["total_area_km2"].max()),
            "median": float(df["total_area_km2"].median()),
        },
        "permanent_water_km2": {
            "mean": float(df["permanent_area_km2"].mean()),
            "std": float(df["permanent_area_km2"].std()),
        },
        "seasonal_water_km2": {
            "mean": float(df["seasonal_area_km2"].mean()),
            "std": float(df["seasonal_area_km2"].std()),
        },
        "num_water_bodies": {
            "mean": float(df["num_water_bodies"].mean()),
            "std": float(df["num_water_bodies"].std()),
        },
        "data_quality": {
            "mean_cloud_fraction": float(df["cloud_fraction"].mean()),
            "mean_valid_pixels": float(df["valid_pixels"].mean()),
        },
    }
    
    output_path = output_dir / "summary_statistics.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary Statistics:")
    print(f"  Time period: {summary['time_period']['start']} to {summary['time_period']['end']}")
    print(f"  Total months: {summary['time_period']['n_months']}")
    print(f"  Mean water area: {summary['water_area_km2']['mean']:.2f} ± {summary['water_area_km2']['std']:.2f} km²")
    print(f"  Water area range: [{summary['water_area_km2']['min']:.2f}, {summary['water_area_km2']['max']:.2f}] km²")


def create_visualizations(df: pd.DataFrame, output_dir: Path, args: argparse.Namespace):
    """Create comprehensive visualizations."""
    
    # Set up plot style
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    
    # 1. Main time series plot with multiple panels
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Total water area
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df["datetime"], df["total_area_km2"], "o-", markersize=3, linewidth=1, label="Total water area")
    if "total_area_km2_trend" in df.columns:
        ax1.plot(df["datetime"], df["total_area_km2_trend"], "r-", linewidth=2, label="Trend")
    ax1.set_ylabel("Water Area (km²)", fontsize=11, fontweight="bold")
    ax1.set_title("Surface Water Area Time Series", fontsize=13, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Water by type
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df["datetime"], df["permanent_area_km2"], label="Permanent", linewidth=1.5)
    ax2.plot(df["datetime"], df["seasonal_area_km2"], label="Seasonal", linewidth=1.5)
    ax2.plot(df["datetime"], df["ephemeral_area_km2"], label="Ephemeral", linewidth=1.5)
    ax2.set_ylabel("Water Area (km²)", fontsize=11, fontweight="bold")
    ax2.set_title("Water Area by Type", fontsize=12, fontweight="bold")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Number of water bodies
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df["datetime"], df["num_water_bodies"], "o-", markersize=3, color="darkblue")
    ax3.set_ylabel("Number of Water Bodies", fontsize=11, fontweight="bold")
    ax3.set_title("Water Body Count", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Fragmentation index
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(df["datetime"], df["fragmentation_index"], "o-", markersize=3, color="darkgreen")
    ax4.set_ylabel("Fragmentation Index", fontsize=11, fontweight="bold")
    ax4.set_title("Water Fragmentation", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Data quality
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(df["datetime"], df["cloud_fraction"] * 100, "o-", markersize=2, color="gray", alpha=0.6)
    ax5.set_ylabel("Cloud Fraction (%)", fontsize=11, fontweight="bold")
    ax5.set_title("Data Quality (Cloud Cover)", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Anomalies (if calculated)
    if "total_area_km2_anomaly_std" in df.columns:
        ax6 = fig.add_subplot(gs[3, :])
        colors = ["red" if x > 0 else "blue" for x in df["total_area_km2_anomaly_std"]]
        ax6.bar(df["datetime"], df["total_area_km2_anomaly_std"], color=colors, alpha=0.6, width=20)
        ax6.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax6.axhline(y=1, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax6.axhline(y=-1, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax6.set_ylabel("Standardized Anomaly (σ)", fontsize=11, fontweight="bold")
        ax6.set_xlabel("Date", fontsize=11, fontweight="bold")
        ax6.set_title("Water Area Anomalies", fontsize=12, fontweight="bold")
        ax6.grid(True, alpha=0.3, axis="y")
    
    # Format x-axes
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
    
    if "total_area_km2_anomaly_std" in df.columns:
        ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax6.xaxis.set_major_locator(mdates.YearLocator(5))
    
    plt.savefig(output_dir / "time_series_overview.png", bbox_inches="tight")
    plt.close()
    
    # 2. Seasonal cycle plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    monthly_mean = df.groupby("month").agg({
        "total_area_km2": ["mean", "std"],
        "permanent_area_km2": ["mean", "std"],
        "seasonal_area_km2": ["mean", "std"],
        "num_water_bodies": ["mean", "std"],
    })
    
    months = range(1, 13)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Total water area seasonal cycle
    ax = axes[0, 0]
    means = [monthly_mean.loc[m, ("total_area_km2", "mean")] if m in monthly_mean.index else 0 for m in months]
    stds = [monthly_mean.loc[m, ("total_area_km2", "std")] if m in monthly_mean.index else 0 for m in months]
    ax.plot(months, means, "o-", linewidth=2, markersize=6)
    ax.fill_between(months, 
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.3)
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_ylabel("Water Area (km²)", fontweight="bold")
    ax.set_title("Total Water Area - Seasonal Cycle", fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # Permanent water seasonal cycle
    ax = axes[0, 1]
    means = [monthly_mean.loc[m, ("permanent_area_km2", "mean")] if m in monthly_mean.index else 0 for m in months]
    stds = [monthly_mean.loc[m, ("permanent_area_km2", "std")] if m in monthly_mean.index else 0 for m in months]
    ax.plot(months, means, "o-", linewidth=2, markersize=6, color="darkblue")
    ax.fill_between(months,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.3)
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_ylabel("Water Area (km²)", fontweight="bold")
    ax.set_title("Permanent Water - Seasonal Cycle", fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # Seasonal water seasonal cycle
    ax = axes[1, 0]
    means = [monthly_mean.loc[m, ("seasonal_area_km2", "mean")] if m in monthly_mean.index else 0 for m in months]
    stds = [monthly_mean.loc[m, ("seasonal_area_km2", "std")] if m in monthly_mean.index else 0 for m in months]
    ax.plot(months, means, "o-", linewidth=2, markersize=6, color="orange")
    ax.fill_between(months,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.3)
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_ylabel("Water Area (km²)", fontweight="bold")
    ax.set_title("Seasonal Water - Seasonal Cycle", fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # Water bodies seasonal cycle
    ax = axes[1, 1]
    means = [monthly_mean.loc[m, ("num_water_bodies", "mean")] if m in monthly_mean.index else 0 for m in months]
    stds = [monthly_mean.loc[m, ("num_water_bodies", "std")] if m in monthly_mean.index else 0 for m in months]
    ax.plot(months, means, "o-", linewidth=2, markersize=6, color="darkgreen")
    ax.fill_between(months,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.3)
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_ylabel("Number of Water Bodies", fontweight="bold")
    ax.set_title("Water Body Count - Seasonal Cycle", fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "seasonal_cycles.png", bbox_inches="tight")
    plt.close()
    
    # 3. Correlation matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    
    correlation_cols = [
        "total_area_km2",
        "permanent_area_km2",
        "seasonal_area_km2",
        "ephemeral_area_km2",
        "num_water_bodies",
        "mean_body_size_km2",
        "fragmentation_index",
    ]
    
    corr_matrix = df[correlation_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Matrix of Water Metrics", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", bbox_inches="tight")
    plt.close()
    
    # 4. Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total water area distribution
    ax = axes[0, 0]
    ax.hist(df["total_area_km2"], bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(df["total_area_km2"].mean(), color="red", linestyle="--", linewidth=2, label="Mean")
    ax.axvline(df["total_area_km2"].median(), color="blue", linestyle="--", linewidth=2, label="Median")
    ax.set_xlabel("Water Area (km²)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Total Water Area Distribution", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # Water body count distribution
    ax = axes[0, 1]
    ax.hist(df["num_water_bodies"], bins=30, alpha=0.7, color="darkblue", edgecolor="black")
    ax.axvline(df["num_water_bodies"].mean(), color="red", linestyle="--", linewidth=2, label="Mean")
    ax.set_xlabel("Number of Water Bodies", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Water Body Count Distribution", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # Fragmentation index distribution
    ax = axes[1, 0]
    ax.hist(df["fragmentation_index"], bins=50, alpha=0.7, color="darkgreen", edgecolor="black")
    ax.axvline(df["fragmentation_index"].mean(), color="red", linestyle="--", linewidth=2, label="Mean")
    ax.set_xlabel("Fragmentation Index", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Fragmentation Distribution", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # Year-over-year comparison (box plot)
    ax = axes[1, 1]
    df.boxplot(column="total_area_km2", by="year", ax=ax)
    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel("Water Area (km²)", fontweight="bold")
    ax.set_title("Year-over-Year Water Area Variability", fontweight="bold")
    plt.suptitle("")  # Remove automatic title
    ax.grid(True, alpha=0.3, axis="y")
    
    # Rotate x-labels if too many years
    if df["year"].nunique() > 10:
        ax.tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "distributions.png", bbox_inches="tight")
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print(f"Surface Water Time Series Analysis")
        print(f"=" * 50)
        print(f"Data directory: {args.data_dir}")
        print(f"Statistics directory: {args.stats_dir}")
        print(f"Output directory: {args.output_dir}")
        print()
    
    # Load statistics
    stats = load_statistics(args.stats_dir, args.verbose)
    
    # Create water body masks
    water_masks = create_water_body_masks(stats["water_prob"], args)
    
    if args.verbose:
        print(f"Water body classifications:")
        print(f"  Permanent water: {np.sum(water_masks['permanent'])} pixels")
        print(f"  Seasonal water: {np.sum(water_masks['seasonal'])} pixels")
        print(f"  Ephemeral water: {np.sum(water_masks['ephemeral'])} pixels")
        print()
    
    # Find monthly mosaics
    mosaics = find_monthly_mosaics(args.data_dir, args.start_date, args.end_date)
    
    if not mosaics:
        print("Error: No monthly mosaics found!")
        return 1
    
    if args.verbose:
        print(f"Found {len(mosaics)} monthly mosaics")
        print(f"Date range: {mosaics[0][0]} to {mosaics[-1][0]}")
        print()
    
    # Extract time series
    time_series = extract_time_series(mosaics, stats, water_masks, args)
    
    # Convert to DataFrame
    df = time_series_to_dataframe(time_series)
    
    # Perform analyses
    if args.verbose:
        print("\nPerforming statistical analyses...")
    
    # Trend analysis
    metrics_to_analyze = [
        "total_area_km2",
        "permanent_area_km2",
        "seasonal_area_km2",
        "num_water_bodies",
        "fragmentation_index",
    ]
    
    trend_results = {}
    for metric in metrics_to_analyze:
        trend_results[metric] = perform_trend_analysis(df, metric, args.trend_method)
    
    # Save trend results
    with open(args.output_dir / "trend_analysis.json", "w") as f:
        json.dump(trend_results, f, indent=2)
    
    if args.verbose:
        print("\nTrend Analysis Results:")
        for metric, results in trend_results.items():
            print(f"\n  {metric}:")
            if "mann_kendall" in results:
                mk = results["mann_kendall"]
                print(f"    Mann-Kendall: {mk['trend']} (p={mk['p_value']:.4f})")
                print(f"    Sen's slope: {mk['sens_slope']:.6f} km²/month")
            if "linear" in results:
                lin = results["linear"]
                print(f"    Linear trend: slope={lin['slope']:.6f}, R²={lin['r_squared']:.4f}, p={lin['p_value']:.4f}")
    
    # Seasonal decomposition
    if len(df) >= 2 * args.seasonal_period:
        for metric in ["total_area_km2", "seasonal_area_km2"]:
            df = seasonal_decomposition(df, metric, args.seasonal_period)
    
    # Calculate anomalies
    df = calculate_anomalies(
        df,
        "total_area_km2",
        args.anomaly_baseline_start,
        args.anomaly_baseline_end,
    )
    
    # Save time series data
    csv_path = args.output_dir / "water_time_series.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nTime series data saved to: {csv_path}")
    
    # Create summary statistics
    create_summary_statistics(df, args.output_dir)
    
    # Create visualizations
    if args.verbose:
        print("\nGenerating visualizations...")
    create_visualizations(df, args.output_dir, args)
    
    # Export NetCDF if requested
    if args.export_netcdf:
        try:
            import xarray as xr
            
            # Create xarray dataset
            ds = xr.Dataset(
                {
                    "total_area_km2": (["time"], df["total_area_km2"].values),
                    "permanent_area_km2": (["time"], df["permanent_area_km2"].values),
                    "seasonal_area_km2": (["time"], df["seasonal_area_km2"].values),
                    "ephemeral_area_km2": (["time"], df["ephemeral_area_km2"].values),
                    "num_water_bodies": (["time"], df["num_water_bodies"].values),
                    "fragmentation_index": (["time"], df["fragmentation_index"].values),
                },
                coords={
                    "time": df["datetime"].values,
                },
                attrs={
                    "title": "Surface Water Time Series",
                    "description": "Monthly surface water metrics from DSWE data",
                    "permanent_threshold": args.permanent_threshold,
                    "seasonal_min": args.seasonal_min,
                    "seasonal_max": args.seasonal_max,
                },
            )
            
            nc_path = args.output_dir / "water_time_series.nc"
            ds.to_netcdf(nc_path)
            print(f"NetCDF file saved to: {nc_path}")
        except ImportError:
            print("Warning: xarray not installed, skipping NetCDF export")
    
    print(f"\n{'='*50}")
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print(f"{'='*50}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

