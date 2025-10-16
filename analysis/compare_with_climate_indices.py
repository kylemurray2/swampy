#!/usr/bin/env python3
"""Compare surface water time series with climate indices and other geophysical data.

This script demonstrates how to integrate surface water metrics with:
- Climate indices (ENSO, PDO, AMO, etc.)
- Groundwater/GRACE data
- GNSS vertical displacements
- Other time series

Performs cross-correlation, time-lag analysis, and composite analysis.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import signal, stats
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.style.use('seaborn-v0_8-darkgrid')


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare surface water time series with climate indices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--water-csv",
        type=Path,
        required=True,
        help="Surface water time series CSV from surface_water_time_series.py",
    )
    parser.add_argument(
        "--climate-csv",
        type=Path,
        help="Climate index CSV (columns: date, index_name)",
    )
    parser.add_argument(
        "--grace-csv",
        type=Path,
        help="GRACE TWS CSV (columns: date, tws_anomaly)",
    )
    parser.add_argument(
        "--gnss-csv",
        type=Path,
        help="GNSS vertical displacement CSV (columns: date, vertical_mm)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=12,
        help="Maximum time lag in months for cross-correlation",
    )
    parser.add_argument(
        "--water-metric",
        default="total_area_km2_anomaly_std",
        help="Water metric to use for comparison",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    return parser.parse_args(argv)


def load_water_data(csv_path: Path, metric: str) -> pd.DataFrame:
    """Load surface water time series."""
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # Ensure the metric exists
    if metric not in df.columns:
        available = [col for col in df.columns if "area" in col or "anomaly" in col]
        raise ValueError(
            f"Metric '{metric}' not found in water data. Available metrics: {available}"
        )
    
    return df[["datetime", metric]].rename(columns={metric: "water_metric"})


def load_climate_index(csv_path: Path, index_col: Optional[str] = None) -> pd.DataFrame:
    """Load climate index data.
    
    Expected format:
    - CSV with 'date' column (YYYY-MM-DD or YYYYMM)
    - Index value column (auto-detected or specified)
    """
    df = pd.read_csv(csv_path)
    
    # Parse date column
    if "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        # Try first column
        df["datetime"] = pd.to_datetime(df.iloc[:, 0])
    
    # Identify index column
    if index_col is None:
        # Find numeric column that's not the date
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in climate data")
        index_col = numeric_cols[0]
    
    df = df[["datetime", index_col]].rename(columns={index_col: "climate_index"})
    return df.sort_values("datetime").reset_index(drop=True)


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cross-correlation at different time lags."""
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = []
    
    for lag in lags:
        if lag < 0:
            # x leads y
            corr = pearsonr(x[:lag], y[-lag:])[0]
        elif lag > 0:
            # y leads x
            corr = pearsonr(x[lag:], y[:-lag])[0]
        else:
            # No lag
            corr = pearsonr(x, y)[0]
        correlations.append(corr)
    
    return lags, np.array(correlations)


def composite_analysis(
    data: pd.DataFrame,
    index_col: str,
    metric_col: str,
    threshold: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    """Perform composite analysis based on index phases.
    
    Separates data into positive/negative index phases and compares metrics.
    """
    # Standardize index
    index_std = (data[index_col] - data[index_col].mean()) / data[index_col].std()
    
    # Define phases
    positive_phase = data[index_std > threshold].copy()
    negative_phase = data[index_std < -threshold].copy()
    neutral_phase = data[(index_std >= -threshold) & (index_std <= threshold)].copy()
    
    return {
        "positive": positive_phase,
        "negative": negative_phase,
        "neutral": neutral_phase,
    }


def plot_time_series_comparison(
    water: pd.DataFrame,
    other: pd.DataFrame,
    other_name: str,
    output_dir: Path,
):
    """Plot time series comparison."""
    # Merge datasets
    merged = pd.merge(water, other, on="datetime", how="inner")
    
    if len(merged) == 0:
        print(f"Warning: No overlapping dates between water and {other_name} data")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top panel: Both time series
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    l1 = ax1.plot(merged["datetime"], merged["water_metric"], "b-", linewidth=1.5, label="Surface Water")
    l2 = ax1_twin.plot(merged["datetime"], merged[other.columns[1]], "r-", linewidth=1.5, label=other_name)
    
    ax1.set_ylabel("Water Metric (σ)", color="b", fontweight="bold")
    ax1_twin.set_ylabel(other_name, color="r", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1_twin.tick_params(axis="y", labelcolor="r")
    
    # Combined legend
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left")
    
    ax1.set_title(f"Surface Water vs {other_name}", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Scatter plot
    ax2 = axes[1]
    ax2.scatter(merged[other.columns[1]], merged["water_metric"], alpha=0.6, s=20)
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        merged[other.columns[1]], merged["water_metric"]
    )
    x_line = np.array([merged[other.columns[1]].min(), merged[other.columns[1]].max()])
    y_line = slope * x_line + intercept
    ax2.plot(x_line, y_line, "r--", linewidth=2, 
             label=f"R={r_value:.3f}, p={p_value:.4f}")
    
    ax2.set_xlabel(other_name, fontweight="bold")
    ax2.set_ylabel("Water Metric (σ)", fontweight="bold")
    ax2.set_title("Correlation Analysis", fontsize=12, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"comparison_{other_name.lower().replace(' ', '_')}.png", 
                dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Correlation with {other_name}: r={r_value:.3f}, p={p_value:.4f}")


def plot_cross_correlation(
    lags: np.ndarray,
    correlations: np.ndarray,
    other_name: str,
    output_dir: Path,
):
    """Plot cross-correlation at different lags."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot correlation vs lag
    ax.bar(lags, correlations, width=0.8, alpha=0.7)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=1)
    
    # Add significance lines (approximate)
    n_points = 100  # approximate, should use actual sample size
    sig_level = 1.96 / np.sqrt(n_points)
    ax.axhline(y=sig_level, color="r", linestyle="--", linewidth=1, alpha=0.5, label="95% confidence")
    ax.axhline(y=-sig_level, color="r", linestyle="--", linewidth=1, alpha=0.5)
    
    # Find maximum correlation
    max_idx = np.argmax(np.abs(correlations))
    max_lag = lags[max_idx]
    max_corr = correlations[max_idx]
    
    ax.plot(max_lag, max_corr, "ro", markersize=10, 
            label=f"Max |r|={abs(max_corr):.3f} at lag={max_lag} months")
    
    ax.set_xlabel("Lag (months)", fontweight="bold")
    ax.set_ylabel("Correlation Coefficient", fontweight="bold")
    ax.set_title(f"Cross-Correlation: Surface Water vs {other_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    # Add interpretation text
    if max_lag < 0:
        interpretation = f"Surface water leads {other_name} by {abs(max_lag)} months"
    elif max_lag > 0:
        interpretation = f"{other_name} leads surface water by {max_lag} months"
    else:
        interpretation = "Surface water and index are in phase"
    
    ax.text(0.02, 0.98, interpretation, transform=ax.transAxes, 
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / f"cross_correlation_{other_name.lower().replace(' ', '_')}.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nCross-correlation with {other_name}:")
    print(f"  Maximum correlation: {max_corr:.3f} at lag {max_lag} months")
    print(f"  {interpretation}")


def plot_composite_analysis(
    composites: Dict[str, pd.DataFrame],
    metric_col: str,
    index_name: str,
    output_dir: Path,
):
    """Plot composite analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot comparison
    ax1 = axes[0]
    data_to_plot = [
        composites["negative"][metric_col].dropna(),
        composites["neutral"][metric_col].dropna(),
        composites["positive"][metric_col].dropna(),
    ]
    labels = ["Negative Phase", "Neutral", "Positive Phase"]
    
    bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ["blue", "gray", "red"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_ylabel("Water Metric (σ)", fontweight="bold")
    ax1.set_title(f"Surface Water During {index_name} Phases", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Statistical test
    from scipy.stats import f_oneway
    f_stat, p_value = f_oneway(*data_to_plot)
    ax1.text(0.02, 0.98, f"ANOVA: F={f_stat:.2f}, p={p_value:.4f}", 
             transform=ax1.transAxes, fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    # Monthly climatology by phase
    ax2 = axes[1]
    for phase, color, label in [("negative", "blue", "Negative Phase"),
                                  ("neutral", "gray", "Neutral"),
                                  ("positive", "red", "Positive Phase")]:
        if len(composites[phase]) > 0:
            composites[phase]["month"] = composites[phase]["datetime"].dt.month
            monthly = composites[phase].groupby("month")[metric_col].mean()
            ax2.plot(monthly.index, monthly.values, "o-", color=color, 
                    linewidth=2, markersize=6, label=label)
    
    ax2.set_xlabel("Month", fontweight="bold")
    ax2.set_ylabel("Water Metric (σ)", fontweight="bold")
    ax2.set_title("Seasonal Cycle by Phase", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"composite_{index_name.lower().replace(' ', '_')}.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nComposite Analysis ({index_name}):")
    print(f"  Negative phase mean: {composites['negative'][metric_col].mean():.3f} (n={len(composites['negative'])})")
    print(f"  Neutral phase mean: {composites['neutral'][metric_col].mean():.3f} (n={len(composites['neutral'])})")
    print(f"  Positive phase mean: {composites['positive'][metric_col].mean():.3f} (n={len(composites['positive'])})")
    print(f"  ANOVA p-value: {p_value:.4f}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Surface Water vs Climate/Geophysical Data Comparison")
    print("=" * 60)
    
    # Load water data
    water = load_water_data(args.water_csv, args.water_metric)
    print(f"\nLoaded water data: {len(water)} months")
    print(f"  Date range: {water['datetime'].min()} to {water['datetime'].max()}")
    print(f"  Using metric: {args.water_metric}")
    
    # Compare with climate index
    if args.climate_csv and args.climate_csv.exists():
        print(f"\n{'='*60}")
        print("Analyzing Climate Index Relationship")
        print("=" * 60)
        
        climate = load_climate_index(args.climate_csv)
        index_name = args.climate_csv.stem.upper().replace("_", " ")
        
        print(f"Loaded {index_name}: {len(climate)} months")
        
        # Time series comparison
        plot_time_series_comparison(water, climate, index_name, args.output_dir)
        
        # Cross-correlation analysis
        merged = pd.merge(water, climate, on="datetime", how="inner")
        if len(merged) >= args.max_lag * 2:
            lags, correlations = cross_correlation(
                merged["water_metric"].values,
                merged["climate_index"].values,
                args.max_lag
            )
            plot_cross_correlation(lags, correlations, index_name, args.output_dir)
        
        # Composite analysis
        composites = composite_analysis(
            merged, "climate_index", "water_metric", threshold=0.5
        )
        plot_composite_analysis(composites, "water_metric", index_name, args.output_dir)
    
    # Compare with GRACE
    if args.grace_csv and args.grace_csv.exists():
        print(f"\n{'='*60}")
        print("Analyzing GRACE TWS Relationship")
        print("=" * 60)
        
        grace = pd.read_csv(args.grace_csv, parse_dates=["datetime"])
        grace_metric = [col for col in grace.columns if col != "datetime"][0]
        grace = grace[["datetime", grace_metric]].rename(columns={grace_metric: "grace_tws"})
        
        print(f"Loaded GRACE data: {len(grace)} months")
        
        plot_time_series_comparison(water, grace, "GRACE TWS", args.output_dir)
        
        # Cross-correlation
        merged = pd.merge(water, grace, on="datetime", how="inner")
        if len(merged) >= args.max_lag * 2:
            lags, correlations = cross_correlation(
                merged["water_metric"].values,
                merged["grace_tws"].values,
                args.max_lag
            )
            plot_cross_correlation(lags, correlations, "GRACE TWS", args.output_dir)
    
    # Compare with GNSS
    if args.gnss_csv and args.gnss_csv.exists():
        print(f"\n{'='*60}")
        print("Analyzing GNSS Vertical Displacement Relationship")
        print("=" * 60)
        
        gnss = pd.read_csv(args.gnss_csv, parse_dates=["datetime"])
        gnss_metric = [col for col in gnss.columns if col != "datetime"][0]
        gnss = gnss[["datetime", gnss_metric]].rename(columns={gnss_metric: "gnss_vertical"})
        
        print(f"Loaded GNSS data: {len(gnss)} months")
        
        plot_time_series_comparison(water, gnss, "GNSS Vertical", args.output_dir)
        
        # Cross-correlation
        merged = pd.merge(water, gnss, on="datetime", how="inner")
        if len(merged) >= args.max_lag * 2:
            lags, correlations = cross_correlation(
                merged["water_metric"].values,
                merged["gnss_vertical"].values,
                args.max_lag
            )
            plot_cross_correlation(lags, correlations, "GNSS Vertical", args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

