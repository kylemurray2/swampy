# Surface Water Time Series Analysis

## Overview

This analysis framework extracts scientifically robust metrics from long-term surface water classifications to enable comparison with climate indices (ENSO, PDO), groundwater models, GRACE, loading models, and GNSS vertical positions.

## Scientific Approach

### 1. Water Body Classification

The analysis uses multi-temporal statistics to define three water body types:

- **Permanent Water**: Areas with water probability > 0.80 (appears >80% of observations)
- **Seasonal Water**: Areas with 0.15 < water probability < 0.80 (intermittent)
- **Ephemeral Water**: Areas with 0 < water probability < 0.15 (rare, event-driven)

This classification is more robust than single-image thresholds and captures the physical processes governing water occurrence.

### 2. Extracted Metrics

#### Spatial Metrics (per month)
- **Total water area** (km²) - Overall surface water extent
- **Water area by type** - Permanent, seasonal, and ephemeral separately
- **Water area by confidence** - High-confidence vs partial water
- **Number of water bodies** - Count of discrete water features
- **Mean/max body size** - Average and largest water body areas
- **Fragmentation index** - Perimeter/area ratio indicating connectivity
- **Centroid location** - Geographic center of water mass (tracks migration)

#### Temporal Metrics
- **Valid pixel count** - Number of cloud-free observations
- **Cloud fraction** - Data quality indicator

### 3. Statistical Analyses

#### Trend Analysis
- **Linear regression** - Classical trend with R², p-value, confidence intervals
- **Mann-Kendall test** - Non-parametric trend detection (robust to outliers)
- **Sen's slope** - Robust slope estimator (median of pairwise slopes)

#### Seasonal Decomposition
- **STL decomposition** - Separates trend, seasonal, and residual components
- Reveals long-term changes vs annual cycles vs noise

#### Anomaly Detection
- **Monthly climatology** - Long-term average for each month
- **Absolute anomalies** - Deviation from climatology (km²)
- **Standardized anomalies** - Z-scores (σ units) for cross-comparison

### 4. Output Products

#### Data Files
- `water_time_series.csv` - Complete time series with all metrics
- `water_time_series.nc` - NetCDF format for geospatial analysis (optional)
- `summary_statistics.json` - Statistical summary
- `trend_analysis.json` - Detailed trend analysis results

#### Visualizations
- `time_series_overview.png` - 6-panel overview showing:
  - Total water area with trend
  - Water by type (permanent/seasonal/ephemeral)
  - Number of water bodies
  - Fragmentation index
  - Cloud fraction (data quality)
  - Standardized anomalies

- `seasonal_cycles.png` - Monthly climatology for key metrics
- `correlation_matrix.png` - Relationships between metrics
- `distributions.png` - Statistical distributions and year-over-year variability

## Usage

### Basic Usage

```bash
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/time_series \
    --verbose
```

### Advanced Options

```bash
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/time_series \
    --start-date 198401 \
    --end-date 202412 \
    --permanent-threshold 0.85 \
    --seasonal-min 0.10 \
    --seasonal-max 0.85 \
    --min-body-size 10 \
    --trend-method both \
    --seasonal-period 12 \
    --anomaly-baseline-start 199001 \
    --anomaly-baseline-end 202001 \
    --export-netcdf \
    --verbose
```

### Key Parameters

- `--permanent-threshold` (default: 0.80): Water probability threshold for permanent water
- `--seasonal-min/max` (default: 0.15/0.80): Range for seasonal water
- `--min-body-size` (default: 10): Minimum water body size in pixels (removes noise)
- `--trend-method` (default: both): "linear", "mann-kendall", or "both"
- `--seasonal-period` (default: 12): Period for seasonal decomposition (months)
- `--anomaly-baseline-start/end`: Define baseline period for anomaly calculation (default: use all data)
- `--export-netcdf`: Save as NetCDF (requires xarray)

## Interpreting Results

### Trend Analysis

**Linear Trend:**
- Slope: Rate of change (km²/month)
- R²: Fraction of variance explained (0-1)
- p-value: Statistical significance (p < 0.05 = significant)

**Mann-Kendall Test:**
- More robust to outliers and non-normal distributions
- z-statistic: Magnitude and direction of trend
- p-value: Significance (p < 0.05 = significant trend)
- Sen's slope: Robust estimate of rate of change

### Anomalies

**Standardized anomalies (σ):**
- -1 to +1 σ: Normal variability (~68% of observations)
- ±1 to ±2 σ: Moderate anomaly (~27% of observations)
- Beyond ±2 σ: Extreme event (~5% of observations)

Positive anomalies = More water than climatology  
Negative anomalies = Less water than climatology

### Fragmentation Index

Higher values = More fragmented (many small bodies, irregular edges)  
Lower values = Less fragmented (fewer large bodies, smooth edges)

Changes in fragmentation can indicate:
- Drying: Fragmentation increases as large bodies shrink and split
- Flooding: Fragmentation decreases as water bodies merge
- Land use changes: Edge complexity changes

## Integration with Other Time Series

The output CSV file (`water_time_series.csv`) is designed for direct comparison with other time series:

### Climate Indices
```python
import pandas as pd

# Load surface water time series
water = pd.read_csv("water_time_series.csv", parse_dates=["datetime"])

# Load ENSO data (example)
enso = pd.read_csv("enso_index.csv", parse_dates=["date"])

# Merge on date
combined = pd.merge(water, enso, left_on="datetime", right_on="date", how="inner")

# Calculate correlation
correlation = combined["total_area_km2_anomaly_std"].corr(combined["enso_index"])
print(f"Correlation with ENSO: {correlation:.3f}")

# Time-lagged correlation (e.g., 3-month lag)
combined["enso_lag3"] = combined["enso_index"].shift(3)
lag_corr = combined["total_area_km2_anomaly_std"].corr(combined["enso_lag3"])
print(f"Correlation with 3-month lagged ENSO: {lag_corr:.3f}")
```

### GRACE/Groundwater
```python
# Load GRACE data
grace = pd.read_csv("grace_tws.csv", parse_dates=["date"])

# Merge with water data
combined = pd.merge(water, grace, left_on="datetime", right_on="date", how="inner")

# Analyze relationship
from scipy.stats import pearsonr

# Correlation between surface water and groundwater
r, p = pearsonr(combined["total_area_km2"], combined["tws_anomaly"])
print(f"Pearson correlation: r={r:.3f}, p={p:.4f}")
```

### GNSS Vertical Positions
```python
# Load GNSS data
gnss = pd.read_csv("gnss_vertical.csv", parse_dates=["date"])

# Compare surface water loading with vertical deformation
combined = pd.merge(water, gnss, left_on="datetime", right_on="date", how="inner")

# Calculate water mass (approximate)
# Assume 1 km² water ≈ 1e9 kg (1m depth)
combined["water_mass_kg"] = combined["total_area_km2"] * 1e9

# Compare with vertical displacement
from scipy.stats import linregress
slope, intercept, r, p, stderr = linregress(
    combined["water_mass_kg"], 
    combined["vertical_displacement_mm"]
)
print(f"Loading sensitivity: {slope:.2e} mm/kg")
```

## Scientific Considerations

### 1. Data Quality

- Monitor `cloud_fraction` - high cloud cover reduces reliability
- Check `valid_pixels` - ensure sufficient observations
- Use anomaly baseline period with good data coverage

### 2. Physical Interpretation

**Permanent vs Seasonal Water:**
- Permanent water changes indicate:
  - Long-term climate trends
  - Groundwater depletion/recharge
  - Land use changes (reservoirs, irrigation)

- Seasonal water changes indicate:
  - Precipitation/drought cycles
  - Snowmelt timing shifts
  - Agricultural water use patterns

**Fragmentation:**
- Increasing fragmentation may indicate:
  - Drying/drought conditions
  - Water extraction
  - Habitat degradation

### 3. Temporal Resolution

- Monthly resolution captures:
  - Seasonal cycles (snowmelt, monsoons)
  - Inter-annual variability (El Niño, droughts)
  - Decadal trends (climate change, land use)

- Cannot capture:
  - Sub-monthly events (storm flooding)
  - Daily variations

### 4. Spatial Considerations

- Centroid migration reveals:
  - Reservoir operation patterns
  - Spatial shifts in water availability
  - River course changes

- Water body count changes indicate:
  - Reservoir filling/draining
  - Wetland creation/loss
  - Stream network expansion/contraction

## Example Workflow

```bash
# 1. Ensure you have statistics and cleaned mosaics
ls /d/surfaceWater/salina/data/usgs_dsw/stats/
ls /d/surfaceWater/salina/data/usgs_dsw/cleaned/

# 2. Run time series analysis
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/time_series_v1 \
    --permanent-threshold 0.80 \
    --trend-method both \
    --export-netcdf \
    --verbose

# 3. Review outputs
ls /d/surfaceWater/salina/analysis/time_series_v1/

# 4. Load and analyze in Python
python
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> df = pd.read_csv("/d/surfaceWater/salina/analysis/time_series_v1/water_time_series.csv")
>>> df.plot(x="datetime", y="total_area_km2")
>>> plt.show()
```

## Next Steps

Once you have the surface water time series, you can:

1. **Compare with climate indices:**
   - Calculate cross-correlations with ENSO, PDO, AMO
   - Identify time lags (e.g., 3-6 month response to El Niño)
   - Perform wavelet coherence analysis

2. **Integrate with hydrological models:**
   - Compare with groundwater model outputs
   - Validate water balance models
   - Constrain recharge/discharge estimates

3. **Analyze loading effects:**
   - Compare with GRACE TWS changes
   - Calculate surface loading from water area
   - Compare with GNSS vertical displacements
   - Run elastic loading models

4. **Publication-quality analysis:**
   - Composite analysis (El Niño vs La Niña years)
   - Attribution analysis (climate vs human factors)
   - Future projections under climate scenarios

## References

**Trend Analysis:**
- Mann, H.B. (1945). "Nonparametric tests against trend". Econometrica 13: 245-259.
- Sen, P.K. (1968). "Estimates of the regression coefficient based on Kendall's tau". JASA 63: 1379-1389.

**Seasonal Decomposition:**
- Cleveland et al. (1990). "STL: A seasonal-trend decomposition procedure based on loess". J. Official Statistics 6: 3-73.

**Surface Water Remote Sensing:**
- Jones, J.W. (2019). "Improved automated detection of subpixel-scale inundation". Remote Sensing 11(20): 2375.
- Pekel et al. (2016). "High-resolution mapping of global surface water and its long-term changes". Nature 540: 418-422.

