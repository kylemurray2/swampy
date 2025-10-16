# Surface Water Time Series Analysis

## Quick Start

This guide shows you how to extract meaningful metrics from your long-term surface water dataset and compare them with climate indices and other geophysical data.

### Step 1: Run Time Series Analysis

```bash
# Basic usage - analyze all available monthly mosaics
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/time_series \
    --verbose
```

This will create:
- `water_time_series.csv` - Complete time series data
- `summary_statistics.json` - Statistical summary
- `trend_analysis.json` - Trend analysis results
- Multiple PNG visualizations

### Step 2: Review Results

The main outputs to examine:

1. **time_series_overview.png** - Multi-panel visualization showing:
   - Total water area with trend
   - Water by type (permanent/seasonal/ephemeral)
   - Number of water bodies
   - Fragmentation index
   - Data quality indicators
   - Anomalies

2. **seasonal_cycles.png** - Monthly climatology for all metrics

3. **water_time_series.csv** - Ready for analysis in Python/R/MATLAB

### Step 3: Compare with Climate Indices

```bash
# Compare with ENSO (example)
python analysis/compare_with_climate_indices.py \
    --water-csv /d/surfaceWater/salina/analysis/time_series/water_time_series.csv \
    --climate-csv /path/to/enso_index.csv \
    --output-dir /d/surfaceWater/salina/analysis/climate_comparison \
    --max-lag 12 \
    --verbose
```

This creates:
- Cross-correlation plots (find time lags)
- Composite analysis (El Niño vs La Niña effects)
- Scatter plots with regression statistics

## Key Metrics Explained

### Spatial Metrics

- **total_area_km2**: Total surface water area
- **permanent_area_km2**: Water bodies present >80% of time
- **seasonal_area_km2**: Intermittently inundated areas
- **ephemeral_area_km2**: Rarely inundated areas
- **num_water_bodies**: Count of discrete water features
- **fragmentation_index**: Edge complexity (higher = more fragmented)

### Anomaly Metrics

- **total_area_km2_anomaly**: Deviation from monthly climatology (km²)
- **total_area_km2_anomaly_std**: Standardized anomaly (σ units)
  - -1 to +1 σ: Normal variability
  - Beyond ±2 σ: Extreme event

### Trend Components (if seasonal decomposition enabled)

- **total_area_km2_trend**: Long-term trend component
- **total_area_km2_seasonal**: Seasonal cycle component
- **total_area_km2_residual**: Residual/noise component

## Example Workflows

### Workflow 1: Basic Characterization

```bash
# Run full analysis
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/baseline \
    --trend-method both \
    --export-netcdf \
    --verbose

# Review summary
cat /d/surfaceWater/salina/analysis/baseline/summary_statistics.json
```

### Workflow 2: Drought Analysis

```bash
# Focus on ephemeral water with sensitive thresholds
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/drought_analysis \
    --permanent-threshold 0.85 \
    --seasonal-min 0.10 \
    --seasonal-max 0.85 \
    --verbose
```

### Workflow 3: Climate Teleconnection Analysis

```bash
# Step 1: Extract water metrics
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/climate_study \
    --verbose

# Step 2: Compare with ENSO
python analysis/compare_with_climate_indices.py \
    --water-csv /d/surfaceWater/salina/analysis/climate_study/water_time_series.csv \
    --climate-csv /path/to/oni_index.csv \
    --output-dir /d/surfaceWater/salina/analysis/climate_study/enso \
    --max-lag 12 \
    --verbose

# Step 3: Compare with PDO
python analysis/compare_with_climate_indices.py \
    --water-csv /d/surfaceWater/salina/analysis/climate_study/water_time_series.csv \
    --climate-csv /path/to/pdo_index.csv \
    --output-dir /d/surfaceWater/salina/analysis/climate_study/pdo \
    --max-lag 12 \
    --verbose
```

### Workflow 4: GRACE/Loading Analysis

```bash
# Compare with GRACE terrestrial water storage
python analysis/compare_with_climate_indices.py \
    --water-csv /d/surfaceWater/salina/analysis/time_series/water_time_series.csv \
    --grace-csv /path/to/grace_tws.csv \
    --output-dir /d/surfaceWater/salina/analysis/grace_comparison \
    --max-lag 6 \
    --verbose
```

## Data Format Requirements

### Climate Index CSV Format

```csv
date,index_value
2000-01-01,0.5
2000-02-01,-0.3
2000-03-01,1.2
...
```

Or:
```csv
date,oni
200001,0.5
200002,-0.3
200003,1.2
...
```

### GRACE CSV Format

```csv
datetime,tws_anomaly
2002-04-01,15.3
2002-05-01,12.1
...
```

### GNSS CSV Format

```csv
datetime,vertical_mm
2000-01-01,2.3
2000-02-01,1.8
...
```

## Python Integration Examples

### Load and Plot Time Series

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("water_time_series.csv", parse_dates=["datetime"])

# Plot total water area
plt.figure(figsize=(12, 6))
plt.plot(df["datetime"], df["total_area_km2"], linewidth=1)
plt.xlabel("Date")
plt.ylabel("Water Area (km²)")
plt.title("Surface Water Area Time Series")
plt.grid(True, alpha=0.3)
plt.show()

# Plot anomalies
plt.figure(figsize=(12, 6))
colors = ["red" if x > 0 else "blue" for x in df["total_area_km2_anomaly_std"]]
plt.bar(df["datetime"], df["total_area_km2_anomaly_std"], color=colors, alpha=0.6)
plt.axhline(y=0, color="black", linewidth=1)
plt.xlabel("Date")
plt.ylabel("Standardized Anomaly (σ)")
plt.title("Water Area Anomalies")
plt.show()
```

### Compare with Climate Index

```python
import pandas as pd
from scipy.stats import pearsonr

# Load data
water = pd.read_csv("water_time_series.csv", parse_dates=["datetime"])
enso = pd.read_csv("enso_index.csv", parse_dates=["date"])

# Merge on date
merged = pd.merge(water, enso, left_on="datetime", right_on="date", how="inner")

# Calculate correlation
r, p = pearsonr(merged["total_area_km2_anomaly_std"], merged["enso_index"])
print(f"Correlation: r = {r:.3f}, p = {p:.4f}")

# Time-lagged correlation (3-month lag)
merged["enso_lag3"] = merged["enso_index"].shift(3)
r_lag, p_lag = pearsonr(
    merged["total_area_km2_anomaly_std"].iloc[3:],
    merged["enso_lag3"].iloc[3:]
)
print(f"3-month lag correlation: r = {r_lag:.3f}, p = {p_lag:.4f}")
```

### Composite Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
water = pd.read_csv("water_time_series.csv", parse_dates=["datetime"])
enso = pd.read_csv("enso_index.csv", parse_dates=["date"])

# Merge
merged = pd.merge(water, enso, left_on="datetime", right_on="date", how="inner")

# Define El Niño and La Niña events (threshold = 0.5)
el_nino = merged[merged["enso_index"] > 0.5]
la_nina = merged[merged["enso_index"] < -0.5]
neutral = merged[(merged["enso_index"] >= -0.5) & (merged["enso_index"] <= 0.5)]

# Compare means
print(f"El Niño mean water area: {el_nino['total_area_km2'].mean():.2f} km²")
print(f"La Niña mean water area: {la_nina['total_area_km2'].mean():.2f} km²")
print(f"Neutral mean water area: {neutral['total_area_km2'].mean():.2f} km²")

# Statistical test
from scipy.stats import f_oneway
f_stat, p_value = f_oneway(
    el_nino['total_area_km2'],
    la_nina['total_area_km2'],
    neutral['total_area_km2']
)
print(f"ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}")
```

## Tips and Best Practices

### 1. Choosing Thresholds

- **Permanent water threshold (default 0.80)**:
  - Increase (0.85-0.95) for more conservative definition
  - Decrease (0.70-0.80) for regions with strong seasonal cycles

- **Seasonal water range (default 0.15-0.80)**:
  - Wider range (0.10-0.85) captures more variability
  - Narrower range (0.20-0.75) focuses on stable seasonal patterns

### 2. Anomaly Baselines

- Use 30-year baseline (1991-2020) for climate comparisons
- Avoid periods with major land use changes for baseline
- Check that baseline period has good data coverage

### 3. Trend Analysis

- Use both linear and Mann-Kendall for robustness
- Mann-Kendall is better for non-normal distributions
- Check for autocorrelation (may affect significance)

### 4. Climate Comparisons

- Test multiple time lags (0-12 months typical)
- Consider physical mechanisms (e.g., 3-6 month lag for groundwater response)
- Use standardized anomalies for comparing different variables

## Troubleshooting

### Issue: "No monthly mosaics found"

**Solution**: Check that cleaned mosaics exist:
```bash
ls /d/surfaceWater/salina/data/usgs_dsw/cleaned/
# or
ls /d/surfaceWater/salina/data/usgs_dsw/mosaics/
```

### Issue: "Metric not found in water data"

**Solution**: Check available metrics:
```python
import pandas as pd
df = pd.read_csv("water_time_series.csv")
print(df.columns.tolist())
```

Use one of: `total_area_km2`, `total_area_km2_anomaly`, `total_area_km2_anomaly_std`

### Issue: "Not enough data for seasonal decomposition"

**Solution**: Need at least 24 months of data. If you have less, disable seasonal decomposition or reduce `--seasonal-period`.

### Issue: Cross-correlation fails

**Solution**: Ensure overlapping dates:
```python
water = pd.read_csv("water_time_series.csv", parse_dates=["datetime"])
climate = pd.read_csv("climate_index.csv", parse_dates=["date"])
print(f"Water: {water['datetime'].min()} to {water['datetime'].max()}")
print(f"Climate: {climate['date'].min()} to {climate['date'].max()}")
```

## Additional Resources

- **Full documentation**: `SURFACE_WATER_ANALYSIS.md`
- **Workflow guide**: `WORKFLOW.md`
- **Example scripts**: `run_water_analysis_examples.sh`

## Citation

If you use this analysis framework for publications, please cite:

```bibtex
@software{swampy_surface_water_analysis,
  title = {Surface Water Time Series Analysis Framework},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/swampy}
}
```

## Support

For issues or questions:
1. Check this README and the full documentation
2. Review example workflows
3. Examine output JSON files for diagnostic information

