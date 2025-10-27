# Surface Water Time Series Analysis - Scientific Framework

## Overview

This framework provides a scientifically robust approach to analyzing decades of surface water classifications from your USGS DSWE pipeline. It's designed specifically for comparing surface water changes with climate indices (ENSO, PDO), groundwater models, GRACE, loading models, and GNSS vertical positions.

## What I've Created for You

### 1. Main Analysis Script: `surface_water_time_series.py`

**Purpose**: Extract meaningful physical metrics from your monthly mosaics

**Key Features**:
- Defines water bodies using multi-temporal statistics (permanent, seasonal, ephemeral)
- Extracts 14+ spatial and temporal metrics per month
- Performs robust trend analysis (linear regression + Mann-Kendall test)
- Seasonal decomposition (STL) to separate trend, seasonal, and residual components
- Anomaly detection with customizable baseline periods
- Exports analysis-ready CSV and optional NetCDF

**Outputs**:
- `water_time_series.csv` - Complete time series (ready for analysis)
- `summary_statistics.json` - Statistical overview
- `trend_analysis.json` - Detailed trend results
- 4 publication-quality visualizations

### 2. Climate Comparison Script: `compare_with_climate_indices.py`

**Purpose**: Compare surface water with external time series

**Capabilities**:
- Cross-correlation analysis (finds optimal time lags)
- Composite analysis (El Niño vs La Niña effects)
- Scatter plots with regression statistics
- Supports multiple data types (climate indices, GRACE, GNSS)

**Outputs**:
- Time series comparison plots
- Cross-correlation plots with lag analysis
- Composite analysis visualizations
- Statistical summaries

### 3. Documentation

- `SURFACE_WATER_ANALYSIS.md` - Complete scientific documentation
- `README_TIME_SERIES.md` - Quick start guide with examples
- `run_water_analysis_examples.sh` - 7 example workflows

## Scientific Approach

### Water Body Classification Strategy

Instead of arbitrary thresholds, we use **multi-temporal statistics** to define physically meaningful water categories:

1. **Permanent Water** (prob > 0.80)
   - Present in >80% of observations
   - Represents lakes, reservoirs, perennial rivers
   - Sensitive to long-term climate trends and groundwater changes

2. **Seasonal Water** (0.15 < prob < 0.80)
   - Intermittently inundated
   - Represents seasonal wetlands, ephemeral streams
   - Responds to precipitation, snowmelt, irrigation

3. **Ephemeral Water** (prob < 0.15)
   - Rarely inundated
   - Represents flood zones, playas, storm runoff
   - Tracks extreme events

**Why this matters**: Different water types respond to different drivers. Permanent water tracks long-term trends, seasonal water tracks annual cycles, ephemeral water tracks extreme events.

### Metrics Designed for Physical Interpretation

#### Area Metrics
- **Total water area**: Overall surface water extent
- **By type**: Permanent vs seasonal vs ephemeral (different physical processes)
- **By confidence**: High-confidence vs partial (data quality indicator)

#### Spatial Structure Metrics
- **Number of water bodies**: Connectivity and fragmentation
- **Mean/max body size**: Distribution of water features
- **Fragmentation index**: Edge complexity
  - Increases during drying (large bodies split)
  - Decreases during flooding (bodies merge)
  - Sensitive to land use changes

#### Centroid Tracking
- **Geographic center of water mass**: Tracks spatial migration
  - Reservoir operations shift centroid
  - Groundwater depletion shifts water distribution
  - Climate trends affect spatial patterns

### Robust Statistical Methods

#### Trend Detection
1. **Linear Regression**
   - Classical approach with R², p-value
   - Good for consistent trends
   - Sensitive to outliers

2. **Mann-Kendall Test**
   - Non-parametric (no distribution assumptions)
   - Robust to outliers
   - Better for environmental data

3. **Sen's Slope**
   - Robust trend estimator (median of pairwise slopes)
   - Resistant to outliers and gaps

**Best practice**: Report both methods for robustness

#### Seasonal Decomposition (STL)

Separates time series into:
- **Trend**: Long-term changes (climate, land use)
- **Seasonal**: Annual cycle (precipitation, snowmelt)
- **Residual**: Noise and events (droughts, floods)

This reveals what drives variability:
- Strong trend + weak seasonal = climate change dominant
- Weak trend + strong seasonal = precipitation-driven
- Large residuals = event-driven (droughts, floods)

#### Anomaly Analysis

**Standardized anomalies** (σ units) enable cross-comparison:
- Compare water area anomalies with ENSO index
- Compare with GRACE TWS anomalies
- Compare with precipitation anomalies

**Physical interpretation**:
- ±1σ: Normal variability (68% of time)
- ±2σ: Unusual (5% of time)
- Beyond ±2σ: Extreme event (<1% of time)

## Integration with Other Data

### Climate Indices (ENSO, PDO, AMO)

**Cross-correlation reveals**:
- **Time lags**: How long after El Niño does water area peak?
  - 0-3 months: Direct precipitation response
  - 3-6 months: Snowmelt/runoff response
  - 6-12 months: Groundwater response

**Composite analysis reveals**:
- El Niño vs La Niña water area differences
- PDO phase effects on seasonal cycles
- Teleconnection patterns

**Example findings**:
- "Surface water area increases by 15% during El Niño events with a 3-month lag"
- "PDO positive phase enhances winter water storage by 20%"

### GRACE Terrestrial Water Storage

**Relationship reveals**:
- Surface water contribution to total TWS
- Groundwater vs surface water changes
- Water storage mechanisms

**Cross-correlation with GRACE**:
- Strong correlation (r > 0.7): Surface water dominates TWS signal
- Moderate correlation (0.4 < r < 0.7): Mixed surface/groundwater
- Weak correlation (r < 0.4): Groundwater dominates

### GNSS Vertical Displacement

**Loading analysis**:
- Surface water mass change → elastic loading
- Compare water area with vertical displacement
- Estimate loading sensitivity (mm displacement per km² water)

**Expected relationships**:
- Positive correlation: Surface loading from water
- Time lag: Elastic vs viscoelastic response
- Spatial patterns: Near-field vs far-field deformation

### Groundwater Models

**Model validation**:
- Compare surface water changes with model outputs
- Constrain recharge/discharge estimates
- Identify surface-groundwater exchange zones

**Complementary information**:
- Surface water: Direct observation
- Models: Subsurface processes
- Together: Complete water balance

## Example Research Questions

### 1. Climate Teleconnections
**Question**: How does ENSO affect surface water in your region?

**Approach**:
```bash
# Extract water time series
python surface_water_time_series.py --data-dir ... --output-dir enso_study

# Compare with ONI (Oceanic Niño Index)
python compare_with_climate_indices.py \
    --water-csv enso_study/water_time_series.csv \
    --climate-csv oni_index.csv \
    --max-lag 12
```

**Look for**:
- Significant cross-correlation at specific lag
- Composite differences (El Niño mean vs La Niña mean)
- Seasonal modulation (stronger in certain months?)

### 2. Groundwater Depletion
**Question**: Is permanent water declining due to groundwater pumping?

**Approach**:
```bash
# Run analysis focusing on permanent water
python surface_water_time_series.py \
    --data-dir ... \
    --output-dir gw_depletion \
    --permanent-threshold 0.85 \
    --trend-method both
```

**Check**:
- `trend_analysis.json` for permanent_area_km2
- Negative trend with p < 0.05 = significant decline
- Compare with groundwater level data

### 3. Drought Characterization
**Question**: How does drought affect water fragmentation?

**Approach**:
```bash
# Extract fragmentation metrics
python surface_water_time_series.py --data-dir ... --output-dir drought_study

# Load in Python and analyze
import pandas as pd
df = pd.read_csv("drought_study/water_time_series.csv")
drought_years = df[df['total_area_km2_anomaly_std'] < -1.5]  # Severe drought
print(f"Fragmentation during drought: {drought_years['fragmentation_index'].mean()}")
```

### 4. Loading Model Validation
**Question**: Does surface water loading match GNSS observations?

**Approach**:
```bash
# Compare with GNSS vertical
python compare_with_climate_indices.py \
    --water-csv water_time_series.csv \
    --gnss-csv gnss_vertical.csv \
    --max-lag 3
```

**Calculate loading**:
- Assume 1 km² water @ 1m depth = 1e9 kg
- Expected displacement ~ 0.1-1 mm per km² (depends on distance)
- Compare observed vs predicted

### 5. Seasonal Water vs Precipitation
**Question**: How does seasonal water respond to precipitation?

**Approach**:
```python
import pandas as pd
water = pd.read_csv("water_time_series.csv", parse_dates=["datetime"])
precip = pd.read_csv("precipitation.csv", parse_dates=["date"])

# Merge and analyze
merged = pd.merge(water, precip, left_on="datetime", right_on="date")

# Seasonal correlation
from scipy.stats import pearsonr
r, p = pearsonr(merged["seasonal_area_km2"], merged["precipitation"])
print(f"Correlation: r={r:.3f}, p={p:.4f}")
```

## Visualization Strategy

### 1. Time Series Overview (`time_series_overview.png`)
- **Panel 1**: Total water area with trend (main story)
- **Panel 2**: Water by type (process attribution)
- **Panel 3**: Water body count (fragmentation indicator)
- **Panel 4**: Fragmentation index (connectivity metric)
- **Panel 5**: Cloud fraction (data quality)
- **Panel 6**: Standardized anomalies (event detection)

### 2. Seasonal Cycles (`seasonal_cycles.png`)
- Monthly climatology for all metrics
- Shows typical annual cycle
- Identifies peak and low water months
- Baseline for anomaly detection

### 3. Correlation Matrix (`correlation_matrix.png`)
- Relationships between metrics
- Identifies redundant metrics
- Reveals composite indicators

### 4. Distributions (`distributions.png`)
- Statistical properties
- Identifies outliers
- Year-over-year variability

### 5. Climate Comparisons
- Time series overlay (visual correlation)
- Scatter plots (regression analysis)
- Cross-correlation (lag detection)
- Composite analysis (phase differences)

## Publication-Ready Results

### Tables

**Table 1: Trend Analysis**
```
Metric                  | Linear Trend      | Mann-Kendall     | Interpretation
------------------------|-------------------|------------------|------------------
Total water area        | -0.5 km²/yr       | Decreasing       | Significant decline
                        | (p=0.003)         | (p=0.001)        |
Permanent water         | -0.3 km²/yr       | Decreasing       | Groundwater depletion?
                        | (p=0.012)         | (p=0.008)        |
Seasonal water          | -0.2 km²/yr       | No trend         | Precipitation-driven
                        | (p=0.15)          | (p=0.21)         |
Fragmentation           | +0.001/yr         | Increasing       | Water bodies shrinking
                        | (p=0.008)         | (p=0.005)        |
```

**Table 2: Climate Correlations**
```
Climate Index | Correlation | Lag (months) | p-value | Interpretation
--------------|-------------|--------------|---------|------------------
ENSO (ONI)    | 0.42        | 3            | <0.001  | El Niño increases water
PDO           | 0.28        | 6            | 0.003   | Weak positive relationship
AMO           | -0.15       | 0            | 0.08    | No significant relationship
```

### Figures

**Figure 1**: Long-term surface water area showing:
- Total area time series with trend
- Permanent vs seasonal water breakdown
- Drought/flood events highlighted

**Figure 2**: Seasonal cycle analysis showing:
- Monthly climatology
- Inter-annual variability (error bars)
- Trend in seasonal amplitude

**Figure 3**: Climate teleconnection showing:
- Cross-correlation with ENSO
- Composite analysis (El Niño vs La Niña)
- Spatial patterns (map of changes)

**Figure 4**: Multi-variable comparison showing:
- Surface water, GRACE TWS, precipitation
- Standardized anomalies for comparison
- Attribution analysis

## Best Practices

### 1. Data Quality
- Monitor cloud fraction (exclude months with >50% clouds)
- Check valid pixel count (ensure sufficient coverage)
- Use conservative baselines for anomalies (30 years, good data)

### 2. Threshold Selection
- **Permanent water**: 0.80-0.90 (region-dependent)
- **Seasonal range**: 0.10-0.80 (adjust for climate)
- **Min body size**: 10-20 pixels (removes noise)

Test sensitivity:
```bash
# Run with different thresholds
for thresh in 0.75 0.80 0.85 0.90; do
    python surface_water_time_series.py \
        --permanent-threshold $thresh \
        --output-dir results_threshold_${thresh}
done
# Compare results
```

### 3. Statistical Significance
- Always report p-values with trends
- Use both parametric and non-parametric tests
- Account for autocorrelation in significance tests
- Consider field significance for multiple comparisons

### 4. Physical Interpretation
- Don't just report correlations, explain mechanisms
- Consider time scales (daily, seasonal, inter-annual, decadal)
- Validate with independent data when possible
- Acknowledge uncertainties and limitations

## Next Steps for Your Analysis

1. **Run baseline analysis**:
   ```bash
   python analysis/surface_water_time_series.py \
       --data-dir /d/surfaceWater/salina/data/usgs_dsw \
       --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
       --output-dir /d/surfaceWater/salina/analysis/baseline \
       --export-netcdf --verbose
   ```

2. **Review results**:
   - Check `summary_statistics.json`
   - Examine `time_series_overview.png`
   - Review `trend_analysis.json`

3. **Download climate indices**:
   - ENSO (ONI): https://www.cpc.ncep.noaa.gov/data/indices/
   - PDO: https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/
   - AMO: https://www.psl.noaa.gov/data/correlation/amon.us.long.data

4. **Run climate comparison**:
   ```bash
   python analysis/compare_with_climate_indices.py \
       --water-csv baseline/water_time_series.csv \
       --climate-csv oni_index.csv \
       --output-dir climate_comparison \
       --verbose
   ```

5. **Integrate with your other data** (GRACE, GNSS, groundwater models)

6. **Publish results**! You now have the tools for a comprehensive surface water analysis.

## Support & Resources

- **Quick Start**: `analysis/README_TIME_SERIES.md`
- **Full Documentation**: `analysis/SURFACE_WATER_ANALYSIS.md`
- **Example Scripts**: `analysis/run_water_analysis_examples.sh`
- **Workflow Guide**: `analysis/WORKFLOW.md`

## References

Key papers that inform this approach:

1. **Surface Water Mapping**:
   - Pekel et al. (2016). "High-resolution mapping of global surface water". Nature.
   - Jones (2019). "Improved automated detection of subpixel-scale inundation". Remote Sensing.

2. **Trend Analysis**:
   - Mann (1945). "Nonparametric tests against trend". Econometrica.
   - Sen (1968). "Estimates of regression coefficient based on Kendall's tau". JASA.

3. **Seasonal Decomposition**:
   - Cleveland et al. (1990). "STL: Seasonal-trend decomposition". J. Official Statistics.

4. **Climate Teleconnections**:
   - Trenberth (1997). "The definition of El Niño". BAMS.
   - Mantua & Hare (2002). "The Pacific Decadal Oscillation". J. Oceanography.

5. **Loading & GNSS**:
   - Blewitt et al. (2001). "A new global mode of Earth deformation". Science.
   - Chanard et al. (2014). "Toward a global horizontal and vertical elastic load deformation model". JGR.

