# Surface Water Time Series Analysis - Implementation Summary

## What Has Been Created

I've implemented a comprehensive scientific framework for analyzing your decades-long surface water time series and comparing it with climate indices and other geophysical data. Here's what you now have:

## 📁 New Files Created

### Core Analysis Scripts

1. **`surface_water_time_series.py`** (843 lines)
   - Main analysis script
   - Extracts 14+ metrics from monthly mosaics
   - Performs trend analysis, seasonal decomposition, anomaly detection
   - Generates publication-quality visualizations
   - Exports CSV and optional NetCDF

2. **`compare_with_climate_indices.py`** (520 lines)
   - Cross-correlation analysis with time lags
   - Composite analysis (e.g., El Niño vs La Niña)
   - Supports climate indices, GRACE, GNSS data
   - Creates comparison visualizations

### Documentation

3. **`SURFACE_WATER_ANALYSIS.md`**
   - Complete scientific documentation
   - Detailed methodology explanations
   - Integration examples with climate data
   - Statistical interpretation guide

4. **`README_TIME_SERIES.md`**
   - Quick start guide
   - Example workflows
   - Troubleshooting tips
   - Python integration examples

5. **`SURFACE_WATER_SCIENCE.md`** (Main overview)
   - High-level scientific framework
   - Research question examples
   - Publication-ready result templates
   - Best practices and next steps

6. **`run_water_analysis_examples.sh`**
   - 7 example workflow scripts
   - Different analysis configurations
   - Ready to run with your data

7. **`SUMMARY.md`** (this file)
   - Quick reference guide

## 🔬 Scientific Approach

### Water Body Classification

Instead of arbitrary thresholds, uses **multi-temporal statistics**:

- **Permanent Water** (prob > 0.80): Lakes, reservoirs → tracks long-term trends
- **Seasonal Water** (0.15-0.80): Wetlands, seasonal streams → tracks annual cycles  
- **Ephemeral Water** (< 0.15): Playas, flood zones → tracks extreme events

### Extracted Metrics (per month)

**Spatial Metrics:**
- Total water area (km²)
- Water area by type (permanent/seasonal/ephemeral)
- Water area by confidence (high/partial)
- Number of water bodies
- Mean/max body size
- Fragmentation index (edge/area ratio)
- Centroid location (tracks migration)

**Temporal Metrics:**
- Valid pixel count
- Cloud fraction

### Statistical Analyses

1. **Trend Analysis**
   - Linear regression (slope, R², p-value)
   - Mann-Kendall test (non-parametric, robust)
   - Sen's slope (robust estimator)

2. **Seasonal Decomposition (STL)**
   - Trend component (long-term changes)
   - Seasonal component (annual cycle)
   - Residual component (noise/events)

3. **Anomaly Detection**
   - Absolute anomalies (km²)
   - Standardized anomalies (σ units) for cross-comparison

### Integration Capabilities

- **Climate Indices**: ENSO, PDO, AMO with time-lag analysis
- **GRACE**: TWS comparison, surface/groundwater partitioning
- **GNSS**: Loading analysis, elastic deformation
- **Groundwater Models**: Validation and constraint

## 📊 Output Products

### From `surface_water_time_series.py`:

```
output_dir/
├── water_time_series.csv          # Complete time series data
├── summary_statistics.json        # Statistical summary
├── trend_analysis.json            # Detailed trend results
├── time_series_overview.png       # 6-panel visualization
├── seasonal_cycles.png            # Monthly climatology
├── correlation_matrix.png         # Metric relationships
├── distributions.png              # Statistical distributions
└── water_time_series.nc          # NetCDF (optional)
```

### From `compare_with_climate_indices.py`:

```
output_dir/
├── comparison_[index].png         # Time series comparison
├── cross_correlation_[index].png  # Lag analysis
└── composite_[index].png          # Phase comparison
```

## 🚀 Quick Start

### Step 1: Run Basic Analysis

```bash
cd /home/km/Software/swampy

python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/time_series \
    --verbose
```

**This will:**
- Process all monthly mosaics in `cleaned/YYYYMM/` or `mosaics/YYYYMM/`
- Use statistics from `stats/` to classify water bodies
- Create comprehensive time series analysis
- Generate 4 publication-quality figures
- Export analysis-ready CSV

### Step 2: Compare with Climate Index

```bash
# First, download ENSO ONI index from:
# https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

# Then run comparison
python analysis/compare_with_climate_indices.py \
    --water-csv /d/surfaceWater/salina/analysis/time_series/water_time_series.csv \
    --climate-csv /path/to/oni_index.csv \
    --output-dir /d/surfaceWater/salina/analysis/enso_comparison \
    --max-lag 12 \
    --verbose
```

**This will:**
- Find optimal time lag between ENSO and surface water
- Perform composite analysis (El Niño vs La Niña)
- Generate comparison visualizations
- Calculate correlation statistics

### Step 3: Review Results

```bash
# View summary statistics
cat /d/surfaceWater/salina/analysis/time_series/summary_statistics.json

# View trend analysis
cat /d/surfaceWater/salina/analysis/time_series/trend_analysis.json

# Open visualizations
xdg-open /d/surfaceWater/salina/analysis/time_series/time_series_overview.png
```

## 📈 Example Use Cases

### 1. Characterize Long-Term Trends

```bash
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/trends \
    --trend-method both \
    --export-netcdf \
    --verbose
```

**Outputs:**
- Trend significance (p-values)
- Rate of change (km²/year)
- Confidence intervals

### 2. Analyze ENSO Teleconnections

```bash
# Extract water metrics
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/enso_study \
    --verbose

# Compare with ENSO
python analysis/compare_with_climate_indices.py \
    --water-csv /d/surfaceWater/salina/analysis/enso_study/water_time_series.csv \
    --climate-csv oni_index.csv \
    --output-dir /d/surfaceWater/salina/analysis/enso_study/results \
    --verbose
```

**Reveals:**
- Time lag (e.g., "3-month lag between El Niño and peak water")
- Magnitude (e.g., "15% increase during El Niño events")
- Statistical significance

### 3. Drought Impact Analysis

```bash
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/drought \
    --permanent-threshold 0.85 \
    --seasonal-min 0.10 \
    --verbose
```

**Identifies:**
- Drought years (negative anomalies > 2σ)
- Fragmentation increase during droughts
- Permanent water loss

### 4. Compare Specific Time Periods

```bash
# Pre-2000 analysis
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/pre_2000 \
    --end-date 199912 \
    --verbose

# Post-2000 analysis
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/post_2000 \
    --start-date 200001 \
    --verbose
```

## 🔧 Key Parameters

### Water Classification Thresholds

```bash
--permanent-threshold 0.80    # Default: 80% probability for permanent water
--seasonal-min 0.15           # Lower bound for seasonal water
--seasonal-max 0.80           # Upper bound for seasonal water
--min-body-size 10            # Minimum water body size (pixels)
```

**Adjust for your region:**
- **Arid regions**: Increase permanent threshold (0.85-0.90)
- **Humid regions**: Decrease permanent threshold (0.70-0.80)
- **Noisy data**: Increase min-body-size (15-20)

### Analysis Options

```bash
--trend-method both           # "linear", "mann-kendall", or "both"
--seasonal-period 12          # Months per seasonal cycle
--export-netcdf              # Save as NetCDF (requires xarray)
```

### Anomaly Baseline

```bash
--anomaly-baseline-start 199001   # Start of baseline period
--anomaly-baseline-end 202001     # End of baseline period
```

**Use 30-year baseline** for climate comparisons (e.g., 1991-2020)

## 📚 Documentation Files

- **`SURFACE_WATER_SCIENCE.md`** ← **Start here** for overview
- **`README_TIME_SERIES.md`** ← Quick start guide
- **`SURFACE_WATER_ANALYSIS.md`** ← Detailed methodology
- **`WORKFLOW.md`** ← Overall pipeline workflow

## 🔬 Scientific Robustness

### What Makes This Scientifically Sound:

1. **Multi-temporal water classification** - Uses statistics, not arbitrary thresholds
2. **Multiple trend tests** - Both parametric and non-parametric
3. **Seasonal decomposition** - Separates different variability modes
4. **Standardized anomalies** - Enables cross-comparison with other variables
5. **Time-lag analysis** - Accounts for delayed responses
6. **Composite analysis** - Tests physical mechanisms (El Niño vs La Niña)
7. **Uncertainty quantification** - Reports p-values, confidence intervals

### Statistical Tests Implemented:

- Linear regression (trend, R², p-value)
- Mann-Kendall test (non-parametric trend)
- Sen's slope (robust trend estimator)
- Pearson/Spearman correlation
- ANOVA (composite analysis)
- Cross-correlation (time lags)

## 🎯 Next Steps

### Immediate Actions:

1. **Run baseline analysis** on your full dataset
2. **Review outputs** to understand your data
3. **Download climate indices** (ENSO, PDO, etc.)
4. **Run climate comparison** to find relationships

### For Publication:

1. **Identify key findings** from trend analysis
2. **Create composite figures** combining multiple analyses
3. **Write methods section** using provided documentation
4. **Prepare data tables** from JSON outputs
5. **Generate final figures** at 300 DPI for publication

### Advanced Analysis:

1. **Integrate GRACE data** for water balance
2. **Compare with GNSS** for loading analysis
3. **Validate groundwater models** with surface water
4. **Perform wavelet analysis** for multi-scale patterns
5. **Run climate projections** using trends

## 💡 Tips

### Data Quality
- Monitor cloud fraction in outputs
- Exclude months with >50% clouds for critical analyses
- Use baseline periods with good coverage

### Interpretation
- **Permanent water trends** → Long-term climate/groundwater changes
- **Seasonal water trends** → Precipitation/snowmelt changes  
- **Fragmentation increase** → Drying/water extraction
- **Centroid migration** → Spatial redistribution of water

### Troubleshooting
- **No mosaics found**: Check paths to `cleaned/` or `mosaics/` directories
- **Metric not found**: Use `total_area_km2_anomaly_std` for climate comparisons
- **Seasonal decomposition fails**: Need ≥24 months of data
- **Cross-correlation fails**: Check date overlap between datasets

## 📝 Example Publications Structure

### Title
"Forty-year surface water trends reveal [finding] driven by [mechanism] in [region]"

### Key Results (from your outputs)

**Finding 1: Long-term trend**
- "Permanent water declined by X km²/year (p < 0.01) from 1984-2024"
- Figure: `time_series_overview.png` panel 1

**Finding 2: Climate teleconnection**
- "Surface water anomalies correlate with ENSO (r=0.42, 3-month lag)"
- Figure: `cross_correlation_enso.png`

**Finding 3: Mechanism**
- "Fragmentation increased 2%/decade, indicating progressive drying"
- Figure: `seasonal_cycles.png` + trend analysis

**Finding 4: Seasonal changes**
- "Spring peak advanced by 2 weeks over study period"
- Figure: STL decomposition of seasonal component

## ✅ What You Can Do Now

- [x] Extract comprehensive surface water metrics
- [x] Perform robust trend analysis
- [x] Detect and quantify anomalies
- [x] Compare with climate indices (ENSO, PDO, etc.)
- [x] Analyze time lags and teleconnections
- [x] Perform composite analysis (phase comparisons)
- [x] Generate publication-quality figures
- [x] Export analysis-ready data (CSV, NetCDF)

## 🎉 You're Ready!

You now have a complete framework for analyzing your surface water time series. The scripts are:
- **Scientifically robust** (peer-review ready)
- **Well documented** (methods are transparent)
- **Flexible** (easily customized for your needs)
- **Integrated** (works with climate data, GRACE, GNSS)

**Start with the basic analysis and explore from there!**

---

*For questions or issues, refer to the detailed documentation files or the example scripts.*

