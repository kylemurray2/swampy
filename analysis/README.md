# Surface Water Time Series Analysis

## 🌊 Overview

A comprehensive framework for analyzing decades of surface water classifications from USGS DSWE data. Designed for comparing surface water changes with climate indices (ENSO, PDO), groundwater models, GRACE, loading models, and GNSS vertical positions.

**What it does:**
- Extracts 14+ physically meaningful metrics from monthly mosaics
- Performs robust statistical analyses (trend detection, seasonal decomposition, anomalies)
- Compares surface water with climate indices and geophysical data
- Generates publication-quality visualizations
- Exports analysis-ready data (CSV, NetCDF)

## 🚀 Quick Start

### 1. Check Dependencies

```bash
python analysis/test_dependencies.py
```

### 2. Run Basic Analysis

```bash
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/results \
    --verbose
```

### 3. Compare with Climate Index

```bash
python analysis/compare_with_climate_indices.py \
    --water-csv /d/surfaceWater/salina/analysis/results/water_time_series.csv \
    --climate-csv path/to/enso_index.csv \
    --output-dir /d/surfaceWater/salina/analysis/climate_comparison \
    --verbose
```

## 📁 What's Included

### Core Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `surface_water_time_series.py` | Extract metrics from mosaics | Trend analysis, seasonal decomposition, anomaly detection |
| `compare_with_climate_indices.py` | Compare with external data | Cross-correlation, composite analysis, time-lag detection |
| `test_dependencies.py` | Validate installation | Checks all required packages |

### Documentation (Read in Order)

1. **`SUMMARY.md`** ⭐ **START HERE** ⭐
   - Complete implementation summary
   - Quick reference guide
   - What you can do now

2. **`WORKFLOW_DIAGRAM.md`**
   - Visual workflow diagrams
   - Analysis flow by use case
   - Command examples

3. **`README_TIME_SERIES.md`**
   - Quick start guide
   - Example workflows
   - Troubleshooting

4. **`SURFACE_WATER_ANALYSIS.md`**
   - Detailed methodology
   - Integration examples
   - Statistical interpretation

5. **`SURFACE_WATER_SCIENCE.md`**
   - High-level scientific framework
   - Research question examples
   - Publication-ready templates

### Helper Files

- `run_water_analysis_examples.sh` - 7 example workflows
- `requirements_analysis.txt` - Python dependencies
- `WORKFLOW.md` - Overall swampy pipeline

## 📊 Key Outputs

### From `surface_water_time_series.py`:

```
output_dir/
├── water_time_series.csv          # Complete time series (main product)
├── summary_statistics.json        # Statistical overview
├── trend_analysis.json            # Trend results
├── time_series_overview.png       # 6-panel visualization
├── seasonal_cycles.png            # Monthly climatology
├── correlation_matrix.png         # Metric relationships
└── distributions.png              # Statistical distributions
```

### From `compare_with_climate_indices.py`:

```
output_dir/
├── comparison_*.png               # Time series overlay
├── cross_correlation_*.png        # Lag analysis
└── composite_*.png                # Phase comparison
```

## 🔬 Scientific Approach

### Water Body Classification

Uses multi-temporal statistics instead of arbitrary thresholds:

- **Permanent Water** (prob > 0.80): Lakes, reservoirs → Long-term trends
- **Seasonal Water** (0.15-0.80): Wetlands, seasonal streams → Annual cycles
- **Ephemeral Water** (< 0.15): Playas, flood zones → Extreme events

### Extracted Metrics (per month)

**Spatial:**
- Total water area (km²)
- Water by type (permanent/seasonal/ephemeral)
- Number of water bodies
- Fragmentation index
- Centroid location

**Temporal:**
- Trend components (STL decomposition)
- Anomalies (absolute and standardized)
- Cloud fraction (quality indicator)

### Statistical Methods

- **Trend Analysis**: Linear regression + Mann-Kendall test
- **Seasonal Decomposition**: STL (trend, seasonal, residual)
- **Anomaly Detection**: Standardized anomalies (σ units)
- **Time-Lag Analysis**: Cross-correlation with climate indices
- **Composite Analysis**: El Niño vs La Niña comparisons

## 💡 Example Use Cases

### 1. Long-term Trend Analysis

```bash
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir results/trends \
    --trend-method both \
    --verbose
```

**Answers:**
- Is surface water increasing or decreasing?
- What is the rate of change (km²/year)?
- Is the trend statistically significant?

### 2. ENSO Teleconnection

```bash
# Extract water metrics
python analysis/surface_water_time_series.py [...]

# Compare with ENSO
python analysis/compare_with_climate_indices.py \
    --water-csv results/water_time_series.csv \
    --climate-csv oni_index.csv \
    --output-dir results/enso \
    --max-lag 12
```

**Answers:**
- How does El Niño affect surface water?
- What is the time lag (months)?
- How much does water area change (%)?

### 3. Drought Characterization

```bash
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir results/drought \
    --permanent-threshold 0.85 \
    --verbose
```

**Answers:**
- Which years had severe droughts (anomalies < -2σ)?
- How does fragmentation change during droughts?
- Does permanent water decline?

### 4. GRACE/Loading Analysis

```bash
python analysis/compare_with_climate_indices.py \
    --water-csv results/water_time_series.csv \
    --grace-csv grace_tws.csv \
    --gnss-csv gnss_vertical.csv \
    --output-dir results/loading \
    --verbose
```

**Answers:**
- Surface water contribution to TWS?
- Loading sensitivity (mm per km² water)?
- Does it match elastic loading models?

## 📚 Documentation Guide

### For Getting Started
→ Read `SUMMARY.md` for complete overview  
→ Check `WORKFLOW_DIAGRAM.md` for visual workflow  
→ Follow `README_TIME_SERIES.md` for quick start

### For Understanding Methods
→ Read `SURFACE_WATER_ANALYSIS.md` for methodology  
→ Review `SURFACE_WATER_SCIENCE.md` for scientific framework

### For Running Analysis
→ Use `run_water_analysis_examples.sh` for templates  
→ Check `test_dependencies.py` for setup validation

## 🔧 Key Parameters

### Water Classification

```bash
--permanent-threshold 0.80    # Permanent water probability threshold
--seasonal-min 0.15           # Seasonal water minimum
--seasonal-max 0.80           # Seasonal water maximum
--min-body-size 10            # Minimum water body size (pixels)
```

**Adjust for your region:**
- Arid: Increase thresholds (0.85-0.90)
- Humid: Decrease thresholds (0.70-0.80)
- Noisy: Increase min-body-size (15-20)

### Analysis Options

```bash
--trend-method both           # "linear", "mann-kendall", or "both"
--seasonal-period 12          # Seasonal period (months)
--export-netcdf              # Save as NetCDF
--max-lag 12                 # Maximum time lag for correlation
```

## 📈 Interpreting Results

### Trend Analysis (`trend_analysis.json`)

```json
{
  "total_area_km2": {
    "linear": {
      "slope": -0.5,           // -0.5 km²/year decline
      "p_value": 0.003         // Statistically significant
    },
    "mann_kendall": {
      "trend": "decreasing",
      "p_value": 0.001
    }
  }
}
```

### Anomalies

- **-1 to +1 σ**: Normal variability (68% of time)
- **±1 to ±2 σ**: Moderate event (27%)
- **Beyond ±2 σ**: Extreme event (5%)

Positive = More water than normal  
Negative = Less water than normal

### Cross-Correlation

```
Maximum correlation: r=0.42 at lag=3 months
→ Surface water responds to ENSO with 3-month delay
→ El Niño increases water by ~15%
```

## ⚙️ Installation

### Required Packages

```bash
pip install -r requirements_analysis.txt
```

Includes:
- numpy, pandas, scipy (core scientific)
- rasterio (geospatial)
- scikit-image (image processing)
- statsmodels (statistics)
- matplotlib, seaborn (visualization)

### Optional Packages

```bash
pip install xarray netCDF4  # For NetCDF export
```

## 🎯 Next Steps

1. ✅ **Test installation**: `python analysis/test_dependencies.py`
2. ✅ **Run baseline analysis** on your full dataset
3. ✅ **Review outputs** to understand your data
4. ✅ **Download climate indices** (ENSO, PDO)
5. ✅ **Compare with climate data**
6. ✅ **Integrate GRACE/GNSS/models**
7. ✅ **Publish your findings!**

## 📝 Citation

If you use this framework in publications:

```bibtex
@software{surface_water_analysis,
  title = {Surface Water Time Series Analysis Framework},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/swampy}
}
```

## 🆘 Support

- **Issues?** Check `README_TIME_SERIES.md` troubleshooting section
- **Questions?** Review the comprehensive documentation
- **Examples?** See `run_water_analysis_examples.sh`

## 🎉 You're Ready!

This framework is:
- ✅ **Scientifically robust** (peer-review ready)
- ✅ **Well documented** (transparent methods)
- ✅ **Flexible** (customizable for your needs)
- ✅ **Integrated** (works with climate data, GRACE, GNSS)

**Start with `SUMMARY.md` and begin your analysis!** 🚀

---

*Created for analyzing long-term surface water changes and their relationships with climate variability and geophysical processes.*

