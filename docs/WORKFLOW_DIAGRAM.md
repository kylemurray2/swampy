# Surface Water Analysis Workflow Diagram

## Complete Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT DATA (Your Existing Files)                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Monthly Mosaics:                                                    │
│  /d/surfaceWater/salina/data/usgs_dsw/cleaned/YYYYMM/               │
│  └── mosaic_intr_clean.tif  (1980s - present)                       │
│                                                                       │
│  Statistics:                                                         │
│  /d/surfaceWater/salina/data/usgs_dsw/stats/                        │
│  ├── water_prob.tif              (water probability 0-1)            │
│  ├── persistent_water_mask.tif    (permanent water mask)            │
│  ├── stable_land_mask.tif         (stable land mask)                │
│  └── ephemeral_water_mask.tif     (ephemeral water mask)            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STEP 1: Surface Water Time Series Analysis             │
│                  (surface_water_time_series.py)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Inputs:                                                             │
│  • Monthly mosaics (YYYYMM)                                          │
│  • Water statistics (masks, probabilities)                           │
│                                                                       │
│  Processing:                                                         │
│  1. Classify water bodies (permanent/seasonal/ephemeral)             │
│  2. Extract spatial metrics per month:                               │
│     ├── Total water area (km²)                                       │
│     ├── Water by type (permanent/seasonal/ephemeral)                 │
│     ├── Water by confidence (high/partial)                           │
│     ├── Number of water bodies                                       │
│     ├── Mean/max body size                                           │
│     ├── Fragmentation index                                          │
│     └── Centroid location                                            │
│  3. Perform statistical analyses:                                    │
│     ├── Trend analysis (linear + Mann-Kendall)                       │
│     ├── Seasonal decomposition (STL)                                 │
│     └── Anomaly detection                                            │
│                                                                       │
│  Outputs:                                                            │
│  • water_time_series.csv          ← Main data product               │
│  • summary_statistics.json                                           │
│  • trend_analysis.json                                               │
│  • time_series_overview.png       (6-panel visualization)            │
│  • seasonal_cycles.png            (monthly climatology)              │
│  • correlation_matrix.png         (metric relationships)             │
│  • distributions.png              (statistical distributions)        │
│  • water_time_series.nc           (optional NetCDF)                  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│           STEP 2: Climate & Geophysical Data Comparison             │
│               (compare_with_climate_indices.py)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Inputs:                                                             │
│  • water_time_series.csv (from Step 1)                               │
│  • Climate indices (ENSO, PDO, AMO, etc.)                            │
│  • GRACE TWS data (optional)                                         │
│  • GNSS vertical displacement (optional)                             │
│  • Groundwater model outputs (optional)                              │
│                                                                       │
│  Processing:                                                         │
│  1. Time series alignment and merging                                │
│  2. Cross-correlation analysis (find optimal lags)                   │
│  3. Composite analysis (El Niño vs La Niña)                          │
│  4. Regression analysis                                              │
│  5. Statistical testing                                              │
│                                                                       │
│  Outputs:                                                            │
│  • comparison_[index].png         (time series overlay)              │
│  • cross_correlation_[index].png  (lag analysis)                     │
│  • composite_[index].png          (phase comparison)                 │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  RESULTS & SCIENTIFIC INSIGHTS                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Scientific Questions Answered:                                      │
│                                                                       │
│  1. TRENDS                                                           │
│     • Is surface water increasing or decreasing?                     │
│     • What is the rate of change? (km²/year)                         │
│     • Is the trend significant? (p-value)                            │
│                                                                       │
│  2. CLIMATE TELECONNECTIONS                                          │
│     • How does ENSO affect surface water?                            │
│     • What is the time lag? (months)                                 │
│     • How much does water area change? (% or km²)                    │
│                                                                       │
│  3. SEASONAL PATTERNS                                                │
│     • What is the typical annual cycle?                              │
│     • When is peak/minimum water?                                    │
│     • Is the seasonal cycle changing?                                │
│                                                                       │
│  4. EXTREME EVENTS                                                   │
│     • Which years had droughts/floods? (anomalies > ±2σ)             │
│     • How severe were they?                                          │
│     • What triggered them? (climate indices)                         │
│                                                                       │
│  5. WATER DYNAMICS                                                   │
│     • Is water becoming more fragmented?                             │
│     • Are water bodies migrating spatially?                          │
│     • Is permanent vs seasonal water changing?                       │
│                                                                       │
│  6. LOADING & DEFORMATION                                            │
│     • Does surface water loading match GNSS? (mm displacement)       │
│     • What is the loading sensitivity? (mm/km²)                      │
│     • Surface vs groundwater contribution to TWS?                    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Analysis Flow by Use Case

### Use Case 1: Basic Characterization

```
Monthly Mosaics + Statistics
           ↓
surface_water_time_series.py
           ↓
Review:
  • summary_statistics.json
  • time_series_overview.png
  • trend_analysis.json
           ↓
Scientific Findings:
  • Long-term trend
  • Seasonal pattern
  • Extreme events
```

### Use Case 2: ENSO Teleconnection

```
Monthly Mosaics + Statistics          ENSO Index (ONI)
           ↓                                  ↓
surface_water_time_series.py                 │
           ↓                                  │
water_time_series.csv ─────────────┬──────────┘
                                   ↓
                    compare_with_climate_indices.py
                                   ↓
                         Cross-Correlation Analysis
                                   ↓
                            Composite Analysis
                                   ↓
Scientific Findings:
  • Correlation: r = 0.42
  • Time lag: 3 months
  • El Niño effect: +15% water area
  • La Niña effect: -10% water area
```

### Use Case 3: GRACE Validation

```
Monthly Mosaics + Statistics          GRACE TWS
           ↓                                ↓
surface_water_time_series.py              │
           ↓                                │
water_time_series.csv ─────────────┬───────┘
                                   ↓
                    compare_with_climate_indices.py
                           (with --grace-csv)
                                   ↓
Scientific Findings:
  • Surface water correlation with TWS: r = 0.65
  • Surface water contributes X% to TWS variability
  • Residual = groundwater signal
```

### Use Case 4: Loading Analysis

```
Monthly Mosaics + Statistics          GNSS Vertical
           ↓                                ↓
surface_water_time_series.py              │
           ↓                                │
Calculate water mass                       │
    (area × depth)                         │
           ↓                                │
water_time_series.csv ─────────────┬───────┘
                                   ↓
                    compare_with_climate_indices.py
                           (with --gnss-csv)
                                   ↓
Scientific Findings:
  • Loading sensitivity: X mm per km² water
  • Compare with elastic loading model
  • Validate deformation predictions
```

## Data Flow

### Input Requirements

| Data Type | Format | Required? | Example Path |
|-----------|--------|-----------|--------------|
| Monthly mosaics | GeoTIFF | Yes | `.../cleaned/YYYYMM/mosaic_intr_clean.tif` |
| Water statistics | GeoTIFF | Yes | `.../stats/water_prob.tif` |
| Climate index | CSV | Optional | `oni_index.csv` (date, value) |
| GRACE TWS | CSV | Optional | `grace_tws.csv` (date, tws) |
| GNSS vertical | CSV | Optional | `gnss_vertical.csv` (date, mm) |

### Output Products

| Output | Type | Use For |
|--------|------|---------|
| `water_time_series.csv` | CSV | All subsequent analyses |
| `summary_statistics.json` | JSON | Quick overview, reports |
| `trend_analysis.json` | JSON | Trend significance, rates |
| `time_series_overview.png` | PNG | Publications, presentations |
| `seasonal_cycles.png` | PNG | Understanding annual cycle |
| `correlation_matrix.png` | PNG | Metric relationships |
| `comparison_*.png` | PNG | Climate comparison results |
| `cross_correlation_*.png` | PNG | Time lag analysis |
| `composite_*.png` | PNG | Phase comparison (El Niño/La Niña) |

## Command Line Workflow

### Minimal Workflow (3 commands)

```bash
# 1. Test dependencies
python analysis/test_dependencies.py

# 2. Run time series analysis
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/results \
    --verbose

# 3. Review results
ls /d/surfaceWater/salina/analysis/results/
```

### Complete Workflow (7 commands)

```bash
# 1. Test dependencies
python analysis/test_dependencies.py

# 2. Run time series analysis
python analysis/surface_water_time_series.py \
    --data-dir /d/surfaceWater/salina/data/usgs_dsw \
    --stats-dir /d/surfaceWater/salina/data/usgs_dsw/stats \
    --output-dir /d/surfaceWater/salina/analysis/baseline \
    --trend-method both \
    --export-netcdf \
    --verbose

# 3. Review summary
cat /d/surfaceWater/salina/analysis/baseline/summary_statistics.json

# 4. Compare with ENSO
python analysis/compare_with_climate_indices.py \
    --water-csv /d/surfaceWater/salina/analysis/baseline/water_time_series.csv \
    --climate-csv oni_index.csv \
    --output-dir /d/surfaceWater/salina/analysis/enso \
    --max-lag 12 \
    --verbose

# 5. Compare with GRACE (if available)
python analysis/compare_with_climate_indices.py \
    --water-csv /d/surfaceWater/salina/analysis/baseline/water_time_series.csv \
    --grace-csv grace_tws.csv \
    --output-dir /d/surfaceWater/salina/analysis/grace \
    --verbose

# 6. Compare with GNSS (if available)
python analysis/compare_with_climate_indices.py \
    --water-csv /d/surfaceWater/salina/analysis/baseline/water_time_series.csv \
    --gnss-csv gnss_vertical.csv \
    --output-dir /d/surfaceWater/salina/analysis/gnss \
    --verbose

# 7. Compile results for publication
# Review all PNG files and JSON summaries
```

## Quick Reference

### Core Scripts

1. **`surface_water_time_series.py`** - Extract metrics from mosaics
2. **`compare_with_climate_indices.py`** - Compare with external data

### Documentation

1. **`SUMMARY.md`** - This overview (start here!)
2. **`README_TIME_SERIES.md`** - Quick start guide
3. **`SURFACE_WATER_ANALYSIS.md`** - Detailed methodology
4. **`SURFACE_WATER_SCIENCE.md`** - Scientific framework

### Helper Files

1. **`run_water_analysis_examples.sh`** - Example workflows
2. **`test_dependencies.py`** - Check installation
3. **`requirements_analysis.txt`** - Python packages needed

## What's Next?

1. ✅ Test your installation: `python analysis/test_dependencies.py`
2. ✅ Run basic analysis on your data
3. ✅ Review outputs and understand your time series
4. ✅ Download climate indices
5. ✅ Run climate comparisons
6. ✅ Integrate with GRACE/GNSS/models
7. ✅ Publish your findings!

---

**You now have everything needed for a comprehensive surface water analysis!** 🎉

