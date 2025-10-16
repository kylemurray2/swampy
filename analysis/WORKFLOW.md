# Surface Water Analysis Workflow

This document describes the proper workflow for processing USGS DSWE data into analysis-ready surface water products.

## Overview

The workflow follows four main stages:

1. **Stitch Daily Mosaics** - Merge raw DSWE scenes into cropped, daily mosaics.
2. **Daily Statistics** - Compute unbiased statistics from the daily mosaics.
3. **Monthly Aggregation** - Create monthly composites using statistics-informed merging.
4. **Final Cleaning** - Apply masks and heuristics to produce analysis-ready data.

## Stage 1: Stitch Daily Mosaics (`usgs/stitch_dsw_usgs.py`)

**Purpose:** The crucial first step is to process the raw DSWE scene directories (e.g., `LC08_CU..._DSWE`) into a clean, consistent time series of daily mosaics. This script handles merging scenes from the same day, optionally cropping them to a specific bounding box, and organizing the file structure.

**Input:** Raw DSWE scene directories.

**Outputs:**
- Daily mosaics for INTR and INWAM products in `stitched_dates/YYYYMMDD/`.
- Moves the original scene directories into `raw/`.

**Usage:**
```bash
python usgs/stitch_dsw_usgs.py \
    --config-dir /path/to/project \
    --year 2024 \
    --bbox 36.3 37.2 -122.0 -121.2 \
    --overwrite
```

**Key Parameters:**
- `--bbox`: `MIN_LAT MAX_LAT MIN_LON MAX_LON` - Crops all outputs to this lat/lon extent. Essential for focusing the analysis and reducing file sizes.
- `--overwrite`: Re-processes and overwrites existing mosaics.

## Stage 2: Daily Statistics (`analysis/build_water_statistics.py`)

**Purpose:** Compute per-pixel statistics from all daily observations to characterize water behavior without bias from pre-aggregated data.

**Input:** Daily stitched mosaics from `stitched_dates/YYYYMMDD/`.

**Outputs:**
- `water_prob.tif` - Probability a pixel is water (0-1).
- `cloud_frac.tif` - Fraction of observations that were cloudy (0-1).
- `valid_count.tif` - Number of valid (non-nodata) observations.
- `mode_class.tif` - Most common classification across all dates.
- `persistent_water_mask.tif`, `ephemeral_water_mask.tif`, `stable_land_mask.tif`
- `ocean_mask.tif`, `shore_buffer.tif`

**Usage:**
```bash
python analysis/build_water_statistics.py \
    --config-dir /path/to/project \
    --start-date 20200101 \
    --end-date 20241231 \
    --workers 8 \
    --chunk-size 100 \
    --verbose
```

**Performance:** Processes data in chunks with parallel I/O. Typical speed: ~100-200 dates/minute on modern hardware.

**Key Parameters:**
- `--chunk-size`: Number of dates to load at once (default: 100). Increase for more RAM, decrease if memory-constrained.
- `--workers`: Parallel I/O threads (default: 4)
- `--persistent-threshold`: Water probability threshold for persistent water (default: 0.95)
- `--ephemeral-min`: Minimum water probability for ephemeral water (default: 0.1)

## Stage 3: Monthly Aggregation (`analysis/build_monthly_products.py`)

**Purpose:** Create monthly mosaics using confidence-weighted voting that considers both INTR and INWAM values.

**Input:** 
- Daily stitched mosaics from `stitched_dates/YYYYMMDD/`
- Statistics from Stage 1 (future enhancement)

**Outputs:**
- `mosaics/YYYYMM/mosaic_intr.tif` - Monthly INTR composite
- `mosaics/YYYYMM/mosaic_inwam.tif` - Monthly INWAM composite

**Usage:**
```bash
python analysis/build_monthly_products.py \
    --config-dir /path/to/project \
    --products INTR INWAM \
    --year 2024 \
    --overwrite \
    --debug
```

**Algorithm:** Uses weighted voting where:
- INWAM values provide confidence weights (1=high, 4=low)
- INTR classes fallback when INWAM unavailable
- Land observations weighted equally to water
- Requires minimum valid observations and confidence thresholds
- Rejects low-confidence water unless strongly supported

## Stage 4: Final Cleaning (`analysis/clean_monthly_mosaics.py`)

**Purpose:** Apply statistics-derived masks and heuristics to produce analysis-ready monthly products.

**Status:** To be implemented

**Planned Features:**
- Apply persistent/ephemeral/stable masks
- Temporal gap filling using neighboring months
- Speckle removal (small isolated patches)
- Confidence raster generation
- Integration with DEM slope filtering

## Data Flow

```
Raw DSWE Scenes
    ↓
[Stage 1: usgs/stitch_dsw_usgs.py]
    ↓
Daily Stitched Mosaics (stitched_dates/YYYYMMDD/)
    ↓
[Stage 2: analysis/build_water_statistics.py]
    ↓
Statistics Masks (stats/)
    ↓ (informs)
[Stage 3: analysis/build_monthly_products.py]
    ↓
Monthly Mosaics (mosaics/YYYYMM/)
    ↓
[Stage 4: analysis/clean_monthly_mosaics.py]
    ↓
Cleaned Products (cleaned/YYYYMM/)
```

## Important Notes

### Why Daily Statistics First?

Computing statistics from daily data (Stage 1) before monthly aggregation (Stage 2) is critical because:

1. **Unbiased observations:** Daily mosaics reflect raw sensor observations without aggregation bias
2. **Full temporal detail:** Captures ephemeral events that monthly composites might miss
3. **Better masks:** Persistent water/land masks derived from all observations, not pre-filtered data
4. **Circular dependency:** Computing stats from monthly mosaics creates feedback loops where aggressive filtering hides real water

### Common Pitfalls

**Don't:** Run statistics on monthly mosaics that were already filtered/aggregated
**Do:** Always compute statistics from raw daily stitched data

**Don't:** Use the same aggressive thresholds for all regions
**Do:** Tune `--persistent-threshold`, `--ephemeral-min` based on your study area

**Don't:** Process entire archive at once without testing
**Do:** Test on a single month/year first, verify outputs, then scale up

## Performance Tips

1. **Chunking:** Adjust `--chunk-size` based on available RAM (each chunk loads ~chunk_size × grid_size × 2 bytes)
2. **Workers:** Set `--workers` to number of physical cores for I/O-bound tasks
3. **Subset:** Use `--start-date` and `--end-date` to process incrementally
4. **Storage:** Statistics outputs are small (~10 MB), monthly mosaics are larger (~50-100 MB each)

## Future Enhancements

- [ ] Integrate statistics masks into monthly aggregation
- [ ] Implement Stage 3 cleaning pipeline
- [ ] Add time-series metrics (trend, seasonality, anomalies)
- [ ] Parallel month processing in Stage 2
- [ ] Optional DEM-based slope filtering
- [ ] Water area time series extraction for AOIs

