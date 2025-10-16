#!/bin/bash
# Example scripts for running surface water time series analysis
# Modify paths and parameters for your specific use case

# Define base paths
DATA_DIR="/d/surfaceWater/salina/data/usgs_dsw"
STATS_DIR="${DATA_DIR}/stats"
OUTPUT_BASE="/d/surfaceWater/salina/analysis"

# =============================================================================
# Example 1: Basic analysis of entire time series
# =============================================================================
echo "Running Example 1: Basic full time series analysis..."

python analysis/surface_water_time_series.py \
    --data-dir "${DATA_DIR}" \
    --stats-dir "${STATS_DIR}" \
    --output-dir "${OUTPUT_BASE}/full_time_series" \
    --permanent-threshold 0.80 \
    --seasonal-min 0.15 \
    --seasonal-max 0.80 \
    --trend-method both \
    --export-netcdf \
    --verbose

# =============================================================================
# Example 2: Analysis for specific time period (2000-2023)
# =============================================================================
echo "Running Example 2: Recent period analysis (2000-2023)..."

python analysis/surface_water_time_series.py \
    --data-dir "${DATA_DIR}" \
    --stats-dir "${STATS_DIR}" \
    --output-dir "${OUTPUT_BASE}/recent_2000_2023" \
    --start-date 200001 \
    --end-date 202312 \
    --permanent-threshold 0.80 \
    --seasonal-min 0.15 \
    --seasonal-max 0.80 \
    --trend-method both \
    --export-netcdf \
    --verbose

# =============================================================================
# Example 3: Conservative permanent water threshold (more restrictive)
# =============================================================================
echo "Running Example 3: Conservative permanent water analysis..."

python analysis/surface_water_time_series.py \
    --data-dir "${DATA_DIR}" \
    --stats-dir "${STATS_DIR}" \
    --output-dir "${OUTPUT_BASE}/conservative_permanent" \
    --permanent-threshold 0.90 \
    --seasonal-min 0.20 \
    --seasonal-max 0.90 \
    --min-body-size 20 \
    --trend-method both \
    --verbose

# =============================================================================
# Example 4: Sensitive to ephemeral water (lower thresholds)
# =============================================================================
echo "Running Example 4: Ephemeral water sensitive analysis..."

python analysis/surface_water_time_series.py \
    --data-dir "${DATA_DIR}" \
    --stats-dir "${STATS_DIR}" \
    --output-dir "${OUTPUT_BASE}/ephemeral_sensitive" \
    --permanent-threshold 0.75 \
    --seasonal-min 0.10 \
    --seasonal-max 0.75 \
    --min-body-size 5 \
    --trend-method mann-kendall \
    --verbose

# =============================================================================
# Example 5: Anomaly analysis with specific baseline period
# =============================================================================
echo "Running Example 5: Anomaly analysis with 1990-2010 baseline..."

python analysis/surface_water_time_series.py \
    --data-dir "${DATA_DIR}" \
    --stats-dir "${STATS_DIR}" \
    --output-dir "${OUTPUT_BASE}/anomaly_1990_2010_baseline" \
    --anomaly-baseline-start 199001 \
    --anomaly-baseline-end 201012 \
    --trend-method both \
    --export-netcdf \
    --verbose

# =============================================================================
# Example 6: Pre/Post comparison (split analysis)
# =============================================================================
echo "Running Example 6a: Pre-2000 analysis..."

python analysis/surface_water_time_series.py \
    --data-dir "${DATA_DIR}" \
    --stats-dir "${STATS_DIR}" \
    --output-dir "${OUTPUT_BASE}/pre_2000" \
    --end-date 199912 \
    --trend-method both \
    --verbose

echo "Running Example 6b: Post-2000 analysis..."

python analysis/surface_water_time_series.py \
    --data-dir "${DATA_DIR}" \
    --stats-dir "${STATS_DIR}" \
    --output-dir "${OUTPUT_BASE}/post_2000" \
    --start-date 200001 \
    --trend-method both \
    --verbose

# =============================================================================
# Example 7: Quick test on single year
# =============================================================================
echo "Running Example 7: Quick test on 2020 data..."

python analysis/surface_water_time_series.py \
    --data-dir "${DATA_DIR}" \
    --stats-dir "${STATS_DIR}" \
    --output-dir "${OUTPUT_BASE}/test_2020" \
    --start-date 202001 \
    --end-date 202012 \
    --verbose

echo "All examples complete!"
echo "Results are in: ${OUTPUT_BASE}"

