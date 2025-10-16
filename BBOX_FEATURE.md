# Bounding Box Crop Feature

## Overview

The `stitch_dsw_usgs.py` script now supports cropping stitched mosaics to a specified lat/lon bounding box. This reduces file sizes and focuses processing on your area of interest.

## Usage

Add the `--bbox` argument with four values: `MIN_LAT MAX_LAT MIN_LON MAX_LON`

```bash
python usgs/stitch_dsw_usgs.py \
    --config-dir /path/to/project \
    --start-date 20240501 \
    --end-date 20240531 \
    --bbox 36.7 36.95 -121.83 -121.69 \
    --overwrite
```

## Example: Salinas Valley Study Area

**Bounding Box:** `36.7 36.95 -121.83 -121.69`

- **Min Latitude:** 36.7°N  
- **Max Latitude:** 36.95°N  
- **Min Longitude:** -121.83°W  
- **Max Longitude:** -121.69°W  

**Results:**
- Original mosaic: 5000 × 10000 pixels (~500 MB)
- Cropped mosaic: 648 × 1011 pixels (~6 MB)
- **~80x reduction in file size**

## How It Works

1. **Input:** Lat/lon bounding box in EPSG:4326 (WGS84)
2. **Transform:** Converts bbox to the mosaic's native CRS (e.g., Albers Equal Area)
3. **Crop:** Extracts the pixel window covering the bbox
4. **Update:** Adjusts the geotransform to match the cropped extent

## Benefits

- **Faster Processing:** Smaller files mean faster I/O, statistics, and analysis
- **Lower Storage:** Cropped mosaics use ~1-2% of original disk space
- **Focused Analysis:** Only process pixels within your study area
- **Downstream Compatibility:** Works seamlessly with `build_water_statistics.py` and `build_monthly_products.py`

## Notes

- The crop happens **after** stitching, so all input scenes are still merged
- Cropped extent may be slightly larger than requested to align with pixel boundaries
- Works with both INTR and INWAM products
- Compatible with parallel processing (`--workers`)

## Verification

Check the cropped extent:

```bash
gdalinfo /path/to/mosaic_intr.tif | grep -E "Size is|Upper Left|Lower Right"
```

Expected output:
```
Size is 648, 1011
Upper Left  (-2258295.000, 1852935.000) (121d54'44.31"W, 36d55'55.08"N)
Lower Right (-2238855.000, 1822605.000) (121d36'30.41"W, 36d43' 3.37"N)
```

## Integration with Workflow

The bbox feature integrates seamlessly with the full analysis pipeline:

```bash
# Step 1: Stitch with bbox
python usgs/stitch_dsw_usgs.py \
    --config-dir . \
    --year 2024 \
    --bbox 36.7 36.95 -121.83 -121.69

# Step 2: Compute statistics (automatically uses cropped data)
python analysis/build_water_statistics.py \
    --config-dir . \
    --start-date 20240101 \
    --end-date 20241231 \
    --verbose

# Step 3: Build monthly mosaics
python analysis/build_monthly_products.py \
    --config-dir . \
    --year 2024
```

All downstream tools automatically work with the cropped extent.

