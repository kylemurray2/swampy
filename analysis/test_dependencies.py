#!/usr/bin/env python3
"""Test if all required dependencies for surface water analysis are installed."""

import sys
from importlib import import_module

# Required packages
REQUIRED = [
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("scipy", "SciPy"),
    ("rasterio", "Rasterio"),
    ("skimage", "scikit-image"),
    ("statsmodels", "Statsmodels"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
]

# Optional packages
OPTIONAL = [
    ("xarray", "xarray (for NetCDF export)"),
    ("netCDF4", "netCDF4 (for NetCDF export)"),
]


def check_package(module_name: str, display_name: str) -> bool:
    """Check if a package is installed and return its version."""
    try:
        module = import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✓ {display_name:30s} version {version}")
        return True
    except ImportError:
        print(f"✗ {display_name:30s} NOT FOUND")
        return False


def main():
    print("Checking dependencies for Surface Water Time Series Analysis")
    print("=" * 70)
    print()
    
    # Check required packages
    print("Required Packages:")
    print("-" * 70)
    missing_required = []
    for module_name, display_name in REQUIRED:
        if not check_package(module_name, display_name):
            missing_required.append(display_name)
    
    print()
    
    # Check optional packages
    print("Optional Packages:")
    print("-" * 70)
    missing_optional = []
    for module_name, display_name in OPTIONAL:
        if not check_package(module_name, display_name):
            missing_optional.append(display_name)
    
    print()
    print("=" * 70)
    
    # Summary
    if missing_required:
        print("\n⚠ WARNING: Missing required packages:")
        for pkg in missing_required:
            print(f"  - {pkg}")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements_analysis.txt")
        return 1
    else:
        print("\n✓ All required packages are installed!")
    
    if missing_optional:
        print("\nℹ Optional packages not installed:")
        for pkg in missing_optional:
            print(f"  - {pkg}")
        print("\nThese are only needed for NetCDF export (--export-netcdf flag)")
        print("Install with: pip install xarray netCDF4")
    
    print("\n✓ You're ready to run surface water time series analysis!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

