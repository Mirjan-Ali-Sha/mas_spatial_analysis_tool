# -*- coding: utf-8 -*-
"""
Dry run verification script for MAS Spatial Analysis Tool.
"""

import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Mock QGIS and GDAL dependencies
sys.modules['qgis.core'] = MagicMock()
sys.modules['qgis.PyQt.QtGui'] = MagicMock()
sys.modules['qgis.PyQt.QtCore'] = MagicMock()
sys.modules['osgeo'] = MagicMock()

# Mock GDAL specifically to return numpy arrays
mock_gdal = MagicMock()
mock_dataset = MagicMock()
mock_band = MagicMock()

# Create synthetic DEM (10x10)
dem_data = np.array([
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
], dtype=np.float64)

mock_band.ReadAsArray.return_value = dem_data
mock_band.GetNoDataValue.return_value = -9999
mock_dataset.GetRasterBand.return_value = mock_band
mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
mock_dataset.GetProjection.return_value = "WGS84"
mock_gdal.Open.return_value = mock_dataset

sys.modules['osgeo'].gdal = mock_gdal

# Add plugin directory to path
plugin_dir = r'c:/Users/acer/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/mas_spatial_analysis_tool'
sys.path.append(plugin_dir)

# Import core modules
print("Importing core modules...")
from core.morphometry import MorphometryProcessor
from core.flow_algorithms import FlowRouter
from core.hydro_utils import HydrologicalAnalyzer
from core.array_utils import focal_statistics

print("Testing MorphometryProcessor...")
proc = MorphometryProcessor("mock_path.tif")
slope = proc.calculate_slope()
print(f"Slope shape: {slope.shape}")

print("Testing Statistical Filters...")
mean_filter = focal_statistics(dem_data, 3, 'mean')
print(f"Mean filter shape: {mean_filter.shape}")
median_filter = focal_statistics(dem_data, 3, 'median')
print(f"Median filter shape: {median_filter.shape}")

print("Testing FlowRouter...")
router = FlowRouter(dem_data, 1.0)
filled = router.fill_depressions()
print(f"Filled DEM shape: {filled.shape}")
flow_dir = router.d8_flow_direction()
print(f"Flow Direction shape: {flow_dir.shape}")
flow_acc = router.d8_flow_accumulation(flow_dir)
print(f"Flow Accumulation shape: {flow_acc.shape}")
streams = router.extract_streams(flow_acc, 5.0)
print(f"Streams shape: {streams.shape}")

print("Testing Stream Ordering...")
strahler = router.strahler_order(flow_dir, streams)
print(f"Strahler Order shape: {strahler.shape}")
shreve = router.shreve_order(flow_dir, streams)
print(f"Shreve Magnitude shape: {shreve.shape}")

print("Testing HydrologicalAnalyzer...")
hydro = HydrologicalAnalyzer(dem_data, 1.0)
twi = hydro.calculate_wetness_index(flow_acc, np.radians(slope))
print(f"TWI shape: {twi.shape}")
spi = hydro.calculate_stream_power_index(flow_acc, np.radians(slope))
print(f"SPI shape: {spi.shape}")

print("Verification complete!")
