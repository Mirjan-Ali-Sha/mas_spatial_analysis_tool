# -*- coding: utf-8 -*-
"""
Native DEM processing utilities using GDAL and NumPy
NO external binary dependencies
"""

import numpy as np
from osgeo import gdal, osr
from qgis.core import QgsProcessingException

class DEMProcessor:
    """Native DEM processing class using GDAL and NumPy."""
    
    def __init__(self, dem_path):
        """Initialize DEM processor.
        
        Args:
            dem_path (str): Path to DEM raster file
        """
        self.dem_path = dem_path
        self.dataset = None
        self.array = None
        self.nodata = None
        self.geotransform = None
        self.projection = None
        self.cellsize_x = None
        self.cellsize_y = None
        
        self._load_dem()
    
    def _load_dem(self):
        """Load DEM into memory."""
        try:
            self.dataset = gdal.Open(self.dem_path, gdal.GA_ReadOnly)
            if self.dataset is None:
                raise QgsProcessingException(f"Cannot open DEM: {self.dem_path}")
            
            band = self.dataset.GetRasterBand(1)
            self.source_dtype = band.DataType  # Store source data type
            self.array = band.ReadAsArray().astype(np.float64)
            self.nodata = band.GetNoDataValue()
            self.geotransform = self.dataset.GetGeoTransform()
            self.projection = self.dataset.GetProjection()
            
            # Calculate cell sizes
            self.cellsize_x = abs(self.geotransform[1])
            self.cellsize_y = abs(self.geotransform[5])
            
            # Replace nodata with NaN (handle floating point comparison issues)
            if self.nodata is not None:
                # Use isclose for float comparison to handle precision issues
                nodata_mask = np.isclose(self.array, self.nodata, rtol=1e-5, atol=1e-8, equal_nan=True)
                self.array[nodata_mask] = np.nan
            
            # Also handle very large negative values (common NoData = -3.4e+38)
            self.array[self.array < -1e30] = np.nan
                
        except Exception as e:
            raise QgsProcessingException(f"Error loading DEM: {str(e)}")
    
    def save_raster(self, output_path, data, dtype=None, nodata=None):
        """Save array as GeoTIFF.
        
        Args:
            output_path (str): Output file path
            data (np.ndarray): Data array to save
            dtype: GDAL data type (default: Float32 for most outputs)
            nodata: NoData value (default: -9999.0)
        """
        try:
            # Default to Float32 for derived products (slope, etc.)
            save_dtype = dtype if dtype is not None else gdal.GDT_Float32
            save_nodata = nodata if nodata is not None else -9999.0
            
            # Create a copy and handle nodata values
            data_copy = data.copy().astype(np.float32)
            
            # Replace NaN with nodata value
            data_copy[np.isnan(data)] = save_nodata
            
            # Also ensure -9999.0 values are preserved (in case input already uses this)
            # This handles cases where algorithms return -9999.0 directly
            
            # Create output dataset
            driver = gdal.GetDriverByName('GTiff')
            rows, cols = data.shape
            
            out_dataset = driver.Create(
                output_path, cols, rows, 1, save_dtype,
                options=['COMPRESS=LZW', 'TILED=YES']
            )
            
            if out_dataset is None:
                raise QgsProcessingException(f"Cannot create output: {output_path}")
            
            # Set georeferencing
            out_dataset.SetGeoTransform(self.geotransform)
            out_dataset.SetProjection(self.projection)
            
            # Write data
            out_band = out_dataset.GetRasterBand(1)
            out_band.WriteArray(data_copy)
            out_band.SetNoDataValue(float(save_nodata))
            out_band.FlushCache()
            
            # Calculate statistics (excludes nodata)
            out_band.ComputeStatistics(False)
            
            # Close dataset
            out_dataset = None
            
        except Exception as e:
            raise QgsProcessingException(f"Error saving raster: {str(e)}")
    
    def close(self):
        """Close dataset."""
        if self.dataset:
            self.dataset = None
