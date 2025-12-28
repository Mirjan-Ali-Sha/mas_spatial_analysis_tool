# -*- coding: utf-8 -*-
"""
Optimized array operations for raster processing
Uses NumPy vectorization and block processing for large rasters
"""

import numpy as np
from osgeo import gdal
from qgis.core import QgsProcessingException, QgsMessageLog, Qgis


class BlockProcessor:
    """Efficient block-based raster processing for large datasets."""
    
    def __init__(self, raster_path, block_size=512):
        """Initialize block processor.
        
        Args:
            raster_path (str): Path to raster file
            block_size (int): Processing block size in pixels
        """
        self.raster_path = raster_path
        self.block_size = block_size
        self.dataset = None
        self.band = None
        
        self._open_dataset()
    
    def _open_dataset(self):
        """Open raster dataset."""
        self.dataset = gdal.Open(self.raster_path, gdal.GA_ReadOnly)
        if self.dataset is None:
            raise QgsProcessingException(f"Cannot open raster: {self.raster_path}")
        
        self.band = self.dataset.GetRasterBand(1)
        self.rows = self.dataset.RasterYSize
        self.cols = self.dataset.RasterXSize
        self.nodata = self.band.GetNoDataValue()
        self.geotransform = self.dataset.GetGeoTransform()
        self.projection = self.dataset.GetProjection()
    
    def process_with_function(self, processing_func, output_path, 
                             output_dtype=gdal.GDT_Float32, **kwargs):
        """Process raster block-by-block with custom function.
        
        Args:
            processing_func (callable): Function to process each block
            output_path (str): Output raster path
            output_dtype: GDAL output data type
            **kwargs: Additional arguments for processing function
            
        Returns:
            bool: Success status
        """
        # Create output dataset
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(
            output_path,
            self.cols,
            self.rows,
            1,
            output_dtype,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
        )
        
        if out_dataset is None:
            raise QgsProcessingException(f"Cannot create output: {output_path}")
        
        out_dataset.SetGeoTransform(self.geotransform)
        out_dataset.SetProjection(self.projection)
        out_band = out_dataset.GetRasterBand(1)
        out_band.SetNoDataValue(self.nodata if self.nodata else -9999)
        
        # Process by blocks
        total_blocks = ((self.rows + self.block_size - 1) // self.block_size) * \
                      ((self.cols + self.block_size - 1) // self.block_size)
        
        processed_blocks = 0
        
        for i in range(0, self.rows, self.block_size):
            for j in range(0, self.cols, self.block_size):
                # Calculate actual block size
                rows_to_read = min(self.block_size, self.rows - i)
                cols_to_read = min(self.block_size, self.cols - j)
                
                # Read block with context (overlap for edge operations)
                overlap = 1
                i_start = max(0, i - overlap)
                j_start = max(0, j - overlap)
                i_end = min(self.rows, i + rows_to_read + overlap)
                j_end = min(self.cols, j + cols_to_read + overlap)
                
                block_data = self.band.ReadAsArray(
                    j_start, i_start,
                    j_end - j_start, i_end - i_start
                ).astype(np.float64)
                
                # Replace nodata with NaN
                if self.nodata is not None:
                    block_data[block_data == self.nodata] = np.nan
                
                # Process block
                try:
                    result = processing_func(
                        block_data,
                        overlap=overlap,
                        **kwargs
                    )
                    
                    # Extract core result (remove overlap)
                    i_offset = overlap if i > 0 else 0
                    j_offset = overlap if j > 0 else 0
                    core_result = result[
                        i_offset:i_offset + rows_to_read,
                        j_offset:j_offset + cols_to_read
                    ]
                    
                    # Replace NaN with nodata
                    if self.nodata is not None:
                        core_result[np.isnan(core_result)] = self.nodata
                    
                    # Write result
                    out_band.WriteArray(core_result, j, i)
                    
                except Exception as e:
                    QgsMessageLog.logMessage(
                        f"Error processing block at ({i}, {j}): {str(e)}",
                        'MAS Geospatial',
                        Qgis.Warning
                    )
                
                processed_blocks += 1
        
        # Compute statistics
        out_band.ComputeStatistics(False)
        out_band.FlushCache()
        
        # Close datasets
        out_dataset = None
        
        return True
    
    def read_full_array(self, as_float=True):
        """Read entire raster as array (for small rasters).
        
        Args:
            as_float (bool): Convert to float64
            
        Returns:
            np.ndarray: Raster array
        """
        array = self.band.ReadAsArray()
        
        if as_float:
            array = array.astype(np.float64)
        
        if self.nodata is not None:
            array[array == self.nodata] = np.nan
        
        return array
    
    def close(self):
        """Close dataset."""
        if self.dataset:
            self.dataset = None


def focal_statistics(array, window_size, statistic='mean', overlap=1):
    """Calculate focal statistics using efficient convolution.
    
    Args:
        array (np.ndarray): Input array
        window_size (int): Window size (must be odd)
        statistic (str): Statistic type ('mean', 'max', 'min', 'std', 'range')
        overlap (int): Overlap size to account for
        
    Returns:
        np.ndarray: Result array
    """
    from scipy import ndimage
    
    if window_size % 2 == 0:
        window_size += 1
    
    # Create kernel
    kernel = np.ones((window_size, window_size))
    kernel /= kernel.sum()
    
    if statistic == 'mean':
        result = ndimage.convolve(array, kernel, mode='nearest')
    
    elif statistic == 'max':
        result = ndimage.maximum_filter(array, size=window_size, mode='nearest')
    
    elif statistic == 'min':
        result = ndimage.minimum_filter(array, size=window_size, mode='nearest')
    
    elif statistic == 'std':
        mean = ndimage.convolve(array, kernel, mode='nearest')
        mean_sq = ndimage.convolve(array**2, kernel, mode='nearest')
        variance = mean_sq - mean**2
        result = np.sqrt(np.maximum(variance, 0))
    
    elif statistic == 'range':
        max_val = ndimage.maximum_filter(array, size=window_size, mode='nearest')
        min_val = ndimage.minimum_filter(array, size=window_size, mode='nearest')
        result = max_val - min_val
        
    elif statistic == 'median':
        result = ndimage.median_filter(array, size=window_size, mode='nearest')
        
    elif statistic == 'sum':
        result = ndimage.convolve(array, kernel * (window_size**2), mode='nearest')
    
    else:
        result = array
    
    return result


def efficient_gradient(array, cellsize_x, cellsize_y):
    """Calculate gradients using optimized finite differences.
    
    Args:
        array (np.ndarray): Input array
        cellsize_x (float): X cell size
        cellsize_y (float): Y cell size
        
    Returns:
        tuple: (dz/dx, dz/dy) arrays
    """
    # Use NumPy's gradient function (optimized C implementation)
    gy, gx = np.gradient(array, cellsize_y, cellsize_x)
    
    return gx, gy


def vectorized_reclassify(array, breaks, values):
    """Vectorized reclassification (much faster than loops).
    
    Args:
        array (np.ndarray): Input array
        breaks (list): Break values
        values (list): Output values
        
    Returns:
        np.ndarray: Reclassified array
    """
    result = np.zeros_like(array)
    
    for i in range(len(breaks) - 1):
        mask = (array >= breaks[i]) & (array < breaks[i + 1])
        result[mask] = values[i]
    
    # Handle last class
    result[array >= breaks[-1]] = values[-1]
    
    return result


def apply_mask(array, mask, mask_value=np.nan):
    """Apply mask to array efficiently.
    
    Args:
        array (np.ndarray): Input array
        mask (np.ndarray): Boolean mask
        mask_value: Value to assign to masked cells
        
    Returns:
        np.ndarray: Masked array
    """
    result = array.copy()
    result[mask] = mask_value
    return result


def safe_divide(numerator, denominator, fill_value=0):
    """Safe division handling zeros and infinities.
    
    Args:
        numerator (np.ndarray): Numerator array
        denominator (np.ndarray): Denominator array
        fill_value: Value for invalid results
        
    Returns:
        np.ndarray: Division result
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(numerator, denominator)
        result[~np.isfinite(result)] = fill_value
    
    return result
