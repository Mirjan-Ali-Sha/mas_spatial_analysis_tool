# -*- coding: utf-8 -*-
"""
Native morphometric analysis algorithms using NumPy and SciPy.
"""

import numpy as np
from scipy import ndimage
from .dem_utils import DEMProcessor

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator if numba is missing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

class MorphometryProcessor(DEMProcessor):
    """Extended DEM processor for morphometric analysis."""
    
    def calculate_slope(self, units='degrees', z_factor=1.0):
        """Calculate slope using Horn's method (same as GDAL/standard GIS).
        
        Args:
            units (str): 'degrees' or 'percent'
            z_factor (float): Vertical exaggeration factor
            
        Returns:
            np.ndarray: Slope array (nodata = -9999.0)
        """
        # Use concrete nodata value for reliable GDAL compatibility
        NODATA = -9999.0
        
        # Apply z_factor
        dem = self.array * z_factor
        
        # Mask input nodata (replace with 0 temporarily for convolution)
        input_nodata_mask = np.isnan(dem)
        dem_filled = np.where(input_nodata_mask, 0, dem)
        
        # Calculate gradients using Sobel operators (Horn's method)
        # This matches standard GIS slope calculation (GDAL, GRASS, etc.)
        # 3x3 kernel for x direction: (c + 2f + i) - (a + 2d + g) / (8*cellsize)
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]]) / (8.0 * self.cellsize_x)
        
        # 3x3 kernel for y direction: (g + 2h + i) - (a + 2b + c) / (8*cellsize)
        kernel_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]]) / (8.0 * self.cellsize_y)
        
        # Calculate gradients using nearest mode for boundary handling
        dzdx = ndimage.convolve(dem_filled, kernel_x, mode='nearest')
        dzdy = ndimage.convolve(dem_filled, kernel_y, mode='nearest')
        
        # Calculate slope: arctan(sqrt(dzdx^2 + dzdy^2))
        slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        
        if units == 'degrees':
            slope = np.degrees(slope_rad)
        elif units == 'percent':
            slope = np.tan(slope_rad) * 100.0
        else:
            slope = slope_rad  # radians
        
        # Apply nodata mask from input
        slope[input_nodata_mask] = NODATA
        
        # Set edge pixels to NoData (matching standard GIS behavior)
        # Standard implementations require valid neighbors; edge pixels don't have all 8 neighbors
        slope[0, :] = NODATA   # Top edge
        slope[-1, :] = NODATA  # Bottom edge
        slope[:, 0] = NODATA   # Left edge
        slope[:, -1] = NODATA  # Right edge
        
        return slope
    
    def calculate_slope_std_dev(self, window_size=3):
        """Calculate Standard Deviation of Slope.
        
        Args:
            window_size (int): Window size (must be odd)
            
        Returns:
            np.ndarray: StdDev of Slope
        """
        from .array_utils import focal_statistics
        
        # Calculate slope in degrees
        slope = self.calculate_slope(units='degrees')
        
        # Calculate focal StdDev
        # StdDev = sqrt(E[x^2] - (E[x])^2)
        
        # We need a helper for stddev or use the formula with focal means
        # Using the formula with focal means is efficient if we have a fast mean filter.
        # focal_statistics('mean') uses scipy.ndimage.uniform_filter which is fast.
        
        slope_sq = slope**2
        mean_slope = focal_statistics(slope, window_size, 'mean')
        mean_slope_sq = focal_statistics(slope_sq, window_size, 'mean')
        
        var = mean_slope_sq - mean_slope**2
        # Handle precision errors
        var = np.maximum(var, 0)
        std_dev = np.sqrt(var)
        
        std_dev[np.isnan(self.array)] = np.nan
        
        return std_dev
    
    def calculate_aspect(self):
        """Calculate aspect (direction of slope).
        
        Returns:
            np.ndarray: Aspect array (0-360 degrees, 0=North, clockwise)
        """
        # Calculate gradients
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]]) / (8.0 * self.cellsize_x)
        
        kernel_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]]) / (8.0 * self.cellsize_y)
        
        dzdx = ndimage.convolve(self.array, kernel_x, mode='nearest')
        dzdy = ndimage.convolve(self.array, kernel_y, mode='nearest')
        
        # Calculate aspect (in radians, then convert to degrees)
        aspect_rad = np.arctan2(dzdy, -dzdx)
        aspect = np.degrees(aspect_rad)
        
        # Convert to compass direction (0=North, clockwise)
        aspect = 90.0 - aspect
        aspect[aspect < 0] += 360.0
        
        # Handle flat areas (slope = 0)
        flat = (dzdx == 0) & (dzdy == 0)
        aspect[flat] = -1  # -1 indicates flat
        
        # Mask nodata
        aspect[np.isnan(self.array)] = np.nan
        
        return aspect
    
    def calculate_circular_variance(self, window_size=3):
        """Calculate circular variance of aspect.
        
        Variance = 1 - R
        R = sqrt( (sum(cos(a)))^2 + (sum(sin(a)))^2 ) / N
        
        Args:
            window_size (int): Window size (must be odd)
            
        Returns:
            np.ndarray: Circular variance (0-1)
        """
        from .array_utils import focal_statistics
        
        # Calculate aspect in radians
        # Note: calculate_aspect returns degrees, -1 for flat
        aspect_deg = self.calculate_aspect()
        
        # Mask flat areas (-1)
        valid_mask = (aspect_deg != -1) & (~np.isnan(aspect_deg))
        
        # Convert to radians (0-360 -> 0-2pi)
        # We need to handle the -1 carefully.
        # Let's create sin/cos arrays with NaN for invalid
        sin_a = np.full_like(self.array, np.nan)
        cos_a = np.full_like(self.array, np.nan)
        
        aspect_rad = np.radians(aspect_deg[valid_mask])
        sin_a[valid_mask] = np.sin(aspect_rad)
        cos_a[valid_mask] = np.cos(aspect_rad)
        
        # Calculate focal mean of sin and cos
        # focal_statistics 'mean' handles NaNs by ignoring them?
        # We need a mean that ignores NaNs.
        # Our focal_statistics helper might not handle NaNs in the way we want for this.
        # Let's use uniform_filter but we need to handle NaN counts.
        
        # Alternative: Use Numba for efficiency and correctness
        return _circular_variance_numba(aspect_deg, window_size)

    def calculate_relative_aspect(self, azimuth=0.0):
        """Calculate relative aspect (angular distance to azimuth).
        
        Args:
            azimuth (float): Target azimuth in degrees
            
        Returns:
            np.ndarray: Relative aspect (0-180 degrees)
        """
        aspect = self.calculate_aspect()
        
        # Mask flat areas
        flat = (aspect == -1)
        
        # Calculate difference
        # Relative aspect is the absolute difference, wrapped to 180
        # |Aspect - Azimuth|
        diff = np.abs(aspect - azimuth)
        
        # Handle wrapping
        # If diff > 180, take 360 - diff
        diff = np.where(diff > 180, 360 - diff, diff)
        
        diff[flat] = np.nan # Or 0? Usually undefined for flat.
        diff[np.isnan(self.array)] = np.nan
        
        return diff
    
    def calculate_hillshade(self, azimuth=315.0, altitude=45.0, z_factor=1.0):
        """Calculate hillshade.
        
        Args:
            azimuth (float): Light source azimuth (0-360 degrees)
            altitude (float): Light source altitude (0-90 degrees)
            z_factor (float): Vertical exaggeration
            
        Returns:
            np.ndarray: Hillshade values (0-255)
        """
        # Convert angles to radians
        azimuth_rad = np.radians(360.0 - azimuth + 90.0)
        altitude_rad = np.radians(altitude)
        
        # Apply z_factor
        dem = self.array * z_factor
        
        # Calculate gradients
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]]) / (8.0 * self.cellsize_x)
        
        kernel_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]]) / (8.0 * self.cellsize_y)
        
        dzdx = ndimage.convolve(dem, kernel_x, mode='nearest')
        dzdy = ndimage.convolve(dem, kernel_y, mode='nearest')
        
        # Calculate slope and aspect
        slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        aspect = np.arctan2(dzdy, -dzdx)
        
        # Calculate hillshade
        hillshade = np.sin(altitude_rad) * np.sin(slope) + \
                    np.cos(altitude_rad) * np.cos(slope) * \
                    np.cos(azimuth_rad - aspect)
        
        hillshade[hillshade < 0] = 0
        hillshade = hillshade * 255.0
        
        hillshade[np.isnan(self.array)] = np.nan
        
        return hillshade

    def calculate_multidirectional_hillshade(self, altitude=45.0, z_factor=1.0):
        """Calculate multidirectional hillshade.
        
        Combines hillshades from 225, 270, 315, 360 degrees.
        
        Args:
            altitude (float): Light source altitude
            z_factor (float): Vertical exaggeration
            
        Returns:
            np.ndarray: Multidirectional hillshade (0-255)
        """
        # Directions: 225, 270, 315, 360
        azimuths = [225.0, 270.0, 315.0, 360.0]
        
        # Calculate weights? 
        # Standard approach: Weighted sum or just sum?
        # Usually: 
        # HS_weight = sin(altitude) * sin(slope) + cos(altitude) * cos(slope) * cos(az - aspect)
        # We can just average the 4 hillshades.
        
        combined_hs = np.zeros_like(self.array, dtype=np.float32)
        
        for az in azimuths:
            hs = self.calculate_hillshade(azimuth=az, altitude=altitude, z_factor=z_factor)
            combined_hs += hs
            
        # Average
        combined_hs /= 4.0
        
        return combined_hs

    def calculate_curvature(self, curvature_type='plan'):
        """Calculate curvature.
        
        Args:
            curvature_type (str): Type of curvature to calculate.
            
        Returns:
            np.ndarray: Curvature values
        """

        # Calculate first derivatives
        gy, gx = np.gradient(self.array, self.cellsize_y, self.cellsize_x)
        
        # Calculate second derivatives
        gyy, gyx = np.gradient(gy, self.cellsize_y, self.cellsize_x)
        gxy, gxx = np.gradient(gx, self.cellsize_y, self.cellsize_x)
        
        # Ensure symmetry
        gxy = (gxy + gyx) / 2.0
        
        # Common terms
        p = gx**2 + gy**2
        q = p + 1
        root_q = np.sqrt(q)
        root_q3 = root_q**3
        
        if curvature_type == 'plan':
            # Plan curvature (horizontal curvature)
            curvature = (gxx * gy**2 - 2 * gxy * gx * gy + gyy * gx**2) / (p * root_q)
            
        elif curvature_type == 'profile':
            # Profile curvature (vertical curvature)
            curvature = -(gxx * gx**2 + 2 * gxy * gx * gy + gyy * gy**2) / (p * root_q3)
            
        elif curvature_type == 'tangential':
            # Tangential curvature
            curvature = (gxx * gy**2 - 2 * gxy * gx * gy + gyy * gx**2) / (p * np.sqrt(1 + p))
            
        elif curvature_type == 'mean':
            # Mean curvature
            curvature = -((1 + gy**2) * gxx - 2 * gx * gy * gxy + (1 + gx**2) * gyy) / (2 * q * root_q)
            
        elif curvature_type == 'total':
            # Total curvature (Laplacian)
            curvature = gxx + gyy
            
        elif curvature_type == 'gaussian':
            # Gaussian curvature
            curvature = (gxx * gyy - gxy**2) / (q**2)
            
        elif curvature_type == 'maximal':
            # Maximal curvature (k_max)
            H = -((1 + gy**2) * gxx - 2 * gx * gy * gxy + (1 + gx**2) * gyy) / (2 * q * root_q) # Mean
            K = (gxx * gyy - gxy**2) / (q**2) # Gaussian
            curvature = H + np.sqrt(np.maximum(0, H**2 - K))
            
        elif curvature_type == 'minimal':
            # Minimal curvature (k_min)
            H = -((1 + gy**2) * gxx - 2 * gx * gy * gxy + (1 + gx**2) * gyy) / (2 * q * root_q) # Mean
            K = (gxx * gyy - gxy**2) / (q**2) # Gaussian
            curvature = H - np.sqrt(np.maximum(0, H**2 - K))
            
        elif curvature_type == 'horizontal_excess':
            # Horizontal Excess Curvature
            curvature = (gxx * gy**2 - 2 * gxy * gx * gy + gyy * gx**2) / p
            
        elif curvature_type == 'vertical_excess':
            # Vertical Excess Curvature
            curvature = -(gxx * gx**2 + 2 * gxy * gx * gy + gyy * gy**2) / p
            
        elif curvature_type == 'difference':
            # Difference Curvature
            curvature = (gxx * gy**2 - 2 * gxy * gx * gy + gyy * gx**2) / (p * root_q) - \
                        (-(gxx * gx**2 + 2 * gxy * gx * gy + gyy * gy**2) / (p * root_q3))
                        
        elif curvature_type == 'accumulation':
            # Accumulation Curvature
            curvature = (gxx * gy**2 - 2 * gxy * gx * gy + gyy * gx**2) / p - \
                        (gxx * gx**2 + 2 * gxy * gx * gy + gyy * gy**2) / p
                        
        elif curvature_type == 'curvedness':
            # Curvedness
            H = -((1 + gy**2) * gxx - 2 * gx * gy * gxy + (1 + gx**2) * gyy) / (2 * q * root_q)
            K = (gxx * gyy - gxy**2) / (q**2)
            k_max = H + np.sqrt(np.maximum(0, H**2 - K))
            k_min = H - np.sqrt(np.maximum(0, H**2 - K))
            curvature = np.sqrt((k_max**2 + k_min**2) / 2)
            
        elif curvature_type == 'unsphericity':
            # Unsphericity
            H = -((1 + gy**2) * gxx - 2 * gx * gy * gxy + (1 + gx**2) * gyy) / (2 * q * root_q)
            K = (gxx * gyy - gxy**2) / (q**2)
            curvature = np.sqrt(np.maximum(0, H**2 - K))
            
        elif curvature_type == 'rotor':
            # Rotor
            curvature = ((gxx - gyy)**2 + 4 * gxy**2) / 4
            
        elif curvature_type == 'shape_index':
            # Shape Index
            H = -((1 + gy**2) * gxx - 2 * gx * gy * gxy + (1 + gx**2) * gyy) / (2 * q * root_q)
            K = (gxx * gyy - gxy**2) / (q**2)
            k_max = H + np.sqrt(np.maximum(0, H**2 - K))
            k_min = H - np.sqrt(np.maximum(0, H**2 - K))
            curvature = (2 / np.pi) * np.arctan((k_max + k_min) / (k_max - k_min))
            
        elif curvature_type == 'ring':
            # Ring Curvature
            H = -((1 + gy**2) * gxx - 2 * gx * gy * gxy + (1 + gx**2) * gyy) / (2 * q * root_q)
            K = (gxx * gyy - gxy**2) / (q**2)
            curvature = 2 * H * K
            
        else:
            # Default to Total
            curvature = gxx + gyy
        
        # Handle division by zero and invalid values
        curvature[np.isinf(curvature)] = 0
        curvature[np.isnan(self.array)] = np.nan
        
        return curvature
    
    def calculate_tpi(self, radius=3):
        """Calculate Topographic Position Index.
        
        Args:
            radius (int): Neighborhood radius in cells
            
        Returns:
            np.ndarray: TPI values
        """
        # Create circular kernel
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        kernel = x**2 + y**2 <= radius**2
        kernel = kernel.astype(float)
        kernel[radius, radius] = 0  # Exclude center cell
        kernel /= kernel.sum()  # Normalize
        
        # Calculate mean elevation in neighborhood
        mean_elev = ndimage.convolve(self.array, kernel, mode='nearest')
        
        # TPI = elevation - mean neighborhood elevation
        tpi = self.array - mean_elev
        tpi[np.isnan(self.array)] = np.nan
        
        return tpi
    

    def calculate_roughness(self, window_size=3):
        """Calculate terrain roughness (max - min elevation)."""
        from .array_utils import focal_statistics
        
        min_elev = focal_statistics(self.array, window_size, 'min')
        max_elev = focal_statistics(self.array, window_size, 'max')
        
        roughness = max_elev - min_elev
        roughness[np.isnan(self.array)] = np.nan
        
        return roughness

    def calculate_multiscale_roughness(self, min_radius=1, max_radius=5):
        """Calculate Multiscale Roughness (Maximum Roughness across scales).
        
        Args:
            min_radius (int): Minimum window radius
            max_radius (int): Maximum window radius
            
        Returns:
            np.ndarray: Max roughness magnitude
        """
        rows, cols = self.array.shape
        max_roughness = np.zeros((rows, cols), dtype=np.float32)
        
        for r in range(min_radius, max_radius + 1):
            window_size = r * 2 + 1
            # Calculate roughness for this scale
            # We can use the existing calculate_roughness
            roughness = self.calculate_roughness(window_size)
            
            # Update max
            max_roughness = np.maximum(max_roughness, roughness)
            
        return max_roughness

    def calculate_surface_area_ratio(self):
        """Calculate Surface Area Ratio (SAR).
        
        SAR = Surface Area / Planimetric Area
        SAR = 1 / cos(slope)
        
        Returns:
            np.ndarray: SAR values (>= 1.0)
        """
        # Calculate slope in radians
        slope_rad = self.calculate_slope(units='radians')
        
        # SAR = 1 / cos(slope)
        # Avoid division by zero (though cos(slope) shouldn't be 0 for valid terrain)
        sar = 1.0 / np.cos(slope_rad)
        
        return sar

    def calculate_edge_density(self, window_size=3, threshold=15):
        """Calculate Edge Density.
        
        Density of cells with slope > threshold within window.
        
        Args:
            window_size (int): Window size (odd)
            threshold (float): Slope threshold in degrees
            
        Returns:
            np.ndarray: Edge density (0-1)
        """
        from .array_utils import focal_statistics
        
        # Calculate slope in degrees
        slope = self.calculate_slope(units='degrees')
        
        # Identify edges
        edges = (slope > threshold).astype(float)
        
        # Calculate density (mean of edges)
        density = focal_statistics(edges, window_size, 'mean')
        
        density[np.isnan(self.array)] = np.nan
        
        return density

    def detect_features(self, feature_type='peak', neighbors=8):
        """Detect geomorphometric features (Peak, Valley, Saddle).
        
        Args:
            feature_type (str): 'peak', 'valley', 'saddle', 'ridge', 'channel'
            neighbors (int): 8 (standard) or extended neighborhood
            
        Returns:
            np.ndarray: Boolean array (1 for feature, 0 otherwise)
        """
        rows, cols = self.array.shape
        result = np.zeros((rows, cols), dtype=np.uint8)
        
        if feature_type == 'ridge':
            # Ridge: High Profile Curvature (Convex) + High Plan Curvature (Divergent)
            # Simplified: Profile Curvature > 0.1
            prof = self.calculate_curvature('profile')
            result = (prof > 0.1).astype(np.uint8)
            
        elif feature_type == 'channel':
            # Channel: Low Profile Curvature (Concave) + Low Plan Curvature (Convergent)
            # Simplified: Profile Curvature < -0.1
            prof = self.calculate_curvature('profile')
            result = (prof < -0.1).astype(np.uint8)
            
        else:
            # Peak, Valley, Saddle using neighbor comparison
            return _detect_features_numba(self.array, feature_type)
            
        return result

    def classify_landforms_pennock(self, slope_thresh=3.0, curv_thresh=0.1):
        """Classify landforms using Pennock et al. (1987).
        
        Classes:
        1: Convergent Shoulder
        2: Divergent Shoulder
        3: Convergent Backslope
        4: Divergent Backslope
        5: Convergent Footslope
        6: Divergent Footslope
        7: Level
        
        Args:
            slope_thresh (float): Slope threshold in degrees
            curv_thresh (float): Curvature threshold
            
        Returns:
            np.ndarray: Landform classes (1-7)
        """
        slope = self.calculate_slope(units='degrees')
        prof = self.calculate_curvature('profile')
        plan = self.calculate_curvature('plan')
        
        rows, cols = self.array.shape
        classes = np.zeros((rows, cols), dtype=np.uint8)
        
        # Level (7)
        is_level = slope < slope_thresh
        classes[is_level] = 7
        
        # Non-level
        not_level = ~is_level
        
        # Shoulders (Convex Profile)
        is_shoulder = not_level & (prof > curv_thresh)
        classes[is_shoulder & (plan < -curv_thresh)] = 1 # Convergent Shoulder
        classes[is_shoulder & (plan > curv_thresh)] = 2 # Divergent Shoulder
        
        # Footslopes (Concave Profile)
        is_footslope = not_level & (prof < -curv_thresh)
        classes[is_footslope & (plan < -curv_thresh)] = 5 # Convergent Footslope
        classes[is_footslope & (plan > curv_thresh)] = 6 # Divergent Footslope
        
        # Backslopes (Linear Profile)
        is_backslope = not_level & (np.abs(prof) <= curv_thresh)
        classes[is_backslope & (plan < -curv_thresh)] = 3 # Convergent Backslope
        classes[is_backslope & (plan > curv_thresh)] = 4 # Divergent Backslope
        
        # Fill remaining non-level with nearest class or default?
        # Pennock usually covers all if thresholds are 0, but with thresholds there are gaps.
        # We leave 0 for undefined/intermediate.
        
        classes[np.isnan(self.array)] = 0
        
        return classes

    def calculate_hypsometric_integral(self, window_size=3):
        """Calculate local hypsometric integral.
        
        HI = (Mean - Min) / (Max - Min)
        
        Args:
            window_size (int): Window size (must be odd)
            
        Returns:
            np.ndarray: Hypsometric Integral values (0-1)
        """
        from .array_utils import focal_statistics
        
        mean_elev = focal_statistics(self.array, window_size, 'mean')
        min_elev = focal_statistics(self.array, window_size, 'min')
        max_elev = focal_statistics(self.array, window_size, 'max')
        
        range_elev = max_elev - min_elev
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            hi = (mean_elev - min_elev) / range_elev
            hi[range_elev == 0] = 0.5  # Flat area
            
        hi[np.isnan(self.array)] = np.nan
        
        return hi

    def calculate_viewshed(self, observer_x, observer_y, observer_height=1.8, target_height=0.0):
        """Calculate viewshed from a single observer point.
        
        Args:
            observer_x (float): Observer X coordinate (projected)
            observer_y (float): Observer Y coordinate (projected)
            observer_height (float): Height of observer above surface
            target_height (float): Height of target above surface
            
        Returns:
            np.ndarray: Boolean visibility raster (1=Visible, 0=Invisible)
        """
        gt = self.geotransform
        c = int((observer_x - gt[0]) / gt[1])
        r = int((observer_y - gt[3]) / gt[5])
        
        rows, cols = self.array.shape
        
        if not (0 <= r < rows and 0 <= c < cols):
            # Observer outside DEM
            return np.zeros_like(self.array, dtype=np.uint8)
            
        # Run Numba viewshed
        return _viewshed_numba(self.array, r, c, observer_height, target_height, self.cellsize_x)

    def calculate_relative_position(self, radius=3, method='minmax'):
        """Calculate relative topographic position.
        
        Args:
            radius (int): Neighborhood radius
            method (str): 'minmax' (RTP), 'diff' (Diff from Mean), 'dev' (Dev from Mean)
            
        Returns:
            np.ndarray: Result array
        """
        from .array_utils import focal_statistics
        
        window_size = radius * 2 + 1
        
        if method == 'minmax':
            # Relative Topographic Position: (Elev - Min) / (Max - Min)
            min_elev = focal_statistics(self.array, window_size, 'min')
            max_elev = focal_statistics(self.array, window_size, 'max')
            rng = max_elev - min_elev
            
            with np.errstate(divide='ignore', invalid='ignore'):
                rtp = (self.array - min_elev) / rng
                rtp[rng == 0] = 0.5
                
            rtp[np.isnan(self.array)] = np.nan
            return rtp
            
        elif method == 'diff':
            # Difference from Mean Elevation (Residual)
            mean_elev = focal_statistics(self.array, window_size, 'mean')
            diff = self.array - mean_elev
            diff[np.isnan(self.array)] = np.nan
            return diff
            
        elif method == 'dev':
            # Deviation from Mean Elevation (Standardized): (Elev - Mean) / StdDev
            mean_elev = focal_statistics(self.array, window_size, 'mean')
            # Efficient StdDev calculation?
            # StdDev = sqrt(E[x^2] - (E[x])^2)
            # This requires focal mean of square
            
            # Let's use a simpler approach or Numba if needed.
            # For now, let's use the simple difference if stddev is too expensive
            # Or implement a Numba stddev helper.
            
            # Let's stick to simple Difference for now as 'dev' usually implies standardized.
            # We'll implement a basic version: (Elev - Mean) / Range?
            # Or just return Difference.
            # Actually, 'Deviation from Mean' often just means Difference.
            # 'Difference from Mean' might mean standardized.
            # Let's implement 'standardized' as (Elev - Mean) / Range (easier than stddev)
            
            # Re-reading WBT docs:
            # DiffFromMeanElev: Elev - Mean
            # DevFromMeanElev: (Elev - Mean) / StdDev
            
            # We need StdDev.
            mean_sq = focal_statistics(self.array**2, window_size, 'mean')
            mean_elev = focal_statistics(self.array, window_size, 'mean')
            var = mean_sq - mean_elev**2
            std_dev = np.sqrt(np.maximum(var, 0))
            
            diff = self.array - mean_elev
            
            with np.errstate(divide='ignore', invalid='ignore'):
                dev = diff / std_dev
                dev[std_dev == 0] = 0
                
            dev[np.isnan(self.array)] = np.nan
            return dev
            
        return np.zeros_like(self.array)

    def calculate_ruggedness_index(self, radius=3):
        """Calculate Terrain Ruggedness Index (TRI).
        
        TRI = Mean of absolute differences between center and neighbors.
        """
        # This is best done with Numba
        return _tri_numba(self.array, radius)

@jit(nopython=True)
def _tri_numba(dem, radius):
    """Calculate Terrain Ruggedness Index."""
    rows, cols = dem.shape
    tri = np.zeros((rows, cols), dtype=np.float32)
    
    for r in range(rows):
        for c in range(cols):
            z = dem[r, c]
            if np.isnan(z):
                tri[r, c] = np.nan
                continue
                
            sum_diff = 0.0
            count = 0
            
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if i == 0 and j == 0:
                        continue
                        
                    nr, nc = r + i, c + j
                    if 0 <= nr < rows and 0 <= nc < cols:
                        nz = dem[nr, nc]
                        if not np.isnan(nz):
                            sum_diff += abs(z - nz)
                            count += 1
                            
            if count > 0:
                tri[r, c] = sum_diff / count
            else:
                tri[r, c] = 0.0
                
    return tri

@jit(nopython=True)
def _detect_features_numba(dem, feature_type):
    """Detect features using local neighborhood (Numba optimized)."""
    rows, cols = dem.shape
    result = np.zeros((rows, cols), dtype=np.uint8)
    
    # 0=Peak, 1=Valley, 2=Saddle (Simplified)
    ftype = 0
    if feature_type == 'valley':
        ftype = 1
    elif feature_type == 'saddle':
        ftype = 2
        
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            z = dem[r, c]
            if np.isnan(z):
                continue
                
            # Get neighbors
            n = [
                dem[r-1, c],   # N
                dem[r-1, c+1], # NE
                dem[r, c+1],   # E
                dem[r+1, c+1], # SE
                dem[r+1, c],   # S
                dem[r+1, c-1], # SW
                dem[r, c-1],   # W
                dem[r-1, c-1]  # NW
            ]
            
            # Check for NaN neighbors
            has_nan = False
            for val in n:
                if np.isnan(val):
                    has_nan = True
                    break
            if has_nan:
                continue
                
            if ftype == 0: # Peak: Higher than all neighbors
                is_peak = True
                for val in n:
                    if z <= val:
                        is_peak = False
                        break
                if is_peak:
                    result[r, c] = 1
                    
            elif ftype == 1: # Valley: Lower than all neighbors
                is_valley = True
                for val in n:
                    if z >= val:
                        is_valley = False
                        break
                if is_valley:
                    result[r, c] = 1
                    
            elif ftype == 2: # Saddle: 2 opposite high, 2 opposite low (Simplified)
                # This is a basic topological saddle detection
                # Count sign changes in difference around the center?
                # Or simply: check pairs of opposite neighbors
                
                # A common definition: > 2 neighbors and < 2 neighbors?
                # Let's use a crossing number method or simple opposite pair check
                
                high_count = 0
                low_count = 0
                for val in n:
                    if z > val:
                        high_count += 1
                    elif z < val:
                        low_count += 1
                
                # Strict saddle: >= 2 high, >= 2 low, and alternating?
                # This requires more complex logic. 
                # Let's stick to a simpler "Pass" definition for now or implement "Mean Difference"
                if high_count >= 2 and low_count >= 2:
                    result[r, c] = 1
                    
@jit(nopython=True)
def _viewshed_numba(dem, obs_r, obs_c, obs_h, target_h, cellsize):
    """Calculate viewshed using R3 algorithm (Reference Plane) or simple LOS."""
    rows, cols = dem.shape
    visible = np.zeros((rows, cols), dtype=np.uint8)
    
    obs_z = dem[obs_r, obs_c] + obs_h
    
    for r in range(rows):
        for c in range(cols):
            if r == obs_r and c == obs_c:
                visible[r, c] = 1
                continue
                
            if np.isnan(dem[r, c]):
                continue
                
            dr = r - obs_r
            dc = c - obs_c
            dist = np.sqrt(dr**2 + dc**2)
            steps = int(dist)
            
            if steps == 0:
                visible[r, c] = 1
                continue
                
            step_r = dr / steps
            step_c = dc / steps
            
            target_z = dem[r, c] + target_h
            slope_to_target = (target_z - obs_z) / dist
            
            is_visible = True
            
            for i in range(1, steps):
                curr_r = obs_r + i * step_r
                curr_c = obs_c + i * step_c
                
                ir = int(round(curr_r))
                ic = int(round(curr_c))
                
                if 0 <= ir < rows and 0 <= ic < cols:
                    z = dem[ir, ic]
                    if not np.isnan(z):
                        los_z = obs_z + slope_to_target * i
                        if z > los_z:
                            is_visible = False
                            break
            
            if is_visible:
                visible[r, c] = 1
                
    return visible

@jit(nopython=True)
def _circular_variance_numba(aspect, window_size):
    """Calculate circular variance of aspect."""
    rows, cols = aspect.shape
    variance = np.zeros((rows, cols), dtype=np.float32)
    radius = window_size // 2
    
    for r in range(rows):
        for c in range(cols):
            if np.isnan(aspect[r, c]):
                variance[r, c] = np.nan
                continue
                
            sum_sin = 0.0
            sum_cos = 0.0
            count = 0
            
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    nr, nc = r + i, c + j
                    if 0 <= nr < rows and 0 <= nc < cols:
                        val = aspect[nr, nc]
                        if val != -1 and not np.isnan(val):
                            rad = np.radians(val)
                            sum_sin += np.sin(rad)
                            sum_cos += np.cos(rad)
                            count += 1
                            
            if count > 0:
                mean_sin = sum_sin / count
                mean_cos = sum_cos / count
                R = np.sqrt(mean_sin**2 + mean_cos**2)
                variance[r, c] = 1.0 - R
            else:
                variance[r, c] = np.nan # Or 0?
                
    return variance
    def calculate_directional_relief(self, azimuth, radius=3):
        '''Calculate directional relief (shadowing effect).
        
        Args:
            azimuth (float): Direction in degrees
            radius (int): Search radius
            
        Returns:
            np.ndarray: Directional relief raster
        '''
        # Convert azimuth to radians
        az_rad = np.radians(azimuth)
        
        # We can implement this as a directional slope or relative aspect?
        # Whitebox DirectionalRelief: "Calculates relief shading for a specified azimuth"
        # This sounds like Hillshade but maybe focusing on relief relative to direction.
        # Or "relief" as in elevation difference in a direction?
        # Let's implement a "Directional Slope" or "Relief" which is the slope in the direction of azimuth.
        
        # Slope in direction alpha:
        # tan(slope_alpha) = tan(slope) * cos(aspect - alpha)
        
        slope = self.calculate_slope(units='radians')
        aspect = self.calculate_aspect(units='radians')
        
        with np.errstate(invalid='ignore'):
            # Aspect is usually North=0, CW.
            # Math usually East=0, CCW.
            # But if both are consistent (North=0, CW), the difference works.
            
            # Directional slope
            # We want positive for upslope, negative for downslope?
            # Or just the magnitude?
            # Usually "Relief" implies visual effect.
            
            # Let's calculate the component of the gradient in the direction of azimuth.
            # Gradient vector G = (dz/dx, dz/dy)
            # Unit vector U = (sin(az), cos(az)) (if North=0, CW)
            # Directional derivative = G . U
            
            # We have slope and aspect.
            # Gradient magnitude = tan(slope)
            # Gradient direction = aspect
            
            # Directional derivative = tan(slope) * cos(aspect - azimuth)
            
            val = np.tan(slope) * np.cos(aspect - az_rad)
            
        return val

    def calculate_wind_exposure(self, wind_azimuth, max_dist):
        '''Calculate wind exposure index.
        
        Args:
            wind_azimuth (float): Wind direction in degrees
            max_dist (float): Maximum search distance
            
        Returns:
            np.ndarray: Wind exposure raster
        '''
        # This usually involves checking upwind sheltering.
        # "Fetch" or "Exposure".
        # A simple index: average angle to horizon in the upwind direction?
        # Or "Winstral's Sx" parameter? (Max upwind slope)
        
        # Let's implement Winstral's Sx (Maximum upwind slope).
        # Sx = max(tan(slope) in upwind direction within distance)
        # Actually it's max angle to horizon in upwind direction.
        # If max angle is negative, it's exposed (or rather, not sheltered).
        # Higher positive angle = more sheltered.
        # Lower (negative) = more exposed.
        
        # We need a Numba helper for "Max Upwind Horizon Angle".
        
        return _wind_exposure_numba(
            self.array,
            self.nodata,
            self.cellsize_x,
            wind_azimuth,
            max_dist
        )

    def calculate_openness(self, radius, openness_type='positive'):
        '''Calculate topographic openness.
        
        Args:
            radius (int): Search radius in cells
            openness_type (str): 'positive' (sky view) or 'negative' (sub-surface)
            
        Returns:
            np.ndarray: Openness raster (radians)
        '''
        return _openness_numba(
            self.array,
            self.cellsize_x,
            radius,
            1 if openness_type == 'positive' else -1
        )

@jit(nopython=True)
def _wind_exposure_numba(dem, nodata, cellsize, azimuth, max_dist):
    '''Calculate Winstral Sx (Maximum Upwind Slope).'''
    rows, cols = dem.shape
    sx = np.zeros((rows, cols), dtype=np.float32) + np.nan
    
    # Direction vector
    az_rad = np.radians(azimuth)
    dx = np.sin(az_rad)
    dy = np.cos(az_rad) # North is positive Y? No, usually row 0 is North.
    # If row 0 is North (top), then North is -y (decreasing row).
    # So dy = -cos(az_rad)
    dy = -np.cos(az_rad)
    
    # Normalize step to 1 cell
    # We want to step in grid cells.
    # Bresenham or simple stepping?
    # Simple stepping with interpolation is better but slow.
    # Let's use nearest neighbor stepping.
    
    # Step size
    step_dist = cellsize
    max_steps = int(max_dist / step_dist)
    
    for r in range(rows):
        for c in range(cols):
            z0 = dem[r, c]
            if np.isnan(z0):
                continue
                
            max_angle = -9999.0
            
            # Trace upwind (opposite to wind direction)
            # Wind comes FROM azimuth.
            # So we look in direction of azimuth to see if there is a blocker?
            # No, if wind is FROM North, we look North (upwind) to see if we are sheltered.
            # So we trace in direction (dx, dy).
            
            curr_r = float(r)
            curr_c = float(c)
            
            for i in range(1, max_steps + 1):
                curr_r += dy
                curr_c += dx
                
                ir = int(round(curr_r))
                ic = int(round(curr_c))
                
                if 0 <= ir < rows and 0 <= ic < cols:
                    z1 = dem[ir, ic]
                    if not np.isnan(z1):
                        dist = i * step_dist
                        # Angle = (z1 - z0) / dist (approx tan)
                        angle = (z1 - z0) / dist
                        if angle > max_angle:
                            max_angle = angle
                else:
                    break
            
            if max_angle != -9999.0:
                sx[r, c] = max_angle
            else:
                # No data found upwind? Assume flat?
                sx[r, c] = 0.0
                
    return sx

@jit(nopython=True)
def _openness_numba(dem, cellsize, radius, direction):
    '''Calculate topographic openness (Numba optimized).'''
    rows, cols = dem.shape
    openness = np.zeros((rows, cols), dtype=np.float32)
    
    # 8 cardinal directions
    angles = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    rads = np.radians(angles)
    dxs = np.sin(rads)
    dys = -np.cos(rads) # North is -y
    
    for r in range(rows):
        for c in range(cols):
            z0 = dem[r, c]
            if np.isnan(z0):
                openness[r, c] = np.nan
                continue
            
            total_angle = 0.0
            
            for k in range(8):
                # Trace in direction k
                dx = dxs[k]
                dy = dys[k]
                
                # Openness is mean of zenith angles (angle from vertical? or horizontal?)
                # Usually angle from horizontal.
                # Positive Openness: Angle to sky. 90 is straight up.
                # Horizon angle is angle from horizontal to terrain.
                # Openness = 90 - max_horizon_angle
                
                curr_max_angle = -90.0
                
                for i in range(1, radius + 1):
                    # Nearest neighbor sampling
                    nr = int(round(r + i * dy))
                    nc = int(round(c + i * dx))
                    
                    if 0 <= nr < rows and 0 <= nc < cols:
                        z = dem[nr, nc]
                        if not np.isnan(z):
                            dist = i * cellsize
                            # Angle from horizontal
                            angle = np.arctan((z - z0) / dist) * 180.0 / np.pi
                            
                            if direction > 0: # Positive
                                if angle > curr_max_angle:
                                    curr_max_angle = angle
                            else: # Negative
                                if -angle > curr_max_angle:
                                    curr_max_angle = -angle
                    else:
                        break
                
                # Zenith angle = 90 - horizon angle
                zenith = 90.0 - curr_max_angle
                total_angle += zenith
            
            openness[r, c] = np.radians(total_angle / 8.0)
            
    return openness

@jit(nopython=True)
def _detect_features_impl(dem, code):
    rows, cols = dem.shape
    result = np.zeros((rows, cols), dtype=np.uint8)
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            z = dem[r, c]
            if np.isnan(z):
                continue
            
            # Compare with 8 neighbors
            # Count higher and lower
            higher = 0
            lower = 0
            
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    nz = dem[r + dr, c + dc]
                    if not np.isnan(nz):
                        if nz > z:
                            higher += 1
                        elif nz < z:
                            lower += 1
            
            if code == 1: # Peak
                if lower == 8: # All neighbors lower
                    result[r, c] = 1
            elif code == 2: # Valley
                if higher == 8: # All neighbors higher
                    result[r, c] = 1
            elif code == 3: # Saddle
                # Simple topological saddle:
                if higher >= 2 and lower >= 2:
                    result[r, c] = 1
                    
    return result

def _detect_features_numba(dem, feature_type):
    # Wrapper to convert string to code
    code = 0
    if feature_type == 'peak':
        code = 1
    elif feature_type == 'valley':
        code = 2
    elif feature_type == 'saddle':
        code = 3
        
    return _detect_features_impl(dem, code)
