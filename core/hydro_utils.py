# -*- coding: utf-8 -*-
"""
Advanced hydrological analysis utilities
Native Python implementation with NumPy optimization
"""

import numpy as np
from scipy import ndimage
from collections import deque
from qgis.core import QgsProcessingException


class HydrologicalAnalyzer:
    """Advanced hydrological analysis tools."""
    
    def __init__(self, dem_array, cellsize, nodata=-9999):
        """Initialize hydrological analyzer.
        
        Args:
            dem_array (np.ndarray): DEM array
            cellsize (float): Cell size
            nodata (float): NoData value
        """
        self.dem = dem_array.copy()
        self.cellsize = cellsize
        self.nodata = nodata
        self.rows, self.cols = dem_array.shape
        
        # Replace nodata with NaN
        self.dem[self.dem == nodata] = np.nan
    
    def breach_depressions_least_cost(self, max_depth=None, max_length=None):
        """Breach depressions using least-cost path algorithm.
        
        Args:
            max_depth (float): Maximum breach depth
            max_length (float): Maximum breach length
            
        Returns:
            np.ndarray: Breached DEM
        """
        breached = self.dem.copy()
        
        # Find all depressions
        depressions = self._identify_depressions()
        
        for depression_id in np.unique(depressions[depressions > 0]):
            # Get depression cells
            dep_mask = depressions == depression_id
            dep_cells = np.argwhere(dep_mask)
            
            if len(dep_cells) == 0:
                continue
            
            # Find outlet (lowest point on edge)
            outlet = self._find_depression_outlet(dep_mask)
            
            if outlet is None:
                continue
            
            # Find breach path
            path = self._find_breach_path(dep_mask, outlet, max_depth, max_length)
            
            if path:
                # Apply breach
                for row, col in path:
                    if row > 0 and col > 0 and row < self.rows and col < self.cols:
                        # Lower cell along path
                        neighbor_min = self._get_neighbor_minimum(breached, row, col)
                        if breached[row, col] > neighbor_min:
                            breached[row, col] = neighbor_min - 0.001
        
        return breached
    
    def _identify_depressions(self):
        """Identify depression regions.
        
        Returns:
            np.ndarray: Labeled depression array
        """
        # Use morphological operations to find depressions
        filled = self._simple_fill()
        diff = filled - self.dem
        
        # Label connected depressions
        depressions = np.zeros_like(self.dem, dtype=np.int32)
        depression_mask = diff > 0.001
        
        labeled, num_features = ndimage.label(depression_mask)
        
        return labeled
    
    def _simple_fill(self):
        """Simple depression filling.
        
        Returns:
            np.ndarray: Filled DEM
        """
        filled = self.dem.copy()
        
        # Priority flood from edges
        from heapq import heappush, heappop
        
        processed = np.zeros_like(filled, dtype=bool)
        pq = []
        
        # Add edge cells
        for i in range(self.rows):
            for j in [0, self.cols - 1]:
                if not np.isnan(filled[i, j]):
                    heappush(pq, (filled[i, j], i, j))
                    processed[i, j] = True
        
        for i in [0, self.rows - 1]:
            for j in range(1, self.cols - 1):
                if not np.isnan(filled[i, j]) and not processed[i, j]:
                    heappush(pq, (filled[i, j], i, j))
                    processed[i, j] = True
        
        # Process queue
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), 
                     (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        
        while pq:
            elev, row, col = heappop(pq)
            
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if not processed[nr, nc] and not np.isnan(filled[nr, nc]):
                        if filled[nr, nc] < elev:
                            filled[nr, nc] = elev
                        
                        heappush(pq, (filled[nr, nc], nr, nc))
                        processed[nr, nc] = True
        
        return filled
    
    def _find_depression_outlet(self, depression_mask):
        """Find lowest outlet for depression.
        
        Args:
            depression_mask (np.ndarray): Boolean mask of depression
            
        Returns:
            tuple: (row, col) of outlet or None
        """
        # Find edge cells of depression
        edge_cells = []
        dep_coords = np.argwhere(depression_mask)
        
        for row, col in dep_coords:
            # Check if on edge of depression
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if not depression_mask[nr, nc] and not np.isnan(self.dem[nr, nc]):
                            edge_cells.append((row, col, self.dem[nr, nc]))
                            break
        
        if not edge_cells:
            return None
        
        # Return cell with lowest neighbor
        edge_cells.sort(key=lambda x: x[2])
        return (edge_cells[0][0], edge_cells[0][1])
    
    def _find_breach_path(self, depression_mask, outlet, max_depth, max_length):
        """Find breach path from depression to outlet.
        
        Args:
            depression_mask (np.ndarray): Depression mask
            outlet (tuple): Outlet coordinates
            max_depth (float): Maximum breach depth
            max_length (float): Maximum breach length
            
        Returns:
            list: List of (row, col) tuples forming path
        """
        # Simple path from highest point in depression to outlet
        dep_coords = np.argwhere(depression_mask)
        
        if len(dep_coords) == 0:
            return []
        
        # Find highest point
        elevations = self.dem[depression_mask]
        max_idx = np.argmax(elevations)
        start = tuple(dep_coords[max_idx])
        
        # Trace path using A* or simple line
        path = self._trace_line(start, outlet)
        
        return path
    
    def _trace_line(self, start, end):
        """Trace line between two points using Bresenham's algorithm.
        
        Args:
            start (tuple): Start (row, col)
            end (tuple): End (row, col)
            
        Returns:
            list: List of (row, col) tuples
        """
        r0, c0 = start
        r1, c1 = end
        
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        
        r_step = 1 if r0 < r1 else -1
        c_step = 1 if c0 < c1 else -1
        
        path = []
        
        if dr > dc:
            err = dr / 2
            c = c0
            for r in range(r0, r1, r_step):
                path.append((r, c))
                err -= dc
                if err < 0:
                    c += c_step
                    err += dr
        else:
            err = dc / 2
            r = r0
            for c in range(c0, c1, c_step):
                path.append((r, c))
                err -= dr
                if err < 0:
                    r += r_step
                    err += dc
        
        path.append((r1, c1))
        return path
    
    def _get_neighbor_minimum(self, array, row, col):
        """Get minimum value of 8 neighbors.
        
        Args:
            array (np.ndarray): Array
            row (int): Row index
            col (int): Column index
            
        Returns:
            float: Minimum neighbor value
        """
        min_val = np.inf
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if not np.isnan(array[nr, nc]):
                        min_val = min(min_val, array[nr, nc])
        
        return min_val if min_val != np.inf else array[row, col]
    
    def calculate_wetness_index(self, flow_acc, slope):
        """Calculate Topographic Wetness Index (TWI).
        
        TWI = ln(a / tan(slope))
        where a = specific catchment area
        
        Args:
            flow_acc (np.ndarray): Flow accumulation
            slope (np.ndarray): Slope in radians
            
        Returns:
            np.ndarray: TWI values
        """
        # Calculate specific catchment area
        specific_area = (flow_acc + 1) * self.cellsize
        
        # Avoid division by zero
        slope_safe = np.where(slope < 0.001, 0.001, slope)
        tan_slope = np.tan(slope_safe)
        
        # Calculate TWI
        twi = np.log(specific_area / tan_slope)
        
        # Mask invalid values
        twi[np.isnan(self.dem)] = np.nan
        twi[np.isinf(twi)] = np.nan
        
    
    def delineate_watershed_from_point(self, flow_dir, pour_point_row, pour_point_col):
        """Delineate watershed from a pour point.
        
        Args:
            flow_dir (np.ndarray): Flow direction array (D8)
            pour_point_row (int): Pour point row
            pour_point_col (int): Pour point column
            
        Returns:
            np.ndarray: Binary watershed mask
        """
        watershed = np.zeros_like(self.dem, dtype=np.uint8)
        
        # Direction codes
        dir_codes = [1, 2, 4, 8, 16, 32, 64, 128]
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), 
                     (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        
        # BFS to find all cells draining to pour point
        to_process = deque([(pour_point_row, pour_point_col)])
        processed = set()
        
        while to_process:
            row, col = to_process.popleft()
            
            if (row, col) in processed:
                continue
            
            processed.add((row, col))
            watershed[row, col] = 1
            
            # Find cells that flow into this cell
            for idx, (dr, dc) in enumerate(directions):
                nr, nc = row - dr, col - dc
                
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if flow_dir[nr, nc] == dir_codes[idx]:
                        if (nr, nc) not in processed:
                            to_process.append((nr, nc))
        
        return watershed

    def calculate_valley_index(self, flow_acc, slope):
        """Calculate Valley Index (simplified MRVBF-like).
        
        Index = (1 - Slope/MaxSlope) * log(Accumulation)
        
        Args:
            flow_acc (np.ndarray): Flow accumulation
            slope (np.ndarray): Slope in degrees
            
        Returns:
            np.ndarray: Valley Index
        """
        # Normalize slope
        max_slope = np.nanmax(slope)
        if max_slope == 0:
            max_slope = 1.0
            
        norm_slope = slope / max_slope
        
        # Log accumulation
        log_acc = np.log1p(flow_acc)
        
        # Valley Index
        # High acc, low slope -> High index
        # (1 - norm_slope) is high for flat areas
        
        vi = (1.0 - norm_slope) * log_acc
        
        # Handle NaNs
        vi[np.isnan(self.dem)] = np.nan
        
        return vi
