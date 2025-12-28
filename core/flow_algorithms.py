# -*- coding: utf-8 -*-
"""
Native hydrological flow algorithms using NumPy and Numba
Implements D8, D-Infinity, and other flow routing methods
"""

import numpy as np

from qgis.core import QgsProcessingException, QgsMessageLog, Qgis

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorators/functions for fallback
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    prange = range

# Numba-optimized static functions
# Note: Fill depressions uses sorted-cell approach for maximum performance

@jit(nopython=True, cache=True)
def _fill_depressions_core(filled, order, rows, cols, epsilon):
    """Core fill logic - processes cells in elevation order (lowest to highest).
    
    This is O(N) after the sort is done externally.
    """
    drs = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
    dcs = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
    
    n = len(order)
    for idx in range(n):
        cell_idx = order[idx]
        r = cell_idx // cols
        c = cell_idx % cols
        
        current_elev = filled[r, c]
        
        # Find minimum neighbor elevation
        min_neighbor = np.inf
        for i in range(8):
            nr = r + drs[i]
            nc = c + dcs[i]
            if 0 <= nr < rows and 0 <= nc < cols:
                if not np.isnan(filled[nr, nc]):
                    if filled[nr, nc] < min_neighbor:
                        min_neighbor = filled[nr, nc]
        
        # If cell is lower than all neighbors, it's in a depression
        # Raise it to minimum neighbor + epsilon
        if min_neighbor != np.inf and current_elev < min_neighbor:
            filled[r, c] = min_neighbor + epsilon
    
    return filled


def _fill_depressions_fast(dem, epsilon=0.001):
    """Fill depressions using fast sorted-cell approach.
    
    Algorithm:
    1. Sort all cells by elevation (O(N log N) - very fast in NumPy)
    2. Iterate in order from lowest to highest (O(N))
    3. For each cell, if lower than all neighbors, raise to min neighbor + epsilon
    4. Repeat until no changes (usually 2-3 passes)
    
    This is much faster than heapq because:
    - NumPy sort is highly optimized C code
    - Main loop is Numba-compiled
    - No Python object overhead per cell
    """
    rows, cols = dem.shape
    filled = dem.copy()
    
    # Handle NaN by replacing with infinity for sorting
    has_nan = np.isnan(filled)
    filled_for_sort = np.where(has_nan, np.inf, filled)
    
    # Create flat indices sorted by elevation
    flat_indices = np.arange(rows * cols, dtype=np.int32)
    order = np.argsort(filled_for_sort.ravel()).astype(np.int32)
    
    # Filter out NaN cells from processing
    valid_mask = ~has_nan.ravel()
    order = order[valid_mask[order]]
    
    # Iterative filling - may need multiple passes
    max_iterations = 10
    for iteration in range(max_iterations):
        old_filled = filled.copy()
        filled = _fill_depressions_core(filled, order, rows, cols, epsilon)
        
        # Check for convergence
        diff = np.abs(filled - old_filled)
        max_diff = np.nanmax(diff)
        if max_diff < epsilon / 2:
            break
    
    return filled


# Keep heapq version as fallback (not used by default)
def _fill_depressions_heapq(dem, epsilon=0.001):
    """Fallback: Fill depressions using Python heapq (slower but reliable)."""
    from heapq import heappush, heappop
    
    rows, cols = dem.shape
    filled = dem.copy()
    processed = np.zeros((rows, cols), dtype=bool)
    priority_queue = []
    
    drs = [0, 1, 1, 1, 0, -1, -1, -1]
    dcs = [1, 1, 0, -1, -1, -1, 0, 1]
    
    # Initialize with edge cells
    for i in range(rows):
        for j in [0, cols - 1]:
            if not np.isnan(filled[i, j]):
                heappush(priority_queue, (filled[i, j], i, j))
                processed[i, j] = True
    
    for j in range(1, cols - 1):
        for i in [0, rows - 1]:
            if not np.isnan(filled[i, j]) and not processed[i, j]:
                heappush(priority_queue, (filled[i, j], i, j))
                processed[i, j] = True
    
    while priority_queue:
        elev, r, c = heappop(priority_queue)
        
        for i in range(8):
            nr = r + drs[i]
            nc = c + dcs[i]
            
            if 0 <= nr < rows and 0 <= nc < cols:
                if not processed[nr, nc] and not np.isnan(filled[nr, nc]):
                    if filled[nr, nc] < elev:
                        filled[nr, nc] = elev + epsilon
                    
                    heappush(priority_queue, (filled[nr, nc], nr, nc))
                    processed[nr, nc] = True
    
    return filled


def _resolve_flats_toward_outlets(dem, epsilon=0.00001):
    """Resolve flat areas by creating micro-gradients toward outlets.
    
    This implements a simplified version of the Garbrecht-Martz algorithm
    that routes flow from higher ground toward outlets through flat areas.
    
    Args:
        dem: Filled DEM with flat areas
        epsilon: Small elevation increment for gradient creation
        
    Returns:
        Modified DEM with micro-gradients in flat areas
    """
    from collections import deque
    
    rows, cols = dem.shape
    resolved = dem.copy()
    
    # Find flat cells (cells where no neighbor is lower)
    flat_mask = np.zeros((rows, cols), dtype=np.bool_)
    
    drs = [0, 1, 1, 1, 0, -1, -1, -1]
    dcs = [1, 1, 0, -1, -1, -1, 0, 1]
    
    for r in range(rows):
        for c in range(cols):
            if np.isnan(dem[r, c]):
                continue
            
            current_elev = dem[r, c]
            has_lower = False
            
            for i in range(8):
                nr, nc = r + drs[i], c + dcs[i]
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not np.isnan(dem[nr, nc]) and dem[nr, nc] < current_elev:
                        has_lower = True
                        break
            
            if not has_lower:
                flat_mask[r, c] = True
    
    # Create gradient away from higher ground using BFS from non-flat edges
    # First pass: identify edge cells of flat areas (flat cells adjacent to lower cells)
    edge_cells = deque()
    distance = np.full((rows, cols), -1, dtype=np.int32)
    
    for r in range(rows):
        for c in range(cols):
            if flat_mask[r, c]:
                current_elev = dem[r, c]
                for i in range(8):
                    nr, nc = r + drs[i], c + dcs[i]
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not np.isnan(dem[nr, nc]) and dem[nr, nc] < current_elev:
                            edge_cells.append((r, c))
                            distance[r, c] = 0
                            break
    
    # BFS to compute distance from outlet edge
    while edge_cells:
        r, c = edge_cells.popleft()
        current_dist = distance[r, c]
        current_elev = dem[r, c]
        
        for i in range(8):
            nr, nc = r + drs[i], c + dcs[i]
            if 0 <= nr < rows and 0 <= nc < cols:
                if flat_mask[nr, nc] and distance[nr, nc] == -1:
                    # Only propagate to cells at same elevation
                    if abs(dem[nr, nc] - current_elev) < epsilon * 10:
                        distance[nr, nc] = current_dist + 1
                        edge_cells.append((nr, nc))
    
    # Apply micro-gradients based on distance
    max_dist = np.max(distance)
    if max_dist > 0:
        for r in range(rows):
            for c in range(cols):
                if distance[r, c] > 0:
                    # Add small increment proportional to distance from outlet
                    resolved[r, c] += distance[r, c] * epsilon
    
    return resolved


# Stub for backward compatibility
@jit(nopython=True, cache=True)
def _fill_depressions_numba(dem, epsilon=0.001):
    """Stub - actual implementation uses _fill_depressions_fast."""
    return dem.copy()

@jit(nopython=True, parallel=True, cache=True)
def _d8_flow_direction_numba(dem, nodata_val, cellsize_x, cellsize_y):
    """Calculate D8 flow direction (Numba optimized).
    
    Returns flow direction using ArcGIS D8 encoding:
    - 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
    
    Uses steepest descent among valid neighbors.
    For flat areas, flows to the lowest neighbor.
    """
    rows, cols = dem.shape
    flow_dir = np.zeros((rows, cols), dtype=np.int32)
    
    # Standard Directions: E, SE, S, SW, W, NW, N, NE (Ascending Codes 1..128)
    drs = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dcs = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    codes = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    
    # Distances for slope calculation
    diag_dist = np.sqrt(cellsize_x**2 + cellsize_y**2)
    dists = np.array([cellsize_x, diag_dist, cellsize_y, diag_dist, 
                      cellsize_x, diag_dist, cellsize_y, diag_dist])
    
    for r in prange(rows):
        for c in range(cols):
            if np.isnan(dem[r, c]):
                continue
            
            current_elev = dem[r, c]
            max_slope = -np.inf
            direction = 0
            lowest_neighbor_elev = np.inf
            lowest_direction = 0
            
            for i in range(8):
                nr, nc = r + drs[i], c + dcs[i]
                
                # Handle boundaries
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                
                if np.isnan(dem[nr, nc]):
                    continue
                
                neighbor_elev = dem[nr, nc]
                drop = current_elev - neighbor_elev
                slope = drop / dists[i]
                
                # Track lowest neighbor for flat area handling
                if neighbor_elev < lowest_neighbor_elev:
                    lowest_neighbor_elev = neighbor_elev
                    lowest_direction = codes[i]
                
                # Select steepest slope
                # Standard Logic: Prefer first found max (Strict >)
                # Order is E -> SE -> S -> SW -> W -> NW -> N -> NE
                if slope > max_slope:
                    max_slope = slope
                    direction = codes[i]
            
            # Handling sinks (no positive slope) and flat areas
            if max_slope <= 0:
                # Flow to lowest neighbor if possible (Fill behavior)
                if lowest_direction != 0:
                    direction = lowest_direction
                
                # Edge Sink Handling removed as per user request
                # "no need to calculate flow direction values for edge pixels"
                # They will remain 0 (or lowest neighbor if found)
            
            flow_dir[r, c] = direction
            
    return flow_dir

@jit(nopython=True, cache=True)
def _strahler_order_numba(flow_dir, streams, codes, drs, dcs):
    """Calculate Strahler stream order."""
    rows, cols = flow_dir.shape
    order = np.zeros((rows, cols), dtype=np.int32)
    
    # Inflow count (only from stream cells)
    inflow_count = np.zeros((rows, cols), dtype=np.int32)
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if streams[r, c] > 0:
                d = flow_dir[r, c]
                if d > 0:
                    for i in range(8):
                        if d == codes[i]:
                            nr, nc = r + drs[i], c + dcs[i]
                            if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc] > 0:
                                inflow_count[nr, nc] += 1
                            break
    
    # Queue for leaf nodes (order 1)
    queue_r = []
    queue_c = []
    
    for r in range(rows):
        for c in range(cols):
            if streams[r, c] > 0 and inflow_count[r, c] == 0:
                order[r, c] = 1
                queue_r.append(r)
                queue_c.append(c)
    
    # Process queue
    # We need to track max order of inflows for each cell
    # Since we can't store lists in numba easily, we process in topological order
    # But Strahler requires knowing ALL inflows before computing.
    # Topological sort ensures we visit a cell only after all its inflows are visited.
    
    # For Strahler:
    # If inflows have orders i, j, ...
    # max_order = max(inflows)
    # if count(max_order) > 1: result = max_order + 1
    # else: result = max_order
    
    # To implement this, we need to store inflow orders.
    # Simplified approach:
    # Use a temporary array to store max order seen so far and count of that max order
    
    max_inflow_order = np.zeros((rows, cols), dtype=np.int32)
    count_max_order = np.zeros((rows, cols), dtype=np.int32)
    
    head = 0
    while head < len(queue_r):
        r = queue_r[head]
        c = queue_c[head]
        head += 1
        
        current_order = order[r, c]
        
        d = flow_dir[r, c]
        if d > 0:
            for i in range(8):
                if d == codes[i]:
                    nr, nc = r + drs[i], c + dcs[i]
                    if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc] > 0:
                        
                        # Update neighbor stats
                        if current_order > max_inflow_order[nr, nc]:
                            max_inflow_order[nr, nc] = current_order
                            count_max_order[nr, nc] = 1
                        elif current_order == max_inflow_order[nr, nc]:
                            count_max_order[nr, nc] += 1
                        
                        inflow_count[nr, nc] -= 1
                        if inflow_count[nr, nc] == 0:
                            # Compute order for neighbor
                            if count_max_order[nr, nc] > 1:
                                order[nr, nc] = max_inflow_order[nr, nc] + 1
                            else:
                                order[nr, nc] = max_inflow_order[nr, nc]
                            
                            queue_r.append(nr)
                            queue_c.append(nc)
                    break
                    
    return order

@jit(nopython=True, cache=True)
def _shreve_order_numba(flow_dir, streams, codes, drs, dcs):
    """Calculate Shreve stream magnitude."""
    rows, cols = flow_dir.shape
    magnitude = np.zeros((rows, cols), dtype=np.int32)
    
    # Inflow count
    inflow_count = np.zeros((rows, cols), dtype=np.int32)
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if streams[r, c] > 0:
                d = flow_dir[r, c]
                if d > 0:
                    for i in range(8):
                        if d == codes[i]:
                            nr, nc = r + drs[i], c + dcs[i]
                            if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc] > 0:
                                inflow_count[nr, nc] += 1
                            break
    
    # Queue for leaf nodes (magnitude 1)
    queue_r = []
    queue_c = []
    
    for r in range(rows):
        for c in range(cols):
            if streams[r, c] > 0 and inflow_count[r, c] == 0:
                magnitude[r, c] = 1
                queue_r.append(r)
                queue_c.append(c)
    
    # Process queue
    head = 0
    while head < len(queue_r):
        r = queue_r[head]
        c = queue_c[head]
        head += 1
        
        current_mag = magnitude[r, c]
        
        d = flow_dir[r, c]
        if d > 0:
            for i in range(8):
                if d == codes[i]:
                    nr, nc = r + drs[i], c + dcs[i]
                    if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc] > 0:
                        magnitude[nr, nc] += current_mag
                        inflow_count[nr, nc] -= 1
                        if inflow_count[nr, nc] == 0:
                            queue_r.append(nr)
                            queue_c.append(nc)
                    break
                    
    return magnitude    

    
    def d8_flow_direction(self):
        """Calculate D8 flow direction."""
        return _d8_flow_direction_numba(
            self.dem, self.nodata, self.cellsize_x, self.cellsize_y
        )
    
    def d8_flow_accumulation(self, flow_dir, weights=None):
        """Calculate D8 flow accumulation."""
        if weights is None:
            weights = np.ones_like(self.dem)
            
        return _d8_flow_accumulation_numba(
            flow_dir.astype(np.int32), 
            weights, 
            self.codes, 
            self.drs, 
            self.dcs
        )


@jit(nopython=True, cache=True)
def _shreve_order_numba(flow_dir, streams, codes, drs, dcs):
    """Calculate Shreve stream magnitude."""
    rows, cols = flow_dir.shape
    magnitude = np.zeros((rows, cols), dtype=np.int32)
    
    # Inflow count
    inflow_count = np.zeros((rows, cols), dtype=np.int32)
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if streams[r, c] > 0:
                d = flow_dir[r, c]
                if d > 0:
                    for i in range(8):
                        if d == codes[i]:
                            nr, nc = r + drs[i], c + dcs[i]
                            if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc] > 0:
                                inflow_count[nr, nc] += 1
                            break
    
    # Queue for leaf nodes (magnitude 1)
    queue_r = []
    queue_c = []
    
    for r in range(rows):
        for c in range(cols):
            if streams[r, c] > 0 and inflow_count[r, c] == 0:
                magnitude[r, c] = 1
                queue_r.append(r)
                queue_c.append(c)
    
    # Process queue
    head = 0
    while head < len(queue_r):
        r = queue_r[head]
        c = queue_c[head]
        head += 1
        
        current_mag = magnitude[r, c]
        
        d = flow_dir[r, c]
        if d > 0:
            for i in range(8):
                if d == codes[i]:
                    nr, nc = r + drs[i], c + dcs[i]
                    if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc] > 0:
                        magnitude[nr, nc] += current_mag
                        inflow_count[nr, nc] -= 1
                        if inflow_count[nr, nc] == 0:
                            queue_r.append(nr)
                            queue_c.append(nc)
                    break
                    
    return magnitude    
    # Process queue
    head = 0
    while head < len(queue_r):
        r, c = queue_r[head], queue_c[head]
        head += 1
        
        d = flow_dir[r, c]
        if d > 0:
            for i in range(8):
                if d == codes[i]:
                    nr, nc = r + drs[i], c + dcs[i]
                    if 0 <= nr < rows and 0 <= nc < cols:
                        acc[nr, nc] += acc[r, c]
                        inflow_count[nr, nc] -= 1
                        if inflow_count[nr, nc] == 0:
                            queue_r.append(nr)
                            queue_c.append(nc)
                    break
                    
    return acc


@jit(nopython=True, cache=True)
def _d8_flow_accumulation_numba(flow_dir, weights, codes, drs, dcs):
    """Calculate D8 flow accumulation (Numba optimized)."""
    rows, cols = flow_dir.shape
    n_cells = rows * cols
    acc = weights.astype(np.float64)
    inflow_count = np.zeros((rows, cols), dtype=np.int32)
    
    # Calculate inflow count
    for r in range(rows):
        for c in range(cols):
            d = flow_dir[r, c]
            if d > 0:
                for i in range(8):
                    if d == codes[i]:
                        nr, nc = r + drs[i], c + dcs[i]
                        if 0 <= nr < rows and 0 <= nc < cols:
                            inflow_count[nr, nc] += 1
                        break
                        
    # Initialize queue with cells having no inflow
    # Use pre-allocated array for queue
    queue = np.zeros(n_cells, dtype=np.int32)
    head = 0
    tail = 0
    
    for r in range(rows):
        for c in range(cols):
            if inflow_count[r, c] == 0:
                queue[tail] = r * cols + c
                tail += 1
                
    # Process queue
    while head < tail:
        curr_idx = queue[head]
        head += 1
        
        r = curr_idx // cols
        c = curr_idx % cols
        
        d = flow_dir[r, c]
        if d > 0:
            for i in range(8):
                if d == codes[i]:
                    nr, nc = r + drs[i], c + dcs[i]
                    if 0 <= nr < rows and 0 <= nc < cols:
                        acc[nr, nc] += acc[r, c]
                        inflow_count[nr, nc] -= 1
                        if inflow_count[nr, nc] == 0:
                            queue[tail] = nr * cols + nc
                            tail += 1
                    break
                    
    return acc

class FlowRouter:
    """Native flow routing algorithms."""
    
    def __init__(self, dem_array, cellsize_x, cellsize_y=None, nodata=-9999, geotransform=None):
        """Initialize flow router.
        
        Args:
            dem_array (np.ndarray): DEM array
            cellsize_x (float): Cell size X
            cellsize_y (float): Cell size Y (optional, defaults to X)
            nodata (float): NoData value
            geotransform (tuple): GDAL geotransform (optional)
        """
        self.dem = dem_array.copy()
        self.cellsize_x = cellsize_x
        self.cellsize_y = cellsize_y if cellsize_y else cellsize_x
        self.nodata = nodata
        self.geotransform = geotransform
        self.rows, self.cols = dem_array.shape
        
        # Replace nodata with NaN
        self.dem[self.dem == nodata] = np.nan
        
        # Direction constants for Python methods
        self.drs = np.array([0, 1, 1, 1, 0, -1, -1, -1])
        self.dcs = np.array([1, 1, 0, -1, -1, -1, 0, 1])
        self.codes = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    
    def fill_depressions(self, epsilon=0.001, resolve_flats=True):
        """Fill depressions using priority-flood algorithm (Wang & Liu 2006).
        
        This is the standard hydrological depression filling algorithm.
        Processes from boundary inward, ensuring correct results matching ArcGIS.
        
        Args:
            epsilon: Small elevation increment to ensure drainage
            resolve_flats: If True, apply flat area resolution after filling
                          to ensure proper flow routing through flat areas
        
        Uses Python heapq which is a C-based heap implementation.
        """
        filled = _fill_depressions_heapq(self.dem, epsilon)
        
        if resolve_flats:
            # Apply flat area resolution to create proper flow paths
            filled = _resolve_flats_toward_outlets(filled, epsilon=epsilon/1000)
        
        return filled

    def fill_single_cell_pits(self):
        """Fill single-cell pits (raise pit to lowest neighbor)."""
        return _fill_single_cell_pits_numba(self.dem)
        
    def breach_single_cell_pits(self):
        """Breach single-cell pits (lower lowest neighbor to pit level)."""
        return _breach_single_cell_pits_numba(self.dem)
    
    def d8_flow_direction(self):
        """Calculate D8 flow direction."""
        return _d8_flow_direction_numba(
            self.dem, self.nodata, self.cellsize_x, self.cellsize_y
        )
    
    def extract_streams(self, flow_acc, threshold):
        """Extract streams from flow accumulation raster.
        
        Creates a binary stream raster using the expression:
        flow_acc >= threshold
        
        This is equivalent to ArcGIS Raster Calculator:
        "flow_acc" >= threshold
        
        Args:
            flow_acc (np.ndarray): Flow accumulation raster
            threshold (float): Accumulation threshold value
            
        Returns:
            np.ndarray: Binary stream raster (0 = non-stream, 1 = stream)
        """
        # Create binary output: 1 where flow_acc >= threshold, 0 otherwise
        streams = np.zeros(flow_acc.shape, dtype=np.int32)
        
        # Apply threshold condition
        valid_mask = ~np.isnan(flow_acc)
        streams[valid_mask & (flow_acc >= threshold)] = 1
        
        return streams
    
    def dinf_flow_direction(self):
        """Calculate D-Infinity flow direction (angle)."""
        return _dinf_flow_direction_numba(
            self.dem, self.nodata, self.cellsize_x, self.cellsize_y
        )

    def fd8_flow_accumulation(self, weights=None):
        """Calculate FD8 flow accumulation."""
        if weights is None:
            weights = np.ones_like(self.dem)
            
        return _fd8_flow_accumulation_numba(
            self.dem, weights, self.cellsize_x
        )


    def rho8_flow_direction(self):
        """Calculate Rho8 flow direction (stochastic)."""
        return _rho8_flow_direction_numba(
            self.dem, self.nodata, self.cellsize_x, self.cellsize_y
        )
    def d8_flow_accumulation(self, flow_dir, weights=None):
        """Calculate D8 flow accumulation."""
        if weights is None:
            weights = np.ones_like(self.dem)
            
        return _d8_flow_accumulation_numba(
            flow_dir.astype(np.int32), 
            weights, 
            self.codes, 
            self.drs, 
            self.dcs
        )

    def dinf_flow_accumulation(self, flow_dir, weights=None):
        """Calculate D-Infinity flow accumulation."""
        if weights is None:
            weights = np.ones_like(self.dem)
        return _dinf_flow_accumulation_numba(
            self.dem, flow_dir, weights, self.cellsize_x
        )

    def fd8_flow_accumulation(self, weights=None):
        """Calculate FD8 flow accumulation."""
        if weights is None:
            weights = np.ones_like(self.dem)
        return _fd8_flow_accumulation_numba(
            self.dem, weights, self.cellsize_x
        )
    
    def extract_streams(self, flow_acc, threshold):
        """Extract streams based on flow accumulation threshold."""
        return (flow_acc > threshold).astype(np.int8)

    def calculate_flow_distance(self, flow_dir, distance_type='outlet'):
        """Calculate flow distance.
        
        Args:
            distance_type (str): 'outlet' (downstream) or 'upstream' (max path)
            
        Returns:
            np.ndarray: Distance array
        """
        if distance_type == 'outlet':
            return _flow_distance_to_outlet_numba(
                flow_dir.astype(np.int32),
                self.codes, self.drs, self.dcs,
                self.cellsize_x, self.cellsize_y
            )
        else:
            return _flow_distance_upstream_numba(
                flow_dir.astype(np.int32),
                self.codes, self.drs, self.dcs,
                self.cellsize_x, self.cellsize_y
            )

    def flow_distance_to_outlet(self, flow_dir):
        """Calculate distance to outlet (downstream)."""
        return self.calculate_flow_distance(flow_dir, distance_type='outlet')

    def flow_distance_upstream(self, flow_dir):
        """Calculate maximum flow distance to ridge (upstream)."""
        return self.calculate_flow_distance(flow_dir, distance_type='upstream')

    def calculate_downslope_distance_to_stream(self, flow_dir, streams):
        """Calculate downslope distance to nearest stream cell."""
        return _downslope_distance_to_stream_numba(
            flow_dir.astype(np.int32),
            streams.astype(np.int8),
            self.codes, self.drs, self.dcs,
            self.cellsize_x, self.cellsize_y
        )

    def calculate_elevation_above_stream(self, flow_dir, streams):
        """Calculate Height Above Nearest Drainage (HAND)."""
        return _elevation_above_stream_numba(
            flow_dir.astype(np.int32),
            streams.astype(np.int8),
            self.dem.astype(np.float32),
            self.codes, self.drs, self.dcs
        )

    def calculate_flow_path_statistics(self, stat_type, flow_dir, streams=None):
        """Calculate various flow path statistics.
        
        Args:
            stat_type (str): Statistic type
            flow_dir (np.ndarray): D8 Flow Direction
            streams (np.ndarray): Stream raster (optional)
            
        Returns:
            np.ndarray: Result array
        """
        rows, cols = self.dem.shape
        
        if stat_type == 'downslope_distance':
            if streams is None:
                raise ValueError("Streams required for downslope distance")
            return self.calculate_downslope_distance_to_stream(flow_dir, streams)
            
        elif stat_type == 'hand':
            if streams is None:
                raise ValueError("Streams required for HAND")
            return self.calculate_elevation_above_stream(flow_dir, streams)
            
        elif stat_type == 'max_upslope_length':
            return self.calculate_flow_distance(flow_dir, 'upstream')
            
        elif stat_type == 'downslope_length':
            return self.calculate_flow_distance(flow_dir, 'outlet')
            
        elif stat_type == 'flow_length_diff':
            upslope = self.calculate_flow_distance(flow_dir, 'upstream')
            downslope = self.calculate_flow_distance(flow_dir, 'outlet')
            return upslope - downslope
            
        elif stat_type == 'longest_flowpath':
            upslope = self.calculate_flow_distance(flow_dir, 'upstream')
            downslope = self.calculate_flow_distance(flow_dir, 'outlet')
            return upslope + downslope
            
        elif stat_type == 'num_inflowing':
            # Calculate inflow counts
            inflow = np.zeros((rows, cols), dtype=np.int32)
            # Use Numba helper or simple loop
            # We can reuse _d8_flow_accumulation_numba logic but just count 1s
            # Or just implement a quick counter here
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    d = flow_dir[r, c]
                    if d > 0:
                        for i in range(8):
                            if d == self.codes[i]:
                                nr, nc = r + self.drs[i], c + self.dcs[i]
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    inflow[nr, nc] += 1
                                break
            return inflow
            
        elif stat_type == 'num_upslope':
            # Flow Accumulation - 1
            acc = self.d8_flow_accumulation(flow_dir)
            return acc - 1
            
        elif stat_type == 'num_downslope':
            # For D8, it's 1 if flow_dir > 0, else 0
            return (flow_dir > 0).astype(np.int32)
            
        elif stat_type == 'avg_flowpath_slope':
            # (Elev - OutletElev) / DownslopeLength
            downslope_len = self.calculate_flow_distance(flow_dir, 'outlet')
            outlet_elev = _get_outlet_value_numba(
                flow_dir.astype(np.int32),
                self.dem.astype(np.float32),
                self.codes, self.drs, self.dcs
            )
            
            # Avoid divide by zero
            mask = (downslope_len > 0)
            result = np.zeros_like(self.dem)
            result[mask] = (self.dem[mask] - outlet_elev[mask]) / downslope_len[mask]
            return result
            
        elif stat_type == 'max_downslope_elev_change':
            # Total Drop = Elev - OutletElev
            outlet_elev = _get_outlet_value_numba(
                flow_dir.astype(np.int32),
                self.dem.astype(np.float32),
                self.codes, self.drs, self.dcs
            )
            return self.dem - outlet_elev
            
        elif stat_type == 'min_downslope_elev_change':
            # Placeholder: Maybe local min drop? 
            # For now return 0
            return np.zeros_like(self.dem)
            
        else:
            return np.zeros_like(self.dem)

    def strahler_order(self, flow_dir, streams):
        """Calculate Strahler Stream Order."""
        return _strahler_order_numba(
            flow_dir.astype(np.int32),
            streams.astype(np.int8),
            self.codes, self.drs, self.dcs
        )
        
    def shreve_order(self, flow_dir, streams):
        """Calculate Shreve Stream Magnitude."""
        return _shreve_order_numba(
            flow_dir.astype(np.int32),
            streams.astype(np.int8),
            self.codes, self.drs, self.dcs
        )
        
    def horton_order(self, flow_dir, streams):
        """Calculate Horton Stream Order."""
        # Requires Strahler and Flow Accumulation
        strahler = self.strahler_order(flow_dir, streams)
        acc = self.d8_flow_accumulation(flow_dir)
        
        return _horton_order_numba(
            flow_dir.astype(np.int32),
            streams.astype(np.int8),
            strahler.astype(np.int32),
            acc.astype(np.float32),
            self.codes, self.drs, self.dcs
        )
        
    def hack_order(self, flow_dir, streams):
        """Calculate Hack (Gravelius) Stream Order."""
        # Requires Flow Accumulation
        acc = self.d8_flow_accumulation(flow_dir)
        
        return _hack_order_numba(
            flow_dir.astype(np.int32),
            streams.astype(np.int8),
            acc.astype(np.float32),
            self.codes, self.drs, self.dcs
        )

    def calculate_hydrological_slope(self, flow_dir):
        """Calculate hydrological slope (drop to downstream neighbor)."""
        return _calculate_hydrological_slope_numba(
            self.dem.astype(np.float32),
            flow_dir.astype(np.int32),
            self.codes, self.drs, self.dcs,
            self.cellsize_x, self.cellsize_y
        )

    def calculate_stream_link_statistics(self, stat_type, flow_dir, streams):
        """Calculate statistics for stream links.
        
        Args:
            stat_type (str): 'id', 'length', 'slope', 'class', 'slope_continuous'
            flow_dir (np.ndarray): Flow direction
            streams (np.ndarray): Stream raster
            
        Returns:
            np.ndarray: Result raster
        """
        # First assign link IDs
        links = self.assign_stream_link_ids(flow_dir, streams)
        
        if stat_type == 'id':
            return links
            
        rows, cols = self.dem.shape
        result = np.zeros((rows, cols), dtype=np.float32)
        
        # Get unique links
        unique_links = np.unique(links)
        unique_links = unique_links[unique_links > 0]
        
        if stat_type == 'length':
            # Calculate length for each link
            # Simplified: Count cells * cell_size (approx)
            # Better: Sum flow lengths
            for link_id in unique_links:
                mask = (links == link_id)
                count = np.sum(mask)
                # Approx length
                length = count * self.cellsize_x
                result[mask] = length
                
        elif stat_type == 'slope':
            # Drop / Length for each link
            for link_id in unique_links:
                mask = (links == link_id)
                if not np.any(mask):
                    continue
                    
                elevs = self.dem[mask]
                max_elev = np.nanmax(elevs)
                min_elev = np.nanmin(elevs)
                drop = max_elev - min_elev
                
                count = np.sum(mask)
                length = count * self.cellsize_x
                
                if length > 0:
                    slope = drop / length
                    result[mask] = slope
                    
        elif stat_type == 'class':
            # Dummy classification based on ID?
            # Or maybe Strahler order?
            # Let's return IDs for now or 1
            result = links.astype(np.float32)
            
        elif stat_type == 'slope_continuous':
            # Slope at each cell
            # We can use the DEM slope masked by streams
            # Hydrological slope: Drop to downstream neighbor
            slope = self.calculate_hydrological_slope(flow_dir)
            # Mask by streams
            result = np.where(streams > 0, slope, 0)
            return result
            
        return result

    def assign_stream_link_ids(self, flow_dir, streams):
        """Assign unique IDs to stream links."""
        # This requires a Numba helper or graph traversal
        # Placeholder for now: Label connected components
        from scipy.ndimage import label, generate_binary_structure
        
        # Stream mask
        mask = (streams > 0)
        
        # Label connected components (8-connectivity)
        s = generate_binary_structure(2, 2)
        labeled, num_features = label(mask, structure=s)
        
        # But stream links should be broken at junctions!
        # Connected components merges junctions.
        # We need to break at junctions (cells with >1 inflow from streams).
        
        # Calculate inflow from streams only
        # ...
        
        # For now, return connected components as a proxy
        return labeled.astype(np.int32)

    def extract_stream_segments(self, streams):
        """Extract stream segments as list of polylines.
        
        Args:
            streams (np.ndarray): Stream raster (stream order values or binary)
            
        Returns:
            list: List of (segment_points, stream_order) tuples
                  where segment_points is list of (x, y) tuples
        """
        if not hasattr(self, 'flow_dir') or self.flow_dir is None:
             if not np.all(np.isnan(self.dem)):
                 self.flow_dir = self.d8_flow_direction()
             else:
                 raise ValueError("Flow direction required for stream extraction")
        
        rows, cols = streams.shape
        flow_dir = self.flow_dir
        
        # Find all stream cells and their order values
        stream_cells = []
        for r in range(rows):
            for c in range(cols):
                if streams[r, c] > 0 and not np.isnan(streams[r, c]):
                    stream_cells.append((r, c, int(streams[r, c])))
        
        if not stream_cells:
            return []
        
        # Build downstream map for stream cells only
        stream_set = set((r, c) for r, c, _ in stream_cells)
        downstream_map = {}
        has_upstream = set()
        
        for r, c, order in stream_cells:
            d = flow_dir[r, c]
            if d > 0:
                for i in range(8):
                    if d == self.codes[i]:
                        nr, nc = r + self.drs[i], c + self.dcs[i]
                        if (nr, nc) in stream_set:
                            downstream_map[(r, c)] = (nr, nc)
                            has_upstream.add((nr, nc))
                        break
        
        # Find source cells (stream cells with no upstream)
        source_cells = [(r, c, order) for r, c, order in stream_cells if (r, c) not in has_upstream]
        
        use_geo = hasattr(self, 'geotransform') and self.geotransform is not None
        segments = []
        visited = set()
        
        # Trace from each source cell
        for start_r, start_c, start_order in source_cells:
            if (start_r, start_c) in visited:
                continue
            
            segment = []
            curr = (start_r, start_c)
            curr_order = start_order
            
            while curr is not None and curr not in visited:
                visited.add(curr)
                r, c = curr
                
                # Get coordinates
                if use_geo:
                    x = self.geotransform[0] + (c + 0.5) * self.geotransform[1]
                    y = self.geotransform[3] + (r + 0.5) * self.geotransform[5]
                else:
                    x, y = float(c), float(r)
                
                segment.append((x, y))
                
                # Move downstream
                next_cell = downstream_map.get(curr)
                
                if next_cell is not None:
                    # Check if stream order changes (junction)
                    nr, nc = next_cell
                    next_order = int(streams[nr, nc])
                    
                    if next_order != curr_order and len(segment) >= 2:
                        # Save current segment and start new one
                        segments.append((segment, curr_order))
                        segment = [(x, y)]  # Start new segment from current point
                        curr_order = next_order
                    
                    curr = next_cell
                else:
                    curr = None
            
            # Save final segment
            if len(segment) >= 2:
                segments.append((segment, curr_order))
        
        return segments

@jit(nopython=True, cache=True)
def _flow_distance_upstream_numba(flow_dir, codes, drs, dcs, cellsize_x, cellsize_y):
    """Calculate maximum upstream flow distance (Distance to Ridge)."""
    rows, cols = flow_dir.shape
    dist = np.zeros((rows, cols), dtype=np.float32)
    
    # Inflow count
    inflow_count = np.zeros((rows, cols), dtype=np.int32)
    
    for r in range(rows):
        for c in range(cols):
            d = flow_dir[r, c]
            if d > 0:
                for i in range(8):
                    if d == codes[i]:
                        nr, nc = r + drs[i], c + dcs[i]
                        if 0 <= nr < rows and 0 <= nc < cols:
                            inflow_count[nr, nc] += 1
                        break
    
    # Queue for ridge cells (inflow=0)
    queue_r = []
    queue_c = []
    
    for r in range(rows):
        for c in range(cols):
            if inflow_count[r, c] == 0:
                queue_r.append(r)
                queue_c.append(c)
                dist[r, c] = 0.0 # Distance at ridge is 0
                
    # Process queue (Downstream propagation of max distance)
    # Wait, we want distance FROM ridge.
    # So if A flows to B, dist[B] = max(dist[B], dist[A] + step)
    
    diag_dist = np.sqrt(cellsize_x**2 + cellsize_y**2)
    
    head = 0
    while head < len(queue_r):
        r = queue_r[head]
        c = queue_c[head]
        head += 1
        
        d = flow_dir[r, c]
        if d > 0:
            for i in range(8):
                if d == codes[i]:
                    nr, nc = r + drs[i], c + dcs[i]
                    if 0 <= nr < rows and 0 <= nc < cols:
                        # Calculate step
                        step = diag_dist if (drs[i] != 0 and dcs[i] != 0) else cellsize_x
                        
                        # Update max distance
                        new_dist = dist[r, c] + step
                        if new_dist > dist[nr, nc]:
                            dist[nr, nc] = new_dist
                            
                        inflow_count[nr, nc] -= 1
                        if inflow_count[nr, nc] == 0:
                            queue_r.append(nr)
                            queue_c.append(nc)
                    break
                    
    return dist

@jit(nopython=True, cache=True)
def _fd8_flow_accumulation_numba(dem, weights, cellsize):
    """Calculate FD8 flow accumulation (Freeman 1991)."""
    rows, cols = dem.shape
    acc = weights.astype(np.float64)
    
    # Flatten and sort indices (high to low)
    flat_indices = np.argsort(dem.ravel())[::-1]
    
    drs = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dcs = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    # Distances
    diag_dist = np.sqrt(2.0) * cellsize
    dists = np.array([cellsize, diag_dist, cellsize, diag_dist, 
                      cellsize, diag_dist, cellsize, diag_dist])
    
    for idx in flat_indices:
        r = idx // cols
        c = idx % cols
        
        if np.isnan(dem[r, c]):
            continue
            
        # Calculate slopes to neighbors
        slopes = np.zeros(8)
        total_slope = 0.0
        
        for i in range(8):
            nr, nc = r + drs[i], c + dcs[i]
            if 0 <= nr < rows and 0 <= nc < cols and not np.isnan(dem[nr, nc]):
                drop = dem[r, c] - dem[nr, nc]
                if drop > 0:
                    slope = drop / dists[i]
                    # FD8 uses slope^1.1
                    slope = slope ** 1.1
                    slopes[i] = slope
                    total_slope += slope
        
        # Distribute flow
        if total_slope > 0:
            for i in range(8):
                if slopes[i] > 0:
                    nr, nc = r + drs[i], c + dcs[i]
                    fraction = slopes[i] / total_slope
                    acc[nr, nc] += acc[r, c] * fraction
                    
    return acc

@jit(nopython=True, cache=True)
def _flow_distance_to_outlet_numba(flow_dir, codes, drs, dcs, cellsize_x, cellsize_y):
    """Calculate distance to outlet (downstream) using Inverted Graph BFS."""
    rows, cols = flow_dir.shape
    n_cells = rows * cols
    dist = np.full((rows, cols), -1.0, dtype=np.float32)
    
    # 1. Calculate Inflow Counts (Degree)
    inflow_count = np.zeros(n_cells, dtype=np.int32)
    
    for r in range(rows):
        for c in range(cols):
            d = flow_dir[r, c]
            if d > 0:
                for i in range(8):
                    if d == codes[i]:
                        nr, nc = r + drs[i], c + dcs[i]
                        if 0 <= nr < rows and 0 <= nc < cols:
                            # r,c flows into nr,nc
                            # increment inflow count of nr,nc
                            idx_dest = nr * cols + nc
                            inflow_count[idx_dest] += 1
                        break
                        
    # 2. Build CSR (Compressed Sparse Row) for Inverted Graph
    # offsets: where neighbors for each cell start
    offsets = np.zeros(n_cells + 1, dtype=np.int32)
    
    # Prefix sum to get offsets
    current_offset = 0
    for i in range(n_cells):
        offsets[i] = current_offset
        current_offset += inflow_count[i]
    offsets[n_cells] = current_offset
    
    # sources: flattened indices of upstream cells
    sources = np.zeros(n_cells, dtype=np.int32)
    
    # Temporary counter to fill sources
    # We can reuse inflow_count or create a new one. Let's create new to be safe.
    current_pos = offsets[:-1].copy()
    
    for r in range(rows):
        for c in range(cols):
            d = flow_dir[r, c]
            if d > 0:
                for i in range(8):
                    if d == codes[i]:
                        nr, nc = r + drs[i], c + dcs[i]
                        if 0 <= nr < rows and 0 <= nc < cols:
                            # r,c flows into nr,nc
                            # Add r,c as source for nr,nc
                            dest_idx = nr * cols + nc
                            pos = current_pos[dest_idx]
                            sources[pos] = r * cols + c
                            current_pos[dest_idx] += 1
                        break

    # 3. Initialize BFS with Outlets
    # Outlets are cells that flow to nodata/edge OR have flow_dir <= 0
    # Actually, distance is 0 for outlets.
    
    queue = np.zeros(n_cells, dtype=np.int32)
    head = 0
    tail = 0
    
    diag_dist = np.sqrt(cellsize_x**2 + cellsize_y**2)
    
    for r in range(rows):
        for c in range(cols):
            d = flow_dir[r, c]
            is_outlet = False
            if d <= 0:
                is_outlet = True
            else:
                # Check if flows off map
                flows_off = True
                for i in range(8):
                    if d == codes[i]:
                        nr, nc = r + drs[i], c + dcs[i]
                        if 0 <= nr < rows and 0 <= nc < cols:
                            flows_off = False
                        break
                if flows_off:
                    is_outlet = True
            
            if is_outlet:
                dist[r, c] = 0.0
                queue[tail] = r * cols + c
                tail += 1
                
    # 4. BFS Upstream
    while head < tail:
        curr_idx = queue[head]
        head += 1
        
        curr_r = curr_idx // cols
        curr_c = curr_idx % cols
        curr_dist = dist[curr_r, curr_c]
        
        # Iterate over upstream neighbors
        start = offsets[curr_idx]
        end = offsets[curr_idx + 1]
        
        for k in range(start, end):
            up_idx = sources[k]
            up_r = up_idx // cols
            up_c = up_idx % cols
            
            if dist[up_r, up_c] == -1.0: # Not visited
                # Calculate distance from up to curr
                # We need direction from up to curr
                d = flow_dir[up_r, up_c]
                step = cellsize_x # Default
                
                # Find step distance
                # We know up flows to curr, so we just need to check if diagonal
                # Optimization: We can store step in CSR? No, too much memory.
                # Just re-check direction.
                for i in range(8):
                    if d == codes[i]:
                        if drs[i] != 0 and dcs[i] != 0:
                            step = diag_dist
                        else:
                            step = cellsize_x
                        break
                
                dist[up_r, up_c] = curr_dist + step
                queue[tail] = up_idx
                tail += 1
                
    return dist

    def delineate_basins(self, flow_dir):
        """Delineate all drainage basins.
        
        Returns:
            np.ndarray: Basin ID raster
        """
        return _delineate_basins_numba(
            flow_dir.astype(np.int32),
            self.codes, self.drs, self.dcs
        )

    def delineate_watersheds(self, seeds):
        """Delineate watersheds from seed raster.
        
        Args:
            seeds (np.ndarray): Raster with seed IDs (0 for non-seeds)
            
        Returns:
            np.ndarray: Watershed ID raster
        """
        if not hasattr(self, 'flow_dir') or self.flow_dir is None:
             if not np.all(np.isnan(self.dem)):
                 self.flow_dir = self.d8_flow_direction()
             else:
                 raise ValueError("Flow direction required for watershed delineation")
                 
        return _delineate_watersheds_numba(
            self.flow_dir.astype(np.int32),
            seeds.astype(np.int32),
            self.codes, self.drs, self.dcs
        )

    def find_no_flow_cells(self):
        """Identify cells with no flow direction.
        
        Returns:
            np.ndarray: Binary mask (1=No Flow)
        """
        if not hasattr(self, 'flow_dir') or self.flow_dir is None:
             raise ValueError("Flow direction required")
             
        # D8 codes are > 0. 0 is usually sink/undefined.
        return (self.flow_dir == 0).astype(np.int8)

    def find_parallel_flow(self):
        """Identify parallel flow patterns.
        
        Returns:
            np.ndarray: Binary mask (1=Parallel Flow)
        """
        if not hasattr(self, 'flow_dir') or self.flow_dir is None:
             raise ValueError("Flow direction required")
             
        return _find_parallel_flow_numba(
            self.flow_dir.astype(np.int32),
            self.codes, self.drs, self.dcs
        )

    def assign_stream_link_ids(self, flow_dir, streams):
        """Assign unique IDs to stream links.
        
        Args:
            flow_dir (np.ndarray): Flow direction raster
            streams (np.ndarray): Stream raster
            
        Returns:
            np.ndarray: Stream link ID raster
        """
        return _assign_stream_link_ids_numba(
            flow_dir.astype(np.int32),
            streams.astype(np.int32),
            self.codes, self.drs, self.dcs
        )

    def assign_tributary_ids(self, flow_dir, streams):
        """Assign unique IDs to tributaries.
        
        Returns:
            np.ndarray: Tributary ID raster
        """
        return self.assign_stream_link_ids(flow_dir, streams)

    def join_stream_gaps(self, streams, gap_threshold):
        """Join gaps in stream network (raster).
        
        Args:
            streams (np.ndarray): Binary stream raster
            gap_threshold (float): Max gap distance in map units
            
        Returns:
            np.ndarray: Filled stream raster
        """
        if not hasattr(self, 'flow_dir') or self.flow_dir is None:
             raise ValueError("Flow direction required for gap joining")
             
        return _join_stream_gaps_numba(
            streams.astype(np.int8),
            self.flow_dir.astype(np.int32),
            self.codes, self.drs, self.dcs,
            self.cellsize_x, self.cellsize_y,
            gap_threshold
        )


    def snap_pour_points(self, points, flow_acc, snap_dist):
        """Snap pour points to high accumulation cells.
        
        Args:
            points (list): List of (x, y) tuples
            flow_acc (np.ndarray): Flow accumulation raster
            snap_dist (float): Snap distance in map units
            
        Returns:
            list: List of snapped (x, y) tuples
        """
        snapped_points = []
        
        # Convert snap distance to pixels
        radius_px = int(snap_dist / self.cellsize_x)
        if radius_px < 1:
            radius_px = 1
            
        rows, cols = flow_acc.shape
        
        for x, y in points:
            # Convert to row, col
            c = int((x - self.geotransform[0]) / self.geotransform[1]) if hasattr(self, 'geotransform') else int(x / self.cellsize_x) # Fallback if geotransform missing
            r = int((y - self.geotransform[3]) / self.geotransform[5]) if hasattr(self, 'geotransform') else int(y / self.cellsize_y)
            
            # Use simple logic if geotransform is missing or just assume it's there
            # The original code used self.geotransform, so let's assume it's there or handle it.
            # But FlowRouter doesn't usually have geotransform.
            # Let's use the original logic but wrapped in try/except or check.
            
            if not hasattr(self, 'geotransform'):
                 # Cannot snap without geotransform to convert coords
                 snapped_points.append((x, y))
                 continue
                 
            c = int((x - self.geotransform[0]) / self.geotransform[1])
            r = int((y - self.geotransform[3]) / self.geotransform[5])
            
            if 0 <= r < rows and 0 <= c < cols:
                # Search window
                r_min = max(0, r - radius_px)
                r_max = min(rows, r + radius_px + 1)
                c_min = max(0, c - radius_px)
                c_max = min(cols, c + radius_px + 1)
                
                window = flow_acc[r_min:r_max, c_min:c_max]
                
                # Handle NaNs
                if np.all(np.isnan(window)):
                    snapped_points.append((x, y))
                    continue
                    
                # Get local max index
                try:
                    max_idx = np.nanargmax(window)
                    local_r, local_c = np.unravel_index(max_idx, window.shape)
                    
                    best_r = r_min + local_r
                    best_c = c_min + local_c
                    
                    new_x = self.geotransform[0] + (best_c + 0.5) * self.geotransform[1]
                    new_y = self.geotransform[3] + (best_r + 0.5) * self.geotransform[5]
                    
                    snapped_points.append((new_x, new_y))
                except:
                    snapped_points.append((x, y))
            else:
                snapped_points.append((x, y))
                
        return snapped_points

    def remove_short_streams(self, flow_dir, streams, min_length):
        """Remove stream links shorter than a threshold.
        
        Args:
            flow_dir (np.ndarray): Flow direction raster
            streams (np.ndarray): Stream raster (1/0 or IDs)
            min_length (float): Minimum length in map units
            
        Returns:
            np.ndarray: Cleaned stream raster
        """
        # 1. Assign Link IDs
        link_ids = self.assign_stream_link_ids(flow_dir, streams)
        
        # 2. Calculate length of each link
        lengths = _calculate_link_lengths_numba(
            link_ids.astype(np.int32),
            flow_dir.astype(np.int32),
            self.codes, self.drs, self.dcs,
            self.cellsize_x, self.cellsize_y
        )
        
        # 3. Filter
        return _filter_short_streams_numba(
            link_ids.astype(np.int32),
            lengths,
            min_length
        )

    def extract_stream_segments(self, streams):
        """Extract stream segments as list of point lists.
        
        Args:
            streams (np.ndarray): Stream raster
            
        Returns:
            list: List of segments, where each segment is a list of (x, y) tuples
        """
        if not hasattr(self, 'flow_dir') or self.flow_dir is None:
             if not np.all(np.isnan(self.dem)):
                 self.flow_dir = self.d8_flow_direction()
             else:
                 raise ValueError("Flow direction required for stream extraction")
                 
        link_ids = self.assign_stream_link_ids(self.flow_dir, streams)
        
        segments = []
        unique_links = np.unique(link_ids)
        unique_links = unique_links[unique_links > 0]
        
        use_geo = hasattr(self, 'geotransform') and self.geotransform is not None
        
        for link_id in unique_links:
            mask = (link_ids == link_id)
            cells = np.argwhere(mask)
            
            if len(cells) == 0:
                continue
            
            # Build mini-graph
            link_cells_set = set((r, c) for r, c in cells)
            downstream_map = {}
            upstream_count = {tuple(c): 0 for c in cells}
            
            for r, c in cells:
                d = self.flow_dir[r, c]
                if d > 0:
                    for i in range(8):
                        if d == self.codes[i]:
                            nr, nc = r + self.drs[i], c + self.dcs[i]
                            if (nr, nc) in link_cells_set:
                                downstream_map[(r, c)] = (nr, nc)
                                upstream_count[(nr, nc)] += 1
                            break
            
            # Find start node
            start_nodes = [n for n, count in upstream_count.items() if count == 0]
            
            if not start_nodes:
                curr = tuple(cells[0])
            else:
                curr = start_nodes[0]
                
            # Trace
            segment = []
            while True:
                if use_geo:
                    x = self.geotransform[0] + (curr[1] + 0.5) * self.geotransform[1]
                    y = self.geotransform[3] + (curr[0] + 0.5) * self.geotransform[5]
                else:
                    x, y = float(curr[1]), float(curr[0])
                    
                segment.append((x, y))
                
                if curr in downstream_map:
                    curr = downstream_map[curr]
                else:
                    break
            
            segments.append(segment)
            
        return segments

@jit(nopython=True, cache=True)
def _delineate_basins_numba(flow_dir, codes, drs, dcs):
    """Delineate basins for all outlets."""
    rows, cols = flow_dir.shape
    basins = np.zeros((rows, cols), dtype=np.int32)
    
    # 1. Identify outlets (cells that flow off map or to nodata)
    # And assign unique IDs
    current_id = 1
    
    # Queue for upstream tracing
    queue_r = []
    queue_c = []
    # Find outlets
    for r in range(rows):
        for c in range(cols):
            if flow_dir[r, c] == 0:  # No flow direction = sink or flat or edge
                # Check if it's a valid outlet (e.g. edge of map or sink)
                # For now, treat all 0 flow dir as outlets if they are not nodata
                # But wait, flow_dir 0 might just be undefined.
                # Let's assume outlets have flow_dir 0 or flow off map.
                # Actually, standard D8: if flow_dir points off map, it's an outlet.
                # Here we assume 0 means undefined/sink.
                
                # Assign ID and add to queue
                basins[r, c] = current_id
                queue_r.append(r)
                queue_c.append(c)
                current_id += 1
                
    # Process queue (upstream tracing)
    # We need to find cells that flow INTO the current cell
    # This is inefficient without an inverted flow direction or checking neighbors.
    # A better way for Numba:
    # Iterate until no changes (slow) or build an adjacency list (hard in Numba).
    # OR: Just scan the whole grid? No.
    
    # Alternative: Use the recursive approach (stack based) or iterative with stack.
    # Since we don't have an inverted index, we have to scan neighbors.
    
    # Actually, let's use a simpler approach for now:
    # Use the existing queue. For each cell in queue, check all 8 neighbors.
    # If a neighbor flows INTO this cell, assign it the same basin ID and add to queue.
    
    head = 0
    while head < len(queue_r):
        r = queue_r[head]
        c = queue_c[head]
        head += 1
        
        bid = basins[r, c]
        
        # Check all 8 neighbors
        for i in range(8):
            nr, nc = r + drs[i], c + dcs[i]
            
            if 0 <= nr < rows and 0 <= nc < cols:
                if basins[nr, nc] == 0:  # Not yet assigned
                    # Check if neighbor flows into current cell (r, c)
                    # Neighbor flow direction
                    nd = flow_dir[nr, nc]
                    
                    # To flow into (r, c), neighbor (nr, nc) must point to (r, c)
                    # We need to check if (nr + d_r) == r and (nc + d_c) == c
                    # But we have codes.
                    
                    if nd > 0:
                        for k in range(8):
                            if nd == codes[k]:
                                tr, tc = nr + drs[k], nc + dcs[k]
                                if tr == r and tc == c:
                                    # Flows into current cell
                                    basins[nr, nc] = bid
                                    queue_r.append(nr)
                                    queue_c.append(nc)
                                break
                                
    return basins

@jit(nopython=True, cache=True)
def _delineate_watersheds_numba(flow_dir, seeds, codes, drs, dcs):
    """Delineate watersheds from seeds."""
    rows, cols = flow_dir.shape
    watersheds = seeds.copy()
    
    queue_r = []
    queue_c = []
    
    # Initialize queue with seeds
    for r in range(rows):
        for c in range(cols):
            if watersheds[r, c] > 0:
                queue_r.append(r)
                queue_c.append(c)
                
    # Process queue (upstream tracing)
    head = 0
    while head < len(queue_r):
        r = queue_r[head]
        c = queue_c[head]
        head += 1
        
        wid = watersheds[r, c]
        
        # Check all 8 neighbors
        for i in range(8):
            nr, nc = r + drs[i], c + dcs[i]
            
            if 0 <= nr < rows and 0 <= nc < cols:
                if watersheds[nr, nc] == 0:  # Not yet assigned
                    # Check if neighbor flows into current cell (r, c)
                    nd = flow_dir[nr, nc]
                    
                    if nd > 0:
                        for k in range(8):
                            if nd == codes[k]:
                                tr, tc = nr + drs[k], nc + dcs[k]
                                if tr == r and tc == c:
                                    # Flows into current cell
                                    watersheds[nr, nc] = wid
                                    queue_r.append(nr)
                                    queue_c.append(nc)
                                break
                                
    return watersheds

@jit(nopython=True, cache=True)
def _find_parallel_flow_numba(flow_dir, codes, drs, dcs):
    """Identify parallel flow patterns."""
    rows, cols = flow_dir.shape
    parallel = np.zeros((rows, cols), dtype=np.int8)
    
    for r in range(rows):
        for c in range(cols):
            d = flow_dir[r, c]
            if d > 0:
                # Check neighbors
                for i in range(8):
                    nr, nc = r + drs[i], c + dcs[i]
                    if 0 <= nr < rows and 0 <= nc < cols:
                        nd = flow_dir[nr, nc]
                        if nd == d:
                            # Neighbor flows in SAME direction
                            # Check if they are adjacent perpendicular to flow?
                            # Parallel flow usually means adjacent cells flowing in same direction.
                            parallel[r, c] = 1
                            break
    return parallel

@jit(nopython=True, cache=True)
def _assign_stream_link_ids_numba(flow_dir, streams, codes, drs, dcs):
    """Assign unique IDs to stream links (segments between junctions)."""
    rows, cols = flow_dir.shape
    links = np.zeros((rows, cols), dtype=np.int32)
    
    # 1. Identify junctions and outlets (start/end of links)
    # A junction is a stream cell with >1 upstream stream neighbors
    # An outlet is a stream cell with 0 downstream stream neighbors (or flows off map)
    # A source is a stream cell with 0 upstream stream neighbors
    
    # We can assign IDs by tracing upstream from outlets/junctions
    
    # First, calculate inflow count for stream cells
    inflow_count = np.zeros((rows, cols), dtype=np.int32)
    
    for r in range(rows):
        for c in range(cols):
            if streams[r, c] > 0:
                d = flow_dir[r, c]
                if d > 0:
                    for i in range(8):
                        if d == codes[i]:
                            nr, nc = r + drs[i], c + dcs[i]
                            if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc] > 0:
                                inflow_count[nr, nc] += 1
                            break
                            
    # 2. Assign IDs
    current_id = 1
    
    # Iterate through all stream cells
    # If a cell is a junction (inflow > 1) or outlet (downstream is not stream), it starts a new link upstream?
    # Actually, links are usually defined as segments between junctions.
    # So each source starts a link, ending at a junction.
    # Each junction starts a new link downstream?
    # Standard definition: A link is a section of stream channel between two successive junctions, 
    # or between a source and a junction, or between a junction and the outlet.
    
    # Let's trace downstream from sources and junctions.
    
    # Find all "Link Heads": Sources (inflow=0) and Junctions (inflow > 1)
    # Wait, junctions are where links END (merging). The cell AFTER a junction starts a new link.
    
    # Let's use a visited array
    visited = np.zeros((rows, cols), dtype=np.bool_)
    
    for r in range(rows):
        for c in range(cols):
            if streams[r, c] > 0 and not visited[r, c]:
                # Start a new link if:
                # 1. It's a source (inflow == 0)
                # 2. It's immediately downstream of a junction (inflow > 1)
                
                # Actually, simpler:
                # Assign ID to current cell. Trace downstream.
                # If downstream cell is a junction (inflow > 1), stop link.
                # If downstream cell has inflow == 1, continue link.
                
                # We need to find unvisited link heads.
                is_head = False
                if inflow_count[r, c] == 0:
                    is_head = True
                elif inflow_count[r, c] > 1:
                    # This is a junction. It belongs to the downstream link?
                    # Usually junctions are the END of upstream links.
                    # The single downstream flow from a junction starts a new link.
                    pass
                else:
                    # Inflow == 1. It's a middle of a link.
                    # Unless the upstream neighbor was a junction?
                    pass
                    
    # Let's try a simpler approach:
    # 1. Mark all junctions (inflow > 1)
    # 2. Remove junctions temporarily? No.
    
    # Correct approach:
    # A link ID is assigned to all cells in a segment.
    # Unique IDs for each segment.
    
    # Iterate all cells.
    # If cell is stream:
    #   If inflow != 1: It's a start of a link (Source or Junction-result)?
    #   No, if inflow > 1, it's a junction cell. The links MERGE here.
    #   So the junction cell itself is usually part of the DOWNSTREAM link.
    
    # Let's assume:
    # Sources (inflow=0) start a link.
    # Junctions (inflow>1) start a NEW link.
    
    # We need to traverse in topological order (upstream to downstream)
    # But we don't have a sorted list.
    
    # Alternative:
    # Give every stream cell a unique ID initially? No.
    
    # Let's use the "Head" approach.
    # Heads are: Sources (inflow=0) AND cells where flow_dir of upstream is a junction?
    
    # Let's iterate and find all cells that START a link.
    # A cell starts a link if:
    # 1. Inflow count == 0 (Source)
    # 2. Inflow count > 1 (Junction - starts the downstream link)
    
    # Wait, if inflow > 1, multiple links merge INTO this cell.
    # So this cell is the start of the new downstream link.
    
    stack_r = []
    stack_c = []
    
    for r in range(rows):
        for c in range(cols):
            if streams[r, c] > 0:
                if inflow_count[r, c] == 0 or inflow_count[r, c] > 1:
                    # Start of a new link
                    links[r, c] = current_id
                    stack_r.append(r)
                    stack_c.append(c)
                    current_id += 1
                    
    # Now trace downstream from these heads
    # BUT, be careful not to overwrite if we hit another head?
    # If we hit a junction (inflow > 1), that's a new head, so we stop.
    
    head_idx = 0
    while head_idx < len(stack_r):
        r = stack_r[head_idx]
        c = stack_c[head_idx]
        head_idx += 1
        
        lid = links[r, c]
        
        # Trace downstream
        d = flow_dir[r, c]
        if d > 0:
            for i in range(8):
                if d == codes[i]:
                    nr, nc = r + drs[i], c + dcs[i]
                    if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc] > 0:
                        # Check if neighbor is a head (junction)
                        if inflow_count[nr, nc] > 1:
                            # It's a junction, so it starts a NEW link.
                            # We stop tracing this link.
                            pass
                        else:
                            # It's a continuation of current link
                            if links[nr, nc] == 0:
                                links[nr, nc] = lid
                                stack_r.append(nr)
                                stack_c.append(nc)
                    break
                    
    return links



@jit(nopython=True, cache=True)
def _calculate_link_lengths_numba(link_ids, flow_dir, codes, drs, dcs, cellsize_x, cellsize_y):
    """Calculate length of each stream link."""
    rows, cols = link_ids.shape
    max_id = np.max(link_ids)
    lengths = np.zeros(max_id + 1, dtype=np.float32)
    
    diag_dist = np.sqrt(cellsize_x**2 + cellsize_y**2)
    
    for r in range(rows):
        for c in range(cols):
            lid = link_ids[r, c]
            if lid > 0:
                # Add length flowing OUT of this cell?
                # Or length of this cell?
                # Length of a cell is usually approximated by the distance to the downstream neighbor.
                
                d = flow_dir[r, c]
                step = 0.0
                if d > 0:
                    for i in range(8):
                        if d == codes[i]:
                            if drs[i] != 0 and dcs[i] != 0:
                                step = diag_dist
                            else:
                                step = cellsize_x
                            break
                else:
                    # Outlet cell, add half cellsize or just 0?
                    # Usually 0 or cellsize. Let's add cellsize.
                    step = cellsize_x
                    
                lengths[lid] += step
                
    return lengths

@jit(nopython=True, cache=True)
def _filter_short_streams_numba(link_ids, lengths, min_length):
    """Filter out short streams.
    
    Args:
        link_ids (np.ndarray): Stream link IDs
        lengths (np.ndarray): Array of lengths where index is link ID
        min_length (float): Minimum length
        
    Returns:
        np.ndarray: Cleaned stream raster (binary 0/1)
    """
    rows, cols = link_ids.shape
    cleaned = np.zeros((rows, cols), dtype=np.int8)
    
    for r in range(rows):
        for c in range(cols):
            lid = link_ids[r, c]
            if lid > 0:
                # Check length
                if lid < len(lengths):
                    if lengths[lid] >= min_length:
                        cleaned[r, c] = 1
                    
    return cleaned

@jit(nopython=True, cache=True)
def _strahler_order_numba(flow_dir, streams, codes, drs, dcs):
    """Calculate Strahler Stream Order."""
    rows, cols = flow_dir.shape
    order = np.zeros((rows, cols), dtype=np.int32)
    
    # Inflow count for stream cells only
    inflow_count = np.zeros((rows, cols), dtype=np.int32)
    
    # Initialize queue with sources
    queue_r = []
    queue_c = []
    
    for r in range(rows):
        for c in range(cols):
            if streams[r, c] > 0:
                d = flow_dir[r, c]
                if d > 0:
                    for i in range(8):
                        if d == codes[i]:
                            nr, nc = r + drs[i], c + dcs[i]
                            if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc] > 0:
                                inflow_count[nr, nc] += 1
                            break
                            
    # Find sources (inflow=0)
    for r in range(rows):
        for c in range(cols):
            if streams[r, c] > 0 and inflow_count[r, c] == 0:
                order[r, c] = 1
                queue_r.append(r)
                queue_c.append(c)
                
    # Process queue
    head = 0
    
    # We need to track max order of tributaries for each cell
    # And count of tributaries with that max order
    # storage: cell -> (max_order, count_max)
    # But we can't easily store tuples in 2D array in Numba.
    # Use two arrays.
    max_orders = np.zeros((rows, cols), dtype=np.int32)
    count_max = np.zeros((rows, cols), dtype=np.int32)
    
    while head < len(queue_r):
        r = queue_r[head]
        c = queue_c[head]
        head += 1
        
        current_ord = order[r, c]
        
        d = flow_dir[r, c]
        if d > 0:
            for i in range(8):
                if d == codes[i]:
                    nr, nc = r + drs[i], c + dcs[i]
                    if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc] > 0:
                        
                        # Update downstream stats
                        if current_ord > max_orders[nr, nc]:
                            max_orders[nr, nc] = current_ord
                            count_max[nr, nc] = 1
                        elif current_ord == max_orders[nr, nc]:
                            count_max[nr, nc] += 1
                            
                        inflow_count[nr, nc] -= 1
                        if inflow_count[nr, nc] == 0:
                            # Calculate order for downstream cell
                            if count_max[nr, nc] > 1:
                                order[nr, nc] = max_orders[nr, nc] + 1
                            else:
                                order[nr, nc] = max_orders[nr, nc]
                                
                            queue_r.append(nr)
                            queue_c.append(nc)
                    break
                    
    return order

@jit(nopython=True, cache=True)
def _horton_order_numba(flow_dir, streams, strahler, flow_acc, codes, drs, dcs):
    """Calculate Horton Stream Order."""
    rows, cols = flow_dir.shape
    horton = np.zeros((rows, cols), dtype=np.int32)
    
    # 1. Find outlets (stream cells with no downstream stream neighbor)
    outlets_r = []
    outlets_c = []
    
    for r in range(rows):
        for c in range(cols):
            if streams[r, c] > 0:
                d = flow_dir[r, c]
                is_outlet = True
                if d > 0:
                    for i in range(8):
                        if d == codes[i]:
                            nr, nc = r + drs[i], c + dcs[i]
                            if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc] > 0:
                                is_outlet = False
                            break
                if is_outlet:
                    outlets_r.append(r)
                    outlets_c.append(c)
                    
    # 2. Process from outlets upstream
    # We use a stack for recursion simulation
    # Stack items: (r, c, order_to_assign)
    stack_r = []
    stack_c = []
    stack_ord = []
    
    for i in range(len(outlets_r)):
        r, c = outlets_r[i], outlets_c[i]
        # Horton order at outlet is its Strahler order
        ord_val = strahler[r, c]
        stack_r.append(r)
        stack_c.append(c)
        stack_ord.append(ord_val)
        
    while len(stack_r) > 0:
        r = stack_r.pop()
        c = stack_c.pop()
        ord_val = stack_ord.pop()
        
        horton[r, c] = ord_val
        
        # Find upstream tributaries
        # We need to pick the "main" tributary to continue the current order
        # Main tributary = same Strahler order as current (if exists)
        # If multiple have same Strahler, pick max Flow Acc
        
        best_r, best_c = -1, -1
        max_acc = -1.0
        
        tribs_r = []
        tribs_c = []
        
        # Scan neighbors
        for i in range(8):
            nr, nc = r + drs[i], c + dcs[i]
            if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc] > 0:
                # Check if flows into current
                nd = flow_dir[nr, nc]
                flows_into = False
                for k in range(8):
                    if nd == codes[k]:
                        tr, tc = nr + drs[k], nc + dcs[k]
                        if tr == r and tc == c:
                            flows_into = True
                        break
                
                if flows_into:
                    tribs_r.append(nr)
                    tribs_c.append(nc)
                    
        if len(tribs_r) == 0:
            continue
            
        # Find main tributary
        # Candidates are those with Strahler == current Strahler (usually max possible)
        # Actually, Strahler logic: Junction order S comes from two S, or one S and others < S.
        # So there should be at least one tributary with Strahler == S (unless it's a source).
        # Wait, if Junction is S+1, then two tribs are S.
        # If Junction is S, then one trib is S, others < S.
        
        # Horton logic: The main stream keeps the order.
        # If we are propagating order X.
        # We look for tribs.
        # If we find tribs with Strahler == Strahler[r,c], one of them continues the main stem.
        # If Strahler[r,c] > max(trib_strahler), then this is a junction where order increased.
        # In that case, BOTH tribs are "main" in their own sub-basins?
        # No, Horton order extends the main stream to the source.
        # So we always try to pick a tributary to continue the CURRENT Horton order.
        
        # But we can only continue Horton order X if the tributary has Strahler order X?
        # No, Horton order replaces Strahler.
        # A 3rd order Horton stream goes all the way to the source.
        # Even if the source is Strahler 1.
        
        # Correct Logic:
        # At any cell with Horton Order H:
        # We look for the "main" upstream tributary.
        # The main tributary gets Horton Order H.
        # All other tributaries get Horton Order = Their Strahler Order (and start a new trace).
        
        # How to define "main"?
        # Usually: Max Flow Accumulation or Longest Path.
        # Let's use Max Flow Accumulation.
        
        best_idx = -1
        max_acc = -1.0
        
        for i in range(len(tribs_r)):
            tr, tc = tribs_r[i], tribs_c[i]
            if flow_acc[tr, tc] > max_acc:
                max_acc = flow_acc[tr, tc]
                best_idx = i
                
        # Assign orders
        for i in range(len(tribs_r)):
            tr, tc = tribs_r[i], tribs_c[i]
            if i == best_idx:
                # Continue main stem
                stack_r.append(tr)
                stack_c.append(tc)
                stack_ord.append(ord_val)
            else:
                # New stream, start with its Strahler order
                stack_r.append(tr)
                stack_c.append(tc)
                stack_ord.append(strahler[tr, tc])
                
    return horton

@jit(nopython=True, cache=True)
def _hack_order_numba(flow_dir, streams, flow_acc, codes, drs, dcs):
    """Calculate Hack (Gravelius) Stream Order."""
    rows, cols = flow_dir.shape
    hack = np.zeros((rows, cols), dtype=np.int32)
    
    # 1. Find outlets
    outlets_r = []
    outlets_c = []
    
    for r in range(rows):
        for c in range(cols):
            if streams[r, c] > 0:
                d = flow_dir[r, c]
                is_outlet = True
                if d > 0:
                    for i in range(8):
                        if d == codes[i]:
                            nr, nc = r + drs[i], c + dcs[i]
                            if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc] > 0:
                                is_outlet = False
                            break
                if is_outlet:
                    outlets_r.append(r)
                    outlets_c.append(c)
                    
    # 2. Trace upstream
    # Stack: (r, c, current_order)
    stack_r = []
    stack_c = []
    stack_ord = []
    
    for i in range(len(outlets_r)):
        r, c = outlets_r[i], outlets_c[i]
        stack_r.append(r)
        stack_c.append(c)
        stack_ord.append(1) # Main stream is 1
        
    while len(stack_r) > 0:
        r = stack_r.pop()
        c = stack_c.pop()
        ord_val = stack_ord.pop()
        
        hack[r, c] = ord_val
        
        # Find upstream tributaries
        tribs_r = []
        tribs_c = []
        
        for i in range(8):
            nr, nc = r + drs[i], c + dcs[i]
            if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc] > 0:
                # Check inflow
                nd = flow_dir[nr, nc]
                flows_into = False
                for k in range(8):
                    if nd == codes[k]:
                        tr, tc = nr + drs[k], nc + dcs[k]
                        if tr == r and tc == c:
                            flows_into = True
                        break
                if flows_into:
                    tribs_r.append(nr)
                    tribs_c.append(nc)
                    
        if len(tribs_r) == 0:
            continue
            
        # Identify main tributary (Max Flow Acc)
        best_idx = -1
        max_acc = -1.0
        
        for i in range(len(tribs_r)):
            tr, tc = tribs_r[i], tribs_c[i]
            if flow_acc[tr, tc] > max_acc:
                max_acc = flow_acc[tr, tc]
                best_idx = i
                
        # Assign orders
        for i in range(len(tribs_r)):
            tr, tc = tribs_r[i], tribs_c[i]
            if i == best_idx:
                # Main stream keeps same order
                stack_r.append(tr)
                stack_c.append(tc)
                stack_ord.append(ord_val)
            else:
                # Tributaries increment order
                stack_r.append(tr)
                stack_c.append(tc)
                stack_ord.append(ord_val + 1)
                
    return hack

@jit(nopython=True, cache=True)
def _calculate_hydrological_slope_numba(dem, flow_dir, codes, drs, dcs, cellsize_x, cellsize_y):
    """Calculate hydrological slope (drop / distance to downstream)."""
    rows, cols = dem.shape
    slope = np.zeros((rows, cols), dtype=np.float32)
    
    diag_dist = np.sqrt(cellsize_x**2 + cellsize_y**2)
    
    for r in range(rows):
        for c in range(cols):
            d = flow_dir[r, c]
            if d > 0:
                for i in range(8):
                    if d == codes[i]:
                        nr, nc = r + drs[i], c + dcs[i]
                        if 0 <= nr < rows and 0 <= nc < cols:
                            drop = dem[r, c] - dem[nr, nc]
                            dist = diag_dist if (drs[i] != 0 and dcs[i] != 0) else cellsize_x
                            if dist > 0:
                                slope[r, c] = drop / dist
                        break
                        
    return slope

@jit(nopython=True, cache=True)
def _flow_distance_upstream_numba(flow_dir, codes, drs, dcs, cellsize_x, cellsize_y):
    """Calculate maximum flow distance to ridge (upstream)."""
    rows, cols = flow_dir.shape
    n_cells = rows * cols
    dist = np.zeros((rows, cols), dtype=np.float32)
    
    # Calculate inflow count
    inflow_count = np.zeros((rows, cols), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            d = flow_dir[r, c]
            if d > 0:
                for i in range(8):
                    if d == codes[i]:
                        nr, nc = r + drs[i], c + dcs[i]
                        if 0 <= nr < rows and 0 <= nc < cols:
                            inflow_count[nr, nc] += 1
                        break
                        
    # Queue with sources (inflow=0)
    # Use pre-allocated array for queue
    queue = np.zeros(n_cells, dtype=np.int32)
    head = 0
    tail = 0
    
    for r in range(rows):
        for c in range(cols):
            if inflow_count[r, c] == 0:
                queue[tail] = r * cols + c
                tail += 1
                
    diag_dist = np.sqrt(cellsize_x**2 + cellsize_y**2)
    
    # Process queue
    while head < tail:
        curr_idx = queue[head]
        head += 1
        
        r = curr_idx // cols
        c = curr_idx % cols
        
        current_dist = dist[r, c]
        
        d = flow_dir[r, c]
        if d > 0:
            for i in range(8):
                if d == codes[i]:
                    nr, nc = r + drs[i], c + dcs[i]
                    if 0 <= nr < rows and 0 <= nc < cols:
                        # Calculate step distance
                        step = diag_dist if (drs[i] != 0 and dcs[i] != 0) else cellsize_x
                        new_dist = current_dist + step
                        
                        # Update max distance
                        if new_dist > dist[nr, nc]:
                            dist[nr, nc] = new_dist
                            
                        # Decrement inflow count
                        inflow_count[nr, nc] -= 1
                        if inflow_count[nr, nc] == 0:
                            queue[tail] = nr * cols + nc
                            tail += 1
                        break
                        
    return dist

@jit(nopython=True, cache=True)
def _join_stream_gaps_numba(streams, flow_dir, codes, drs, dcs, cellsize_x, cellsize_y, gap_threshold):
    """Join gaps in stream network by tracing downstream."""
    rows, cols = streams.shape
    filled_streams = streams.copy()
    
    diag_dist = np.sqrt(cellsize_x**2 + cellsize_y**2)
    
    # 1. Identify "endpoints" - stream cells that flow into NON-stream cells
    # We scan all stream cells.
    
    for r in range(rows):
        for c in range(cols):
            if streams[r, c] > 0:
                d = flow_dir[r, c]
                is_endpoint = False
                if d > 0:
                    for i in range(8):
                        if d == codes[i]:
                            nr, nc = r + drs[i], c + dcs[i]
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if streams[nr, nc] == 0:
                                    is_endpoint = True
                            break
                            
                if is_endpoint:
                    # Trace downstream
                    curr_r, curr_c = r, c
                    accum_dist = 0.0
                    path_r = []
                    path_c = []
                    
                    found_connection = False
                    
                    while accum_dist < gap_threshold:
                        curr_d = flow_dir[curr_r, curr_c]
                        if curr_d <= 0:
                            break # Reached edge or sink
                            
                        # Move downstream
                        next_r, next_c = -1, -1
                        step_dist = 0.0
                        
                        for i in range(8):
                            if curr_d == codes[i]:
                                next_r, next_c = curr_r + drs[i], curr_c + dcs[i]
                                step_dist = diag_dist if (drs[i] != 0 and dcs[i] != 0) else cellsize_x
                                break
                        
                        if next_r == -1: 
                            break
                            
                        if not (0 <= next_r < rows and 0 <= next_c < cols):
                            break
                            
                        # Check if we hit a stream
                        if streams[next_r, next_c] > 0:
                            found_connection = True
                            break
                            
                        # Add to potential path
                        path_r.append(next_r)
                        path_c.append(next_c)
                        accum_dist += step_dist
                        
                        curr_r, curr_c = next_r, next_c
                        
                    if found_connection:
                        # Fill the gap
                        for k in range(len(path_r)):
                            filled_streams[path_r[k], path_c[k]] = 1
                            
    return filled_streams

