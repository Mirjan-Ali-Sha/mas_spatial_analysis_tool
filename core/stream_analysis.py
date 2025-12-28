# -*- coding: utf-8 -*-
"""
Stream network analysis utilities
Native implementation for stream ordering and characterization
"""

import numpy as np
from scipy import ndimage
from collections import defaultdict, deque


class StreamAnalyzer:
    """Stream network analysis tools."""
    
    def __init__(self, stream_raster, flow_dir, dem, cellsize):
        """Initialize stream analyzer.
        
        Args:
            stream_raster (np.ndarray): Binary stream network (1=stream)
            flow_dir (np.ndarray): D8 flow direction
            dem (np.ndarray): DEM array
            cellsize (float): Cell size
        """
        self.streams = stream_raster.copy()
        self.flow_dir = flow_dir
        self.dem = dem
        self.cellsize = cellsize
        self.rows, self.cols = stream_raster.shape
        
        # Direction mappings
        self.dir_codes = [1, 2, 4, 8, 16, 32, 64, 128]
        self.directions = [(0, 1), (1, 1), (1, 0), (1, -1),
                          (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    
    def strahler_stream_order(self):
        """Calculate Strahler stream order.
        
        Returns:
            np.ndarray: Stream order array
        """
        order = np.zeros_like(self.streams, dtype=np.int32)
        
        # Find stream network topology
        topology = self._build_stream_topology()
        
        # Find headwater streams
        headwaters = self._find_headwaters()
        
        # Initialize headwaters with order 1
        for row, col in headwaters:
            order[row, col] = 1
        
        # Process downstream
        processed = set(headwaters)
        to_process = deque(headwaters)
        
        while to_process:
            row, col = to_process.popleft()
            current_order = order[row, col]
            
            # Find downstream cell
            downstream = self._get_downstream_cell(row, col)
            
            if downstream is None:
                continue
            
            dr, dc = downstream
            
            # Find all upstream tributaries
            upstream_orders = []
            for ur, uc in self._get_upstream_cells(dr, dc):
                if order[ur, uc] > 0:
                    upstream_orders.append(order[ur, uc])
            
            # Apply Strahler rules
            if len(upstream_orders) == 0:
                new_order = 1
            elif len(upstream_orders) == 1:
                new_order = upstream_orders[0]
            else:
                upstream_orders.sort(reverse=True)
                if upstream_orders[0] == upstream_orders[1]:
                    new_order = upstream_orders[0] + 1
                else:
                    new_order = upstream_orders[0]
            
            order[dr, dc] = max(order[dr, dc], new_order)
            
            if (dr, dc) not in processed:
                processed.add((dr, dc))
                to_process.append((dr, dc))
        
        return order
    
    def shreve_stream_magnitude(self):
        """Calculate Shreve stream magnitude.
        
        Returns:
            np.ndarray: Stream magnitude array
        """
        magnitude = np.zeros_like(self.streams, dtype=np.int32)
        
        # Find headwaters
        headwaters = self._find_headwaters()
        
        # Initialize headwaters with magnitude 1
        for row, col in headwaters:
            magnitude[row, col] = 1
        
        # Sort stream cells by elevation (high to low)
        stream_cells = np.argwhere(self.streams == 1)
        elevations = self.dem[self.streams == 1]
        sorted_indices = np.argsort(elevations)[::-1]
        sorted_cells = stream_cells[sorted_indices]
        
        # Process downstream
        for row, col in sorted_cells:
            if magnitude[row, col] == 0:
                magnitude[row, col] = 1
            
            # Add to downstream cell
            downstream = self._get_downstream_cell(row, col)
            if downstream:
                dr, dc = downstream
                magnitude[dr, dc] += magnitude[row, col]
        
        return magnitude
    
    def hack_stream_order(self):
        """Calculate Hack stream order (main stem ordering).
        
        Returns:
            np.ndarray: Hack order array
        """
        order = np.zeros_like(self.streams, dtype=np.int32)
        
        # Find outlets
        outlets = self._find_outlets()
        
        current_order = 1
        
        for outlet_row, outlet_col in outlets:
            # Trace main stem upstream
            main_stem = self._trace_main_stem(outlet_row, outlet_col)
            
            for row, col in main_stem:
                order[row, col] = current_order
            
            current_order += 1
            
            # Process tributaries
            order = self._process_tributaries_hack(main_stem, order, current_order)
        
        return order
    
    def calculate_stream_slope(self):
        """Calculate slope along stream channels.
        
        Returns:
            np.ndarray: Stream slope array
        """
        slope = np.zeros_like(self.streams, dtype=np.float32)
        
        stream_cells = np.argwhere(self.streams == 1)
        
        for row, col in stream_cells:
            # Get upstream and downstream elevations
            downstream = self._get_downstream_cell(row, col)
            
            if downstream:
                dr, dc = downstream
                elev_drop = self.dem[row, col] - self.dem[dr, dc]
                
                # Calculate distance
                distance = self.cellsize
                if abs(dr - row) == 1 and abs(dc - col) == 1:
                    distance *= np.sqrt(2)
                
                # Calculate slope
                if distance > 0:
                    slope[row, col] = elev_drop / distance
        
        # Convert to degrees
        slope = np.degrees(np.arctan(slope))
        slope[self.streams == 0] = 0
        
        return slope
    
    def stream_link_identifier(self):
        """Identify stream links (segments between junctions).
        
        Returns:
            np.ndarray: Link ID array
        """
        links = np.zeros_like(self.streams, dtype=np.int32)
        
        # Find junctions and endpoints
        junctions = self._find_junctions()
        headwaters = self._find_headwaters()
        outlets = self._find_outlets()
        
        link_id = 1
        processed = set()
        
        # Process from each headwater
        for start_row, start_col in headwaters:
            if (start_row, start_col) in processed:
                continue
            
            # Trace until junction or outlet
            current = (start_row, start_col)
            link_cells = []
            
            while current:
                row, col = current
                
                if current in processed:
                    break
                
                link_cells.append(current)
                processed.add(current)
                
                # Assign link ID
                links[row, col] = link_id
                
                # Check if at junction or outlet
                if current in junctions or current in outlets:
                    break
                
                # Move downstream
                downstream = self._get_downstream_cell(row, col)
                if downstream is None:
                    break
                
                current = downstream
            
            if link_cells:
                link_id += 1
        
        return links
    
    def _build_stream_topology(self):
        """Build stream network topology.
        
        Returns:
            dict: Topology dictionary
        """
        topology = defaultdict(list)
        
        stream_cells = np.argwhere(self.streams == 1)
        
        for row, col in stream_cells:
            downstream = self._get_downstream_cell(row, col)
            if downstream:
                topology[(row, col)].append(downstream)
        
        return topology
    
    def _find_headwaters(self):
        """Find headwater cells (no upstream streams).
        
        Returns:
            list: List of (row, col) tuples
        """
        headwaters = []
        stream_cells = np.argwhere(self.streams == 1)
        
        for row, col in stream_cells:
            upstream_count = 0
            
            for ur, uc in self._get_upstream_cells(row, col):
                if self.streams[ur, uc] == 1:
                    upstream_count += 1
            
            if upstream_count == 0:
                headwaters.append((row, col))
        
        return headwaters
    
    def _find_outlets(self):
        """Find outlet cells (no downstream streams).
        
        Returns:
            list: List of (row, col) tuples
        """
        outlets = []
        stream_cells = np.argwhere(self.streams == 1)
        
        for row, col in stream_cells:
            downstream = self._get_downstream_cell(row, col)
            if downstream is None:
                outlets.append((row, col))
        
        return outlets
    
    def _find_junctions(self):
        """Find junction cells (multiple upstream streams).
        
        Returns:
            list: List of (row, col) tuples
        """
        junctions = []
        stream_cells = np.argwhere(self.streams == 1)
        
        for row, col in stream_cells:
            upstream_count = 0
            
            for ur, uc in self._get_upstream_cells(row, col):
                if self.streams[ur, uc] == 1:
                    upstream_count += 1
            
            if upstream_count > 1:
                junctions.append((row, col))
        
        return junctions
    
    def _get_downstream_cell(self, row, col):
        """Get downstream cell for given cell.
        
        Args:
            row (int): Row index
            col (int): Column index
            
        Returns:
            tuple: (row, col) of downstream cell or None
        """
        dir_code = self.flow_dir[row, col]
        
        if dir_code == 0:
            return None
        
        # Find direction index
        try:
            idx = self.dir_codes.index(dir_code)
            dr, dc = self.directions[idx]
            nr, nc = row + dr, col + dc
            
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.streams[nr, nc] == 1:
                    return (nr, nc)
        except ValueError:
            pass
        
        return None
    
    def _get_upstream_cells(self, row, col):
        """Get all upstream cells for given cell.
        
        Args:
            row (int): Row index
            col (int): Column index
            
        Returns:
            list: List of (row, col) tuples
        """
        upstream = []
        
        for idx, (dr, dc) in enumerate(self.directions):
            nr, nc = row - dr, col - dc
            
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.flow_dir[nr, nc] == self.dir_codes[idx]:
                    upstream.append((nr, nc))
        
        return upstream
    
    def _trace_main_stem(self, start_row, start_col):
        """Trace main stem (longest path) upstream.
        
        Args:
            start_row (int): Starting row
            start_col (int): Starting column
            
        Returns:
            list: List of (row, col) tuples forming main stem
        """
        main_stem = [(start_row, start_col)]
        current = (start_row, start_col)
        
        while True:
            upstream_cells = self._get_upstream_cells(current[0], current[1])
            upstream_streams = [(r, c) for r, c in upstream_cells 
                               if self.streams[r, c] == 1]
            
            if not upstream_streams:
                break
            
            # Choose upstream with highest elevation (longest path)
            upstream_elevs = [self.dem[r, c] for r, c in upstream_streams]
            max_idx = np.argmax(upstream_elevs)
            current = upstream_streams[max_idx]
            main_stem.append(current)
        
        return main_stem
    
    def _process_tributaries_hack(self, main_stem, order, current_order):
        """Process tributaries for Hack ordering.
        
        Args:
            main_stem (list): Main stem cells
            order (np.ndarray): Current order array
            current_order (int): Current order value
            
        Returns:
            np.ndarray: Updated order array
        """
        for row, col in main_stem:
            upstream_cells = self._get_upstream_cells(row, col)
            
            for ur, uc in upstream_cells:
                if self.streams[ur, uc] == 1 and order[ur, uc] == 0:
                    # This is a tributary
                    tributary = self._trace_main_stem(ur, uc)
                    
                    for tr, tc in tributary:
                        order[tr, tc] = current_order
                    
                    current_order += 1
                    
                    # Recursively process tributary's tributaries
                    order = self._process_tributaries_hack(tributary, order, current_order)
        
        return order
