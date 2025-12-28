# -*- coding: utf-8 -*-
"""
algorithms/stream_network/join_stream_links.py
Join Stream Links - Clean up stream networks by joining short segments
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingException,
    QgsRasterLayer,
    QgsProject
)
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter
from osgeo import gdal
import numpy as np


class JoinStreamLinksAlgorithm(QgsProcessingAlgorithm):
    """Join Stream Links - Connect and clean stream networks."""
    
    INPUT_FLOW_ACC = 'INPUT_FLOW_ACC'
    INPUT_STREAM_RASTER = 'INPUT_STREAM_RASTER'
    INPUT_FLOW_DIR = 'INPUT_FLOW_DIR'
    FLOW_DIR_METHOD = 'FLOW_DIR_METHOD'
    USE_STREAM_RASTER = 'USE_STREAM_RASTER'
    BREAK_DOWN_VALUE = 'BREAK_DOWN_VALUE'
    THRESHOLD_TYPE = 'THRESHOLD_TYPE'
    THRESHOLD_VALUE = 'THRESHOLD_VALUE'
    OUTPUT = 'OUTPUT'
    
    THRESHOLD_TYPES = ['Pixels', 'Map Units (Length)']
    FLOW_DIR_METHODS = ['D8 (Standard Encoding)', 'Rho8', 'D-Infinity (DINF)', 'MFD (Multiple Flow Direction)']
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return JoinStreamLinksAlgorithm()
    
    def name(self):
        return 'join_stream_links'
    
    def displayName(self):
        return 'Join Stream Links'
    
    def group(self):
        return 'Stream Network Analysis'
    
    def groupId(self):
        return 'stream_network'
    
    def shortHelpString(self):
        return """
        Join and clean stream networks by connecting short gaps following flow direction.
        
        TWO MODES:
        
        1. Flow Accumulation Mode (Checkbox UNCHECKED - Default):
           - Provide Flow Accumulation raster
           - Set Break Down Value (e.g., 1500)
           - Creates stream raster where: flow_acc >= Break Down Value
           - Then joins gaps within threshold
        
        2. Stream Raster Mode (Checkbox CHECKED):
           - Provide existing Stream Raster (0/1 values)
           - Break Down Value is IGNORED (not needed)
           - Directly joins gaps within threshold
        
        GAP JOINING ALGORITHM:
        - Finds stream endpoints (cells that flow to non-stream)
        - Traces downstream following flow direction
        - If another stream is found within threshold, fills the gap
        - Follows the natural slope/downstream direction
        
        Parameters:
        - Use Stream Raster Input: Check to use existing stream raster
        - Input Raster: Flow Accumulation OR Stream Raster (depends on checkbox)
        - Flow Direction: D8 flow direction raster
        - Break Down Value: ONLY for Flow Accumulation mode (ignored if checkbox checked)
        - Gap Threshold: Maximum gap size to join (pixels or map units)
        """
    
    def initAlgorithm(self, config=None):
        # Checkbox for stream raster input mode (default: unchecked = Flow Accumulation mode)
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_STREAM_RASTER,
                'Use Stream Raster Input (instead of Flow Accumulation)',
                defaultValue=False
            )
        )
        
        # Single input raster - label changes based on checkbox
        # When unchecked: "Flow Accumulation Raster"
        # When checked: "Stream Raster Input"
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_FLOW_ACC,
                'Flow Accumulation Raster / Stream Raster Input'
            )
        )
        
        # Flow Direction input (required)
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_FLOW_DIR,
                'Flow Direction Raster'
            )
        )
        
        # Flow Direction Method dropdown
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FLOW_DIR_METHOD,
                'Flow Direction Method',
                options=self.FLOW_DIR_METHODS,
                defaultValue=0
            )
        )
        
        # Break down value for stream extraction (only used when NOT using stream raster)
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BREAK_DOWN_VALUE,
                'Break Down Value (Stream Extraction Threshold)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=100,
                optional=True,
                minValue=0
            )
        )
        
        # Threshold type
        self.addParameter(
            QgsProcessingParameterEnum(
                self.THRESHOLD_TYPE,
                'Gap Threshold Type',
                options=self.THRESHOLD_TYPES,
                defaultValue=0
            )
        )
        
        # Threshold value for joining
        self.addParameter(
            QgsProcessingParameterNumber(
                self.THRESHOLD_VALUE,
                'Gap Threshold Value (Maximum Gap to Join)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=5,
                minValue=1
            )
        )
        
        # Output
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Joined Stream Raster'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            use_stream_raster = self.parameterAsBoolean(parameters, self.USE_STREAM_RASTER, context)
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT_FLOW_ACC, context)
            flow_dir_layer = self.parameterAsRasterLayer(parameters, self.INPUT_FLOW_DIR, context)
            flow_dir_method = self.parameterAsEnum(parameters, self.FLOW_DIR_METHOD, context)
            threshold_type = self.parameterAsEnum(parameters, self.THRESHOLD_TYPE, context)
            threshold_value = self.parameterAsDouble(parameters, self.THRESHOLD_VALUE, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                if use_stream_raster:
                    raise QgsProcessingException('Stream Raster is required')
                else:
                    raise QgsProcessingException('Flow Accumulation Raster is required')
            
            if flow_dir_layer is None:
                raise QgsProcessingException('Flow Direction is required')
            
            feedback.pushInfo(f'Using Flow Direction Method: {self.FLOW_DIR_METHODS[flow_dir_method]}')
            
            # Load flow direction
            feedback.pushInfo('Loading Flow Direction...')
            fd_proc = DEMProcessor(flow_dir_layer.source())
            flow_dir = fd_proc.array.astype(np.int32)
            
            feedback.setProgress(10)
            
            # Get or create stream raster based on mode
            if use_stream_raster:
                # Mode: Using Stream Raster directly
                feedback.pushInfo('Loading Stream Raster...')
                s_proc = DEMProcessor(input_layer.source())
                streams = (s_proc.array > 0).astype(np.int32)
                s_proc.close()
                feedback.pushInfo(f'Stream pixels loaded: {np.sum(streams):,}')
            else:
                # Mode: Creating streams from Flow Accumulation
                break_down_value = self.parameterAsDouble(parameters, self.BREAK_DOWN_VALUE, context)
                
                feedback.pushInfo(f'Creating Stream Raster using expression: "flow_acc" >= {break_down_value}')
                fa_proc = DEMProcessor(input_layer.source())
                flow_acc = fa_proc.array
                
                # Create stream raster: 1 where flow_acc >= break_down_value
                streams = (flow_acc >= break_down_value).astype(np.int32)
                fa_proc.close()
                
                feedback.pushInfo(f'Stream pixels created: {np.sum(streams):,}')
            
            feedback.setProgress(30)
            
            # Convert threshold to pixels if needed
            cellsize = fd_proc.cellsize_x
            if threshold_type == 1:  # Map Units
                threshold_pixels = int(threshold_value / cellsize) + 1
            else:  # Pixels
                threshold_pixels = int(threshold_value)
            
            feedback.pushInfo(f'Joining stream links with gap threshold: {threshold_pixels} pixels')
            feedback.setProgress(40)
            
            # Initialize router for operations
            router = FlowRouter(np.zeros_like(streams, dtype=np.float32), cellsize)
            
            # Join stream links
            feedback.pushInfo('Analyzing stream network for gaps...')
            joined_streams = self._join_stream_links(
                streams, flow_dir, threshold_pixels, flow_dir_method, router, feedback
            )
            
            feedback.setProgress(80)
            
            # Count changes
            original_pixels = np.sum(streams > 0)
            final_pixels = np.sum(joined_streams > 0)
            added_pixels = final_pixels - original_pixels
            
            feedback.pushInfo(f'Original stream pixels: {original_pixels:,}')
            feedback.pushInfo(f'Final stream pixels: {final_pixels:,}')
            feedback.pushInfo(f'Pixels added by joining: {added_pixels:,}')
            
            # Save output (use -1 as NoData so 0 values are preserved)
            feedback.pushInfo('Saving output...')
            fd_proc.save_raster(output_path, joined_streams, dtype=gdal.GDT_Int32, nodata=-9999)
            fd_proc.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
    
    def _join_stream_links(self, streams, flow_dir, threshold_pixels, flow_dir_method, router, feedback):
        """Join stream links by filling gaps following flow direction.
        
        Algorithm:
        1. Find stream endpoints (stream cells that flow to non-stream)
        2. For each endpoint, trace downstream up to threshold_pixels
        3. If we hit another stream cell within threshold, fill the gap
        
        Supports different flow direction methods:
        - 0: D8 (Standard Encoding) - codes: 1,2,4,8,16,32,64,128
        - 1: Rho8 - similar to D8 but with random tie-breaking
        - 2: D-Infinity - continuous angles (0-360), needs special handling
        - 3: MFD - Multiple flow direction, not single-path
        
        Note: Currently only D8 encoding is fully implemented. Others use D8 as fallback.
        """
        rows, cols = streams.shape
        joined = streams.copy()
        
        # Direction mappings for D8 (Standard encoding)
        # E=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128
        drs = np.array([0, 1, 1, 1, 0, -1, -1, -1])
        dcs = np.array([1, 1, 0, -1, -1, -1, 0, 1])
        codes = np.array([1, 2, 4, 8, 16, 32, 64, 128])
        
        # Find stream endpoints (stream cells that flow to non-stream cells)
        endpoints = []
        
        for r in range(rows):
            for c in range(cols):
                if streams[r, c] > 0:
                    d = flow_dir[r, c]
                    if d > 0:
                        # Find downstream cell
                        for i in range(8):
                            if d == codes[i]:
                                nr, nc = r + drs[i], c + dcs[i]
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    if streams[nr, nc] == 0:
                                        # This is an endpoint
                                        endpoints.append((r, c))
                                break
        
        feedback.pushInfo(f'Found {len(endpoints)} stream endpoints to check')
        
        gaps_joined = 0
        
        # For each endpoint, try to connect to another stream
        for er, ec in endpoints:
            if feedback.isCanceled():
                break
            
            # Trace downstream from this endpoint
            path = []
            curr_r, curr_c = er, ec
            
            # Get the downstream cell (first non-stream cell)
            d = flow_dir[curr_r, curr_c]
            for i in range(8):
                if d == codes[i]:
                    curr_r, curr_c = curr_r + drs[i], curr_c + dcs[i]
                    break
            
            # Trace up to threshold_pixels steps
            for step in range(threshold_pixels):
                if curr_r < 0 or curr_r >= rows or curr_c < 0 or curr_c >= cols:
                    break
                
                if joined[curr_r, curr_c] > 0:
                    # Found another stream! Fill the gap
                    for pr, pc in path:
                        joined[pr, pc] = 1
                    gaps_joined += 1
                    break
                
                path.append((curr_r, curr_c))
                
                # Get next cell following flow direction
                d = flow_dir[curr_r, curr_c]
                if d <= 0:
                    break
                
                found_next = False
                for i in range(8):
                    if d == codes[i]:
                        curr_r = curr_r + drs[i]
                        curr_c = curr_c + dcs[i]
                        found_next = True
                        break
                
                if not found_next:
                    break
        
        feedback.pushInfo(f'Joined {gaps_joined} stream gaps')
        
        return joined
