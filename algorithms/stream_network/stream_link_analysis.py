# -*- coding: utf-8 -*-
"""
algorithms/stream_network/stream_link_analysis.py
Unified stream link analysis tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterEnum,
    QgsProcessingException
)
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter
import numpy as np

class StreamLinkAlgorithm(QgsProcessingAlgorithm):
    """Unified stream link analysis tool."""
    
    INPUT_STREAMS = 'INPUT_STREAMS'
    INPUT_FLOW_DIR = 'INPUT_FLOW_DIR'
    INPUT_DEM = 'INPUT_DEM'
    METHOD = 'METHOD'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'Stream Link Identifier',
        'Stream Link Length',
        'Stream Link Slope',
        'Stream Link Class',
        'Stream Slope Continuous'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return StreamLinkAlgorithm()
    
    def name(self):
        return 'stream_link_analysis'
    
    def displayName(self):
        return 'Stream Link Analysis'
    
    def group(self):
        return 'Stream Network Analysis'
    
    def groupId(self):
        return 'stream_network'
    
    def shortHelpString(self):
        return """
        Analyze stream links.
        
        - Identifier: Unique ID for each link.
        - Length: Length of each link.
        - Slope: Average slope of each link.
        - Class: Classification (Placeholder).
        - Slope Continuous: Local slope along stream.
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_STREAMS,
                'Input Stream Raster'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_FLOW_DIR,
                'Input Flow Direction'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM,
                'Input DEM (Required for Slope)',
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                'Analysis Method',
                options=self.METHOD_OPTIONS,
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Raster'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            streams_layer = self.parameterAsRasterLayer(parameters, self.INPUT_STREAMS, context)
            flow_dir_layer = self.parameterAsRasterLayer(parameters, self.INPUT_FLOW_DIR, context)
            dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
            method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if streams_layer is None or flow_dir_layer is None:
                raise QgsProcessingException('Invalid inputs')
            
            if method_idx == 2 and dem_layer is None:
                raise QgsProcessingException('DEM is required for Stream Link Slope')
            
            feedback.pushInfo('Loading data...')
            
            # Load inputs using DEMProcessor
            proc_streams = DEMProcessor(streams_layer.source())
            streams = proc_streams.array
            
            proc_flow = DEMProcessor(flow_dir_layer.source())
            flow_dir = proc_flow.array
            
            if dem_layer:
                proc_dem = DEMProcessor(dem_layer.source())
                dem_array = proc_dem.array
                main_proc = proc_dem
            else:
                dem_array = streams # Dummy DEM (shape only)
                main_proc = proc_streams
                
            # Initialize router
            router = FlowRouter(dem_array, main_proc.cellsize_x, main_proc.cellsize_y, main_proc.nodata)
            
            method_map = {
                0: 'id',
                1: 'length',
                2: 'slope',
                3: 'class',
                4: 'slope_continuous'
            }
            
            stat_type = method_map.get(method_idx, 'id')
            
            feedback.pushInfo(f'Calculating {self.METHOD_OPTIONS[method_idx]}...')
            feedback.setProgress(30)
            
            result = router.calculate_stream_link_statistics(stat_type, flow_dir, streams)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            main_proc.save_raster(output_path, result, nodata=-9999)
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
