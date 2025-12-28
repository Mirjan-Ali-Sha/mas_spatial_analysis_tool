# -*- coding: utf-8 -*-
"""
algorithms/stream_network/stream_cleaning.py
Unified stream cleaning tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException
)
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter

class StreamCleaningAlgorithm(QgsProcessingAlgorithm):
    """Unified stream cleaning tool."""
    
    INPUT_STREAMS = 'INPUT_STREAMS'
    INPUT_DIR = 'INPUT_DIR'
    MIN_LENGTH = 'MIN_LENGTH'
    OUTPUT = 'OUTPUT'
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return StreamCleaningAlgorithm()
    
    def name(self):
        return 'stream_cleaning'
    
    def displayName(self):
        return 'Stream Network Cleaning'
    
    def group(self):
        return 'Stream Network Analysis'
    
    def groupId(self):
        return 'stream_network'
    
    def shortHelpString(self):
        return """
        Clean stream network by removing short links.
        
        - Remove Short Streams: Removes stream links shorter than the specified threshold.
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_STREAMS,
                'Stream Raster'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DIR,
                'D8 Flow Direction'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_LENGTH,
                'Minimum Length (map units)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=100.0,
                minValue=0.0
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
            stream_layer = self.parameterAsRasterLayer(parameters, self.INPUT_STREAMS, context)
            dir_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DIR, context)
            min_length = self.parameterAsDouble(parameters, self.MIN_LENGTH, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if stream_layer is None or dir_layer is None:
                raise QgsProcessingException('Stream Raster and Flow Direction are required')
            
            feedback.pushInfo('Loading data...')
            dir_proc = DEMProcessor(dir_layer.source())
            stream_proc = DEMProcessor(stream_layer.source())
            
            # Ensure dimensions match
            if dir_proc.array.shape != stream_proc.array.shape:
                raise QgsProcessingException('Input rasters must have same dimensions')
                
            router = FlowRouter(dir_proc.array, dir_proc.cellsize_x)
            # We need to set flow_dir in router if it's not the DEM
            router.flow_dir = dir_proc.array
            streams = stream_proc.array
            flow_dir = dir_proc.array
            
            feedback.pushInfo('Removing short streams...')
            feedback.setProgress(20)
            
            result = router.remove_short_streams(flow_dir, streams, min_length)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            # Save using stream processor to keep metadata
            stream_proc.save_raster(output_path, result)
            stream_proc.close()
            dir_proc.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
