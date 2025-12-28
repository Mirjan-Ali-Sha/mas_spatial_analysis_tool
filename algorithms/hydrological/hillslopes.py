# -*- coding: utf-8 -*-
"""
algorithms/hydrological/hillslopes.py
Unified hillslopes tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException
)
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter

class HillslopesAlgorithm(QgsProcessingAlgorithm):
    """Unified hillslopes tool."""
    
    INPUT_STREAMS = 'INPUT_STREAMS'
    INPUT_DIR = 'INPUT_DIR'
    OUTPUT = 'OUTPUT'
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return HillslopesAlgorithm()
    
    def name(self):
        return 'hillslopes'
    
    def displayName(self):
        return 'Hillslopes'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Delineate hillslopes for each stream link.
        
        Each hillslope is assigned the ID of the stream link it drains into.
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_STREAMS,
                'Stream Link Raster'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DIR,
                'D8 Flow Direction'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Hillslopes'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            stream_layer = self.parameterAsRasterLayer(parameters, self.INPUT_STREAMS, context)
            dir_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DIR, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if stream_layer is None or dir_layer is None:
                raise QgsProcessingException('Stream Links and Flow Direction are required')
            
            feedback.pushInfo('Loading data...')
            dir_proc = DEMProcessor(dir_layer.source())
            stream_proc = DEMProcessor(stream_layer.source())
            
            if dir_proc.array.shape != stream_proc.array.shape:
                raise QgsProcessingException('Input rasters must have same dimensions')
                
            router = FlowRouter(dir_proc.array, dir_proc.cellsize_x)
            router.flow_dir = dir_proc.array
            streams = stream_proc.array
            
            feedback.pushInfo('Delineating hillslopes...')
            feedback.setProgress(20)
            
            # Ensure streams have unique IDs?
            # If input is just 1s, we should assign IDs first?
            # The tool expects "Stream Link Raster", implying IDs.
            # But if user passes binary streams, we should probably assign IDs.
            # Let's check max value.
            if streams.max() <= 1:
                feedback.pushInfo('Assigning stream link IDs...')
                streams = router.assign_stream_link_ids(router.flow_dir, streams)
            
            result = router.delineate_watersheds(streams)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            dir_proc.save_raster(output_path, result)
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
