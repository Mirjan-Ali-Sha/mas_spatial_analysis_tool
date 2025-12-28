# -*- coding: utf-8 -*-
"""
algorithms/stream_network/stream_network_analysis.py
Unified stream network analysis tool
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

class StreamNetworkAnalysisAlgorithm(QgsProcessingAlgorithm):
    """Unified stream network analysis tool."""
    
    INPUT_STREAMS = 'INPUT_STREAMS'
    INPUT_FLOW_DIR = 'INPUT_FLOW_DIR'
    INPUT_FLOW_ACC = 'INPUT_FLOW_ACC'
    ANALYSIS_TYPE = 'ANALYSIS_TYPE'
    OUTPUT = 'OUTPUT'
    
    TYPE_OPTIONS = ['Main Stream Identification', 'Tributary ID']
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return StreamNetworkAnalysisAlgorithm()
    
    def name(self):
        return 'stream_network_analysis'
    
    def displayName(self):
        return 'Stream Network Analysis'
    
    def group(self):
        return 'Stream Network Analysis'
    
    def groupId(self):
        return 'stream_network'
    
    def shortHelpString(self):
        return """
        Analyze stream network properties.
        
        - Main Stream: Identifies the main channel (longest path/max accumulation) from outlets.
        - Tributary ID: Assigns unique IDs to tributaries.
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
                self.INPUT_FLOW_ACC,
                'Input Flow Accumulation',
                optional=True # Only needed for Main Stream
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ANALYSIS_TYPE,
                'Analysis Type',
                options=self.TYPE_OPTIONS,
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
            flow_acc_layer = self.parameterAsRasterLayer(parameters, self.INPUT_FLOW_ACC, context)
            type_idx = self.parameterAsEnum(parameters, self.ANALYSIS_TYPE, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if streams_layer is None or flow_dir_layer is None:
                raise QgsProcessingException('Invalid inputs')
            
            feedback.pushInfo('Loading data...')
            processor = DEMProcessor(streams_layer.source())
            streams = processor.array
            
            fd_processor = DEMProcessor(flow_dir_layer.source())
            flow_dir = fd_processor.array
            
            router = FlowRouter(processor.array, processor.cellsize_x)
            
            if type_idx == 0: # Main Stream
                if flow_acc_layer is None:
                    raise QgsProcessingException('Flow Accumulation required for Main Stream analysis')
                
                fa_processor = DEMProcessor(flow_acc_layer.source())
                flow_acc = fa_processor.array
                
                feedback.pushInfo('Identifying Main Stream...')
                result = router.find_main_stream(flow_dir, streams, flow_acc)
                fa_processor.close()
                
            else: # Tributary ID
                feedback.pushInfo('Assigning Tributary IDs...')
                result = router.assign_tributary_ids(flow_dir, streams)
            
            feedback.pushInfo('Saving output...')
            processor.save_raster(output_path, result, nodata=-9999)
            processor.close()
            fd_processor.close()
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
