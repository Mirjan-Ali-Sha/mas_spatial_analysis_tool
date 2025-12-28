# -*- coding: utf-8 -*-
"""
algorithms/hydrological/sink_analysis.py
Unified sink analysis tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterEnum,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException
)
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter
import numpy as np

class SinkAnalysisAlgorithm(QgsProcessingAlgorithm):
    """Unified sink analysis tool."""
    
    INPUT = 'INPUT'
    METHOD = 'METHOD'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'Depth in Sink',
        'Identify Sinks'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return SinkAnalysisAlgorithm()
    
    def name(self):
        return 'sink_analysis'
    
    def displayName(self):
        return 'Sink Analysis'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Analyze depressions (sinks) in the DEM.
        
        - Depth in Sink: Calculates the depth of each sink (Filled DEM - Original DEM).
        - Identify Sinks: Returns a boolean raster where 1 indicates a sink.
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                'Input DEM'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                'Method',
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
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
            method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = DEMProcessor(input_layer.source())
            router = FlowRouter(processor.array, processor.cellsize_x)
            
            feedback.pushInfo('Filling depressions to identify sinks...')
            feedback.setProgress(20)
            
            # Calculate filled DEM
            filled_dem = router.fill_depressions()
            
            feedback.pushInfo('Calculating sink properties...')
            feedback.setProgress(60)
            
            if method_idx == 0: # Depth in Sink
                # Depth = Filled - Original
                original_dem = processor.array
                result = filled_dem - original_dem
                
            elif method_idx == 1: # Identify Sinks
                original_dem = processor.array
                diff = filled_dem - original_dem
                result = np.where(diff > 0, 1, 0).astype(np.int8)
                # Handle nodata
                result[np.isnan(original_dem)] = 0 # Sinks are usually 0/1 mask
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
