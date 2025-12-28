# -*- coding: utf-8 -*-
"""
algorithms/hydrological/basin_analysis.py
Unified basin analysis tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException
)
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter
import numpy as np

class BasinAnalysisAlgorithm(QgsProcessingAlgorithm):
    """Unified basin analysis tool."""
    
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return BasinAnalysisAlgorithm()
    
    def name(self):
        return 'basin_analysis'
    
    def displayName(self):
        return 'Basin Analysis'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Delineate all drainage basins in the DEM.
        
        Identifies all outlets (sinks or edge cells) and delineates their contributing areas.
        Each basin is assigned a unique ID.
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                'Input DEM'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Basins'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = DEMProcessor(input_layer.source())
            
            feedback.pushInfo('Calculating Flow Direction...')
            feedback.setProgress(20)
            
            router = FlowRouter(processor.array, processor.cellsize_x)
            flow_dir = router.d8_flow_direction()
            
            feedback.pushInfo('Delineating Basins...')
            feedback.setProgress(50)
            
            result = router.delineate_basins(flow_dir)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result, nodata=-9999)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
