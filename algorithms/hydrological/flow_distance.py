# -*- coding: utf-8 -*-
"""
algorithms/hydrological/flow_distance.py
Unified flow distance tool
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

class FlowDistanceAlgorithm(QgsProcessingAlgorithm):
    """Unified flow distance tool."""
    
    INPUT = 'INPUT'
    DISTANCE_TYPE = 'DISTANCE_TYPE'
    OUTPUT = 'OUTPUT'
    
    TYPE_OPTIONS = ['Distance to Outlet (Downstream)']
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return FlowDistanceAlgorithm()
    
    def name(self):
        return 'flow_distance'
    
    def displayName(self):
        return 'Flow Distance'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Calculate flow distance.
        
        Currently supports:
        - Distance to Outlet: Distance from each cell to the nearest outlet (nodata or edge) following the flow path.
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
                self.DISTANCE_TYPE,
                'Distance Type',
                options=self.TYPE_OPTIONS,
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Distance Raster'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
            type_idx = self.parameterAsEnum(parameters, self.DISTANCE_TYPE, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = DEMProcessor(input_layer.source())
            
            feedback.pushInfo('Calculating Flow Direction...')
            feedback.setProgress(20)
            
            router = FlowRouter(processor.array, processor.cellsize_x)
            flow_dir = router.d8_flow_direction()
            
            feedback.pushInfo('Calculating Flow Distance...')
            feedback.setProgress(50)
            
            result = router.calculate_flow_distance(flow_dir, distance_type='outlet')
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result, nodata=-9999)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
