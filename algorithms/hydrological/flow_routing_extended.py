# -*- coding: utf-8 -*-
"""
algorithms/hydrological/flow_routing_extended.py
Unified advanced flow routing tool
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

class FlowRoutingAlgorithm(QgsProcessingAlgorithm):
    """Unified advanced flow routing tool."""
    
    INPUT = 'INPUT'
    METHOD = 'METHOD'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = ['FD8 (Freeman)', 'D8 (Standard)']
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return FlowRoutingAlgorithm()
    
    def name(self):
        return 'flow_routing_extended'
    
    def displayName(self):
        return 'Advanced Flow Routing'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Calculate flow accumulation using advanced routing methods.
        
        Methods:
        - FD8 (Freeman): Multiple flow direction, distributes flow to all lower neighbors. Good for divergent flow.
        - D8 (Standard): Single flow direction.
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
                'Routing Method',
                options=self.METHOD_OPTIONS,
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Accumulation'
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
            
            feedback.pushInfo(f'Calculating {self.METHOD_OPTIONS[method_idx]}...')
            feedback.setProgress(20)
            
            router = FlowRouter(processor.array, processor.cellsize_x)
            
            if method_idx == 0:  # FD8
                result = router.fd8_flow_accumulation()
            else:  # D8
                flow_dir = router.d8_flow_direction()
                result = router.d8_flow_accumulation(flow_dir)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
