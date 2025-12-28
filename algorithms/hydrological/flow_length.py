# -*- coding: utf-8 -*-
"""
algorithms/hydrological/flow_length.py
Unified flow length tool
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

class FlowLengthAlgorithm(QgsProcessingAlgorithm):
    """Unified flow length tool."""
    
    INPUT_DIR = 'INPUT_DIR'
    METHOD = 'METHOD'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'Downslope Flowpath Length (Distance to Outlet)',
        'Upslope Flowpath Length (Distance to Ridge)'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return FlowLengthAlgorithm()
    
    def name(self):
        return 'flow_length'
    
    def displayName(self):
        return 'Flow Length'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Calculate flowpath lengths.
        
        - Downslope: Distance from each cell to the outlet (or nodata).
        - Upslope: Maximum distance from each cell to the drainage divide (ridge).
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DIR,
                'D8 Flow Direction'
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
            dir_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DIR, context)
            method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if dir_layer is None:
                raise QgsProcessingException('Flow Direction is required')
            
            feedback.pushInfo('Loading data...')
            processor = DEMProcessor(dir_layer.source())
            router = FlowRouter(processor.array, processor.cellsize_x)
            router.flow_dir = processor.array
            
            feedback.pushInfo('Calculating flow length...')
            feedback.setProgress(20)
            
            if method_idx == 0: # Downslope
                result = router.calculate_flow_distance(router.dem, distance_type='outlet')
            else: # Upslope
                # I need to ensure 'upstream' is implemented in calculate_flow_distance
                # Currently it returns zeros in the placeholder.
                # I should implement it or warn.
                # Let's implement it in FlowRouter now.
                result = router.calculate_flow_distance(router.dem, distance_type='upstream')
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result)
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
