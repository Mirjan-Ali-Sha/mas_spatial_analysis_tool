# -*- coding: utf-8 -*-
"""
algorithms/hydrological/dem_quality.py
Unified DEM quality check tool
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

class DemQualityAlgorithm(QgsProcessingAlgorithm):
    """Unified DEM quality check tool."""
    
    INPUT_DIR = 'INPUT_DIR'
    METHOD = 'METHOD'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'Find No Flow Cells',
        'Find Parallel Flow'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return DemQualityAlgorithm()
    
    def name(self):
        return 'dem_quality'
    
    def displayName(self):
        return 'DEM Quality Analysis'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Analyze DEM/Flow Direction quality.
        
        - Find No Flow Cells: Identifies cells with undefined flow direction (sinks or flats).
        - Find Parallel Flow: Identifies areas with parallel flow, often indicating artifacts in flat areas.
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
                'Output Mask'
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
            
            feedback.pushInfo('Analyzing...')
            feedback.setProgress(20)
            
            if method_idx == 0: # No Flow
                result = router.find_no_flow_cells()
            else: # Parallel Flow
                result = router.find_parallel_flow()
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result)
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
