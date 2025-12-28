# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/relative_position.py
Unified relative topographic position tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException
)
from ...core.morphometry import MorphometryProcessor

class RelativePositionAlgorithm(QgsProcessingAlgorithm):
    """Unified relative topographic position tool."""
    
    INPUT = 'INPUT'
    METHOD = 'METHOD'
    RADIUS = 'RADIUS'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'Relative Topographic Position (Min-Max)', 
        'Difference from Mean Elevation', 
        'Deviation from Mean Elevation (Standardized)'
    ]
    METHOD_KEYS = ['minmax', 'diff', 'dev']
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return RelativePositionAlgorithm()
    
    def name(self):
        return 'relative_position'
    
    def displayName(self):
        return 'Relative Topographic Position'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Calculate relative topographic position indices.
        
        - RTP (Min-Max): (Elevation - Min) / (Max - Min)
        - Difference from Mean: Elevation - Mean
        - Deviation from Mean: (Elevation - Mean) / StdDev
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
            QgsProcessingParameterNumber(
                self.RADIUS,
                'Search Radius (cells)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=1
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
            radius = self.parameterAsInt(parameters, self.RADIUS, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = MorphometryProcessor(input_layer.source())
            
            method_key = self.METHOD_KEYS[method_idx]
            feedback.pushInfo(f'Calculating {self.METHOD_OPTIONS[method_idx]}...')
            feedback.setProgress(20)
            
            result = processor.calculate_relative_position(radius=radius, method=method_key)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
