# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/terrain_complexity.py
Unified terrain complexity tool
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

class TerrainComplexityAlgorithm(QgsProcessingAlgorithm):
    """Unified terrain complexity tool."""
    
    INPUT = 'INPUT'
    METHOD = 'METHOD'
    RADIUS = 'RADIUS'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'Terrain Ruggedness Index (TRI)'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return TerrainComplexityAlgorithm()
    
    def name(self):
        return 'terrain_complexity'
    
    def displayName(self):
        return 'Terrain Complexity'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Calculate terrain complexity indices.
        
        - TRI: Mean of absolute differences between center cell and neighbors.
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
                defaultValue=1,
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
            radius = self.parameterAsInt(parameters, self.RADIUS, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = MorphometryProcessor(input_layer.source())
            
            feedback.pushInfo('Calculating TRI...')
            feedback.setProgress(20)
            
            result = processor.calculate_ruggedness_index(radius=radius)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
