# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/position_algorithms.py
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterNumber,
    QgsProcessingException
)
from ...core.morphometry import MorphometryProcessor

class TPIAlgorithm(QgsProcessingAlgorithm):
    """Calculate Topographic Position Index (TPI)."""
    
    INPUT = 'INPUT'
    RADIUS = 'RADIUS'
    OUTPUT = 'OUTPUT'
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return TPIAlgorithm()
    
    def name(self):
        return 'tpi'
    
    def displayName(self):
        return 'Topographic Position Index (TPI)'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Calculate Topographic Position Index (TPI).
        
        TPI compares the elevation of each cell to the mean elevation
        of a specified neighborhood around that cell.
        
        Positive values: Ridges/Hills
        Negative values: Valleys/Depressions
        Near zero: Flat areas or constant slope
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                'Input DEM'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RADIUS,
                'Neighborhood radius (cells)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=1
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output TPI'
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
            
            feedback.pushInfo('Calculating TPI...')
            feedback.setProgress(30)
            
            tpi = processor.calculate_tpi(radius=radius)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, tpi)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
