# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/openness.py
Unified openness tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException
)
from ...core.morphometry import MorphometryProcessor

class OpennessAlgorithm(QgsProcessingAlgorithm):
    """Unified openness tool."""
    
    INPUT = 'INPUT'
    TYPE = 'TYPE'
    RADIUS = 'RADIUS'
    OUTPUT = 'OUTPUT'
    
    TYPE_OPTIONS = [
        'Positive Openness',
        'Negative Openness'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return OpennessAlgorithm()
    
    def name(self):
        return 'openness'
    
    def displayName(self):
        return 'Topographic Openness'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Calculate Topographic Openness (Yokoyama et al. 2002).
        
        - Positive Openness: Degree of openness above the surface (sky view).
        - Negative Openness: Degree of openness below the surface.
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
                self.TYPE,
                'Openness Type',
                options=self.TYPE_OPTIONS,
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RADIUS,
                'Search Radius (cells)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=10,
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
            type_idx = self.parameterAsEnum(parameters, self.TYPE, context)
            radius = self.parameterAsInt(parameters, self.RADIUS, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = MorphometryProcessor(input_layer.source())
            
            feedback.pushInfo('Calculating Openness...')
            feedback.setProgress(20)
            
            openness_type = 'positive' if type_idx == 0 else 'negative'
            result = processor.calculate_openness(radius, openness_type)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
