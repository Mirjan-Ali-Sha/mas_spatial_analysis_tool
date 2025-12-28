# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/directional_analysis.py
Unified directional analysis tool
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

class DirectionalAnalysisAlgorithm(QgsProcessingAlgorithm):
    """Unified directional analysis tool."""
    
    INPUT = 'INPUT'
    METHOD = 'METHOD'
    AZIMUTH = 'AZIMUTH'
    MAX_DIST = 'MAX_DIST'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'Directional Relief',
        'Wind Exposure (Winstral Sx)'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return DirectionalAnalysisAlgorithm()
    
    def name(self):
        return 'directional_analysis'
    
    def displayName(self):
        return 'Directional Analysis'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Calculate directional terrain attributes.
        
        - Directional Relief: Gradient component in a specific direction.
        - Wind Exposure: Maximum upwind slope (Winstral's Sx), indicating sheltering/exposure.
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
                self.AZIMUTH,
                'Azimuth (degrees)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=315.0,
                minValue=0.0,
                maxValue=360.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_DIST,
                'Maximum Search Distance (map units)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=100.0,
                minValue=0.0,
                optional=True # Only for Wind Exposure
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
            azimuth = self.parameterAsDouble(parameters, self.AZIMUTH, context)
            max_dist = self.parameterAsDouble(parameters, self.MAX_DIST, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = MorphometryProcessor(input_layer.source())
            
            feedback.pushInfo('Calculating...')
            feedback.setProgress(20)
            
            if method_idx == 0: # Directional Relief
                result = processor.calculate_directional_relief(azimuth)
            else: # Wind Exposure
                result = processor.calculate_wind_exposure(azimuth, max_dist)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
