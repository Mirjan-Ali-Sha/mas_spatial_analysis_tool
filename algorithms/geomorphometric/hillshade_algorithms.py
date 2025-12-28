# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/hillshade_algorithms.py
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingException
)
from ...core.morphometry import MorphometryProcessor
from osgeo import gdal

class HillshadeAlgorithm(QgsProcessingAlgorithm):
    """Unified hillshade analysis tool."""
    
    INPUT = 'INPUT'
    METHOD = 'METHOD'
    AZIMUTH = 'AZIMUTH'
    ALTITUDE = 'ALTITUDE'
    Z_FACTOR = 'Z_FACTOR'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'Hillshade',
        'Multidirectional Hillshade'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return HillshadeAlgorithm()
    
    def name(self):
        return 'hillshade'
    
    def displayName(self):
        return 'Hillshade Analysis'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Calculate hillshade (shaded relief).
        
        - Hillshade: Standard single-direction shading.
        - Multidirectional: Combines shading from 4 directions (225, 270, 315, 360).
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
                'Azimuth (for Standard Hillshade)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=315.0,
                minValue=0.0,
                maxValue=360.0,
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.ALTITUDE,
                'Altitude (light angle)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=45.0,
                minValue=0.0,
                maxValue=90.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.Z_FACTOR,
                'Z-factor',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Hillshade'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
            method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
            azimuth = self.parameterAsDouble(parameters, self.AZIMUTH, context)
            altitude = self.parameterAsDouble(parameters, self.ALTITUDE, context)
            z_factor = self.parameterAsDouble(parameters, self.Z_FACTOR, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = MorphometryProcessor(input_layer.source())
            
            feedback.pushInfo(f'Calculating {self.METHOD_OPTIONS[method_idx]}...')
            feedback.setProgress(30)
            
            if method_idx == 0: # Standard
                hillshade = processor.calculate_hillshade(
                    azimuth=azimuth,
                    altitude=altitude,
                    z_factor=z_factor
                )
            elif method_idx == 1: # Multidirectional
                hillshade = processor.calculate_multidirectional_hillshade(
                    altitude=altitude,
                    z_factor=z_factor
                )
            else:
                hillshade = processor.calculate_hillshade(
                    azimuth=azimuth,
                    altitude=altitude,
                    z_factor=z_factor
                )
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, hillshade, dtype=gdal.GDT_Byte, nodata=255)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
