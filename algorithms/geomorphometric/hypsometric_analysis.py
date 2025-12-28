# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/hypsometric_analysis.py
Unified hypsometric analysis tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterNumber,
    QgsProcessingException
)
from ...core.morphometry import MorphometryProcessor

class HypsometricAnalysisAlgorithm(QgsProcessingAlgorithm):
    """Unified hypsometric analysis tool."""
    
    INPUT = 'INPUT'
    WINDOW_SIZE = 'WINDOW_SIZE'
    OUTPUT = 'OUTPUT'
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return HypsometricAnalysisAlgorithm()
    
    def name(self):
        return 'hypsometric_analysis'
    
    def displayName(self):
        return 'Hypsometric Analysis'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Calculate Local Hypsometric Integral (HI).
        
        HI = (Mean - Min) / (Max - Min)
        
        Values range from 0 to 1:
        - High values (>0.6): Convex, youthful topography.
        - Low values (<0.4): Concave, mature topography.
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
                self.WINDOW_SIZE,
                'Window size (cells, must be odd)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=3
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output HI Raster'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
            window_size = self.parameterAsInt(parameters, self.WINDOW_SIZE, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            if window_size % 2 == 0:
                raise QgsProcessingException('Window size must be odd')
            
            feedback.pushInfo('Loading DEM...')
            processor = MorphometryProcessor(input_layer.source())
            
            feedback.pushInfo('Calculating Hypsometric Integral...')
            feedback.setProgress(30)
            
            result = processor.calculate_hypsometric_integral(window_size=window_size)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
