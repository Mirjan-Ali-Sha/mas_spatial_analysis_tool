# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/multiscale_analysis.py
Unified multiscale analysis tool
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
import numpy as np

class MultiscaleAnalysisAlgorithm(QgsProcessingAlgorithm):
    """Unified multiscale analysis tool."""
    
    INPUT = 'INPUT'
    METRIC = 'METRIC'
    MIN_RADIUS = 'MIN_RADIUS'
    MAX_RADIUS = 'MAX_RADIUS'
    STEP_RADIUS = 'STEP_RADIUS'
    OUTPUT = 'OUTPUT'
    
    METRIC_OPTIONS = [
        'Multiscale Roughness (Max Deviation)',
        'Multiscale Topographic Position (Max Deviation)'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return MultiscaleAnalysisAlgorithm()
    
    def name(self):
        return 'multiscale_analysis'
    
    def displayName(self):
        return 'Multiscale Analysis'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Calculate geomorphometric metrics across multiple scales (neighborhood sizes).
        
        Returns the maximum deviation or magnitude across the specified range of scales.
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
                self.METRIC,
                'Metric',
                options=self.METRIC_OPTIONS,
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_RADIUS,
                'Minimum Radius (cells)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=1
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_RADIUS,
                'Maximum Radius (cells)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=1
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.STEP_RADIUS,
                'Step Size (cells)',
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
            metric_idx = self.parameterAsEnum(parameters, self.METRIC, context)
            min_r = self.parameterAsInt(parameters, self.MIN_RADIUS, context)
            max_r = self.parameterAsInt(parameters, self.MAX_RADIUS, context)
            step_r = self.parameterAsInt(parameters, self.STEP_RADIUS, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = MorphometryProcessor(input_layer.source())
            
            # Initialize result with zeros
            max_magnitude = np.zeros_like(processor.array)
            
            radii = range(min_r, max_r + 1, step_r)
            total_steps = len(radii)
            
            for i, r in enumerate(radii):
                if feedback.isCanceled():
                    break
                    
                feedback.pushInfo(f'Processing radius {r} ({i+1}/{total_steps})...')
                feedback.setProgress(int((i / total_steps) * 100))
                
                if metric_idx == 0: # Roughness (using StdDev or similar?)
                    # Multiscale Roughness usually means finding the scale with max roughness, or max roughness across scales.
                    # Let's use TPI-like deviation or Roughness (StdDev of Slope? Or just StdDev of Elev?)
                    # WBT MultiscaleRoughness uses "std dev of normal vectors" or similar.
                    # Let's use our existing roughness (std dev of elevation) for simplicity, or TRI.
                    # Let's use TRI (Ruggedness).
                    val = processor.calculate_ruggedness_index(radius=r)
                    
                elif metric_idx == 1: # TPI
                    # TPI = Elev - Mean
                    # We want the magnitude (abs) or just the value?
                    # Usually "Max Deviation" means max absolute TPI.
                    tpi = processor.calculate_tpi(radius=r)
                    val = np.abs(tpi)
                
                # Update max
                max_magnitude = np.maximum(max_magnitude, val)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(100)
            
            processor.save_raster(output_path, max_magnitude)
            processor.close()
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
