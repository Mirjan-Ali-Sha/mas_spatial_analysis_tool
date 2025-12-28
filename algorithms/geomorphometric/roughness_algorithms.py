# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/roughness_algorithms.py
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

class RoughnessAlgorithm(QgsProcessingAlgorithm):
    """Unified roughness analysis tool."""
    
    INPUT = 'INPUT'
    TYPE = 'TYPE'
    WINDOW_SIZE = 'WINDOW_SIZE'
    MAX_RADIUS = 'MAX_RADIUS'
    OUTPUT = 'OUTPUT'
    
    TYPES = [
        'Terrain Roughness (Max-Min)', 
        'Terrain Ruggedness Index (TRI)', 
        'Surface Area Ratio', 
        'Edge Density',
        'Multiscale Roughness (Magnitude)'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return RoughnessAlgorithm()
    
    def name(self):
        return 'roughness'
    
    def displayName(self):
        return 'Terrain Roughness'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Calculate various terrain roughness indices.
        
        Supported types:
        - Terrain Roughness: Difference between max and min elevation (Max-Min).
        - TRI: Terrain Ruggedness Index (Riley et al., 1999).
        - Surface Area Ratio: Ratio of true surface area to planimetric area.
        - Edge Density: Density of high-slope areas.
        - Multiscale Roughness: Maximum roughness magnitude across a range of scales.
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
                'Roughness Type',
                options=self.TYPES,
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.WINDOW_SIZE,
                'Window size (cells, odd) / Min Radius',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=3
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_RADIUS,
                'Max Radius (for Multiscale)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=3,
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Roughness'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
            type_idx = self.parameterAsEnum(parameters, self.TYPE, context)
            window_size = self.parameterAsInt(parameters, self.WINDOW_SIZE, context)
            max_radius = self.parameterAsInt(parameters, self.MAX_RADIUS, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            if window_size % 2 == 0:
                raise QgsProcessingException('Window size must be odd')
            
            feedback.pushInfo('Loading DEM...')
            processor = MorphometryProcessor(input_layer.source())
            
            selected_type = self.TYPES[type_idx]
            feedback.pushInfo(f'Calculating {selected_type}...')
            feedback.setProgress(30)
            
            if type_idx == 0: # Max-Min
                result = processor.calculate_roughness(window_size=window_size)
            elif type_idx == 1: # TRI
                result = processor.calculate_ruggedness_index(radius=window_size//2)
            elif type_idx == 2: # SAR
                result = processor.calculate_surface_area_ratio()
            elif type_idx == 3: # Edge Density
                result = processor.calculate_edge_density(window_size=window_size)
            elif type_idx == 4: # Multiscale
                min_r = window_size // 2
                result = processor.calculate_multiscale_roughness(min_radius=min_r, max_radius=max_radius)
            else:
                result = processor.calculate_roughness(window_size=window_size)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
