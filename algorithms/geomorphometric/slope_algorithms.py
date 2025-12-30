# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/slope_algorithms.py
"""

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingException
)
import os
from ...core.morphometry import MorphometryProcessor


class SlopeAlgorithm(QgsProcessingAlgorithm):
    """Unified slope analysis tool (Slope, StdDev of Slope)."""
    
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    METHOD = 'METHOD'
    UNITS = 'UNITS'
    Z_FACTOR = 'Z_FACTOR'
    WINDOW_SIZE = 'WINDOW_SIZE'
    
    METHOD_OPTIONS = [
        'Slope',
        'Standard Deviation of Slope'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return SlopeAlgorithm()
    
    def name(self):
        return 'slope'
    
    def displayName(self):
        return 'Slope Analysis'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Calculate slope and related indices.
        
        - Slope: Steepness of terrain (Degrees or Percent).
        - Standard Deviation of Slope: Variability of slope within a window.
        """
    
    def initAlgorithm(self, config=None):
        # Input DEM
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                'Input DEM',
                defaultValue=None
            )
        )
        
        # Method
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                'Method',
                options=self.METHOD_OPTIONS,
                defaultValue=0
            )
        )
        
        # Units (Only for Slope)
        self.addParameter(
            QgsProcessingParameterEnum(
                self.UNITS,
                'Slope units (for Slope method)',
                options=['Degrees', 'Percent'],
                defaultValue=0,
                optional=True
            )
        )
        
        # Z-factor
        self.addParameter(
            QgsProcessingParameterNumber(
                self.Z_FACTOR,
                'Z-factor (vertical exaggeration)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                minValue=0.001
            )
        )
        
        # Window Size (Only for StdDev)
        self.addParameter(
            QgsProcessingParameterNumber(
                self.WINDOW_SIZE,
                'Window Size (for StdDev)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=3,
                optional=True
            )
        )
        
        # Output
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output raster'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        """Process algorithm."""
        # Get parameters
        input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
        units_idx = self.parameterAsEnum(parameters, self.UNITS, context)
        z_factor = self.parameterAsDouble(parameters, self.Z_FACTOR, context)
        window_size = self.parameterAsInt(parameters, self.WINDOW_SIZE, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        
        if input_layer is None:
            raise QgsProcessingException('Invalid input DEM layer')
        
        # Convert units
        units = 'degrees' if units_idx == 0 else 'percent'
        
        feedback.pushInfo('Loading DEM...')
        dem_path = input_layer.source()
        
        try:
            # Initialize Morphometry processor
            processor = MorphometryProcessor(dem_path)
            
            feedback.pushInfo(f'Calculating {self.METHOD_OPTIONS[method_idx]}...')
            feedback.setProgress(30)
            
            if method_idx == 0: # Slope
                result = processor.calculate_slope(units=units, z_factor=z_factor)
            elif method_idx == 1: # StdDev Slope
                if window_size % 2 == 0:
                    raise QgsProcessingException('Window size must be odd')
                result = processor.calculate_slope_std_dev(window_size=window_size)
            else:
                result = processor.calculate_slope(units=units, z_factor=z_factor)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(70)
            
            # Save result
            processor.save_raster(output_path, result)
            processor.close()
            
            feedback.setProgress(80)
            
            # Apply Slope symbology (only for Slope method, not StdDev)
            if method_idx == 0:
                feedback.pushInfo('Applying Slope symbology...')
                try:
                    from ...core.symbology_utils import apply_slope_symbology
                    from qgis.core import QgsRasterLayer, QgsProject
                    
                    # Check if QGIS will load this layer (user checked "Open output file")
                    will_load = context.willLoadLayerOnCompletion(output_path)
                    
                    if will_load:
                        feedback.pushInfo('Loading output with symbology...')
                        
                        # Create the layer ourselves
                        import os
                        layer_name = os.path.splitext(os.path.basename(output_path))[0]
                        styled_layer = QgsRasterLayer(output_path, layer_name)
                        
                        if styled_layer.isValid():
                            # Apply symbology
                            apply_slope_symbology(styled_layer)
                            feedback.pushInfo('Slope symbology applied (green to red gradient)')
                            
                            # Add to project
                            QgsProject.instance().addMapLayer(styled_layer)
                            feedback.pushInfo('Styled layer added to project')
                            
                            # Tell QGIS NOT to load this layer again
                            context.setLayersToLoadOnCompletion({})
                        else:
                            feedback.pushInfo('Warning: Could not load styled layer')
                            
                except Exception as e:
                    feedback.pushInfo(f'Note: Could not apply symbology: {e}')
            
            feedback.pushInfo('Calculation complete!')
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error processing slope: {str(e)}')

class AspectAlgorithm(QgsProcessingAlgorithm):
    """Unified aspect analysis tool (Aspect, Circular Variance, Relative Aspect)."""
    
    INPUT = 'INPUT'
    METHOD = 'METHOD'
    WINDOW_SIZE = 'WINDOW_SIZE'
    AZIMUTH = 'AZIMUTH'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'Aspect',
        'Circular Variance of Aspect',
        'Relative Aspect'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return AspectAlgorithm()
    
    def name(self):
        return 'aspect'
    
    def displayName(self):
        return 'Aspect Analysis'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Calculate aspect and related indices.
        
        - Aspect: Slope direction (0-360 degrees).
        - Circular Variance: Measure of how variable the aspect is within a window (0-1).
        - Relative Aspect: Angular difference between aspect and a target azimuth (0-180).
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
                self.WINDOW_SIZE,
                'Window Size (for Circular Variance)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=3,
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.AZIMUTH,
                'Target Azimuth (for Relative Aspect)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
                maxValue=360.0,
                optional=True
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
            window_size = self.parameterAsInt(parameters, self.WINDOW_SIZE, context)
            azimuth = self.parameterAsDouble(parameters, self.AZIMUTH, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = MorphometryProcessor(input_layer.source())
            
            feedback.pushInfo(f'Calculating {self.METHOD_OPTIONS[method_idx]}...')
            feedback.setProgress(30)
            
            if method_idx == 0: # Aspect
                result = processor.calculate_aspect()
            elif method_idx == 1: # Circular Variance
                if window_size % 2 == 0:
                    raise QgsProcessingException('Window size must be odd')
                result = processor.calculate_circular_variance(window_size)
            elif method_idx == 2: # Relative Aspect
                result = processor.calculate_relative_aspect(azimuth)
            else:
                result = processor.calculate_aspect()
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(70)
            
            processor.save_raster(output_path, result)
            processor.close()
            
            feedback.setProgress(80)
            
            # Apply Aspect symbology (only for Aspect method)
            if method_idx == 0:
                feedback.pushInfo('Applying Aspect symbology...')
                try:
                    from ...core.symbology_utils import apply_aspect_symbology
                    from qgis.core import QgsRasterLayer, QgsProject
                    
                    # Check if QGIS will load this layer (user checked "Open output file")
                    will_load = context.willLoadLayerOnCompletion(output_path)
                    
                    if will_load:
                        feedback.pushInfo('Loading output with symbology...')
                        
                        # Create the layer ourselves
                        import os
                        layer_name = os.path.splitext(os.path.basename(output_path))[0]
                        styled_layer = QgsRasterLayer(output_path, layer_name)
                        
                        if styled_layer.isValid():
                            # Apply symbology
                            apply_aspect_symbology(styled_layer)
                            feedback.pushInfo('Aspect symbology applied (directional colors)')
                            
                            # Add to project
                            QgsProject.instance().addMapLayer(styled_layer)
                            feedback.pushInfo('Styled layer added to project')
                            
                            # Tell QGIS NOT to load this layer again
                            context.setLayersToLoadOnCompletion({})
                        else:
                            feedback.pushInfo('Warning: Could not load styled layer')
                            
                except Exception as e:
                    feedback.pushInfo(f'Note: Could not apply symbology: {e}')
            
            feedback.pushInfo('Calculation complete!')
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error processing aspect: {str(e)}')

