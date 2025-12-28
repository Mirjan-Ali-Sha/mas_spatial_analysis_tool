# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/feature_detection.py
Unified feature detection tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterEnum,
    QgsProcessingException
)
from ...core.morphometry import MorphometryProcessor

class FeatureDetectionAlgorithm(QgsProcessingAlgorithm):
    """Unified feature detection tool (Peak, Valley, Saddle, Landforms)."""
    
    INPUT = 'INPUT'
    FEATURE_TYPE = 'FEATURE_TYPE'
    OUTPUT = 'OUTPUT'
    
    FEATURE_OPTIONS = [
        'Peak', 'Valley', 'Saddle', 
        'Ridge', 'Channel', 
        'Landform Classification (Pennock)'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return FeatureDetectionAlgorithm()
    
    def name(self):
        return 'feature_detection'
    
    def displayName(self):
        return 'Feature Detection'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Detect topographic features and classify landforms.
        
        Features:
        - Peak: Cell higher than all 8 neighbors.
        - Valley: Cell lower than all 8 neighbors.
        - Saddle: Cell with mixed high/low neighbors.
        - Ridge: Convex profile curvature.
        - Channel: Concave profile curvature.
        - Landform Classification (Pennock): 7-class classification based on slope and curvature.
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
                self.FEATURE_TYPE,
                'Feature Type',
                options=self.FEATURE_OPTIONS,
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Feature Raster'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
            feature_idx = self.parameterAsEnum(parameters, self.FEATURE_TYPE, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = MorphometryProcessor(input_layer.source())
            
            feature_type = self.FEATURE_OPTIONS[feature_idx]
            feedback.pushInfo(f'Detecting {feature_type}...')
            feedback.setProgress(30)
            
            if feature_idx == 5: # Landform Classification
                result = processor.classify_landforms_pennock()
            else:
                result = processor.detect_features(feature_type=feature_type.lower())
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result, nodata=-9999)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
