# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/visibility_algorithms.py
Unified visibility analysis tool
"""

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException,
    QgsFeature
)
from ...core.morphometry import MorphometryProcessor

class VisibilityAlgorithm(QgsProcessingAlgorithm):
    """Unified visibility analysis tool (Viewshed)."""
    
    INPUT_DEM = 'INPUT_DEM'
    OBSERVER_POINTS = 'OBSERVER_POINTS'
    OBSERVER_HEIGHT = 'OBSERVER_HEIGHT'
    TARGET_HEIGHT = 'TARGET_HEIGHT'
    OUTPUT = 'OUTPUT'
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return VisibilityAlgorithm()
    
    def name(self):
        return 'visibility_analysis'
    
    def displayName(self):
        return 'Visibility Analysis'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Calculate Viewshed (Visibility) from observer points.
        
        Determines which cells are visible from the observer location(s).
        Currently supports the first point in the observer layer.
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM,
                'Input DEM'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.OBSERVER_POINTS,
                'Observer Points',
                types=[QgsProcessing.TypeVectorPoint]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OBSERVER_HEIGHT,
                'Observer Height (m)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.8,
                minValue=0.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TARGET_HEIGHT,
                'Target Height (m)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Viewshed'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
            observer_source = self.parameterAsSource(parameters, self.OBSERVER_POINTS, context)
            obs_h = self.parameterAsDouble(parameters, self.OBSERVER_HEIGHT, context)
            target_h = self.parameterAsDouble(parameters, self.TARGET_HEIGHT, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if dem_layer is None or observer_source is None:
                raise QgsProcessingException('Invalid inputs')
            
            # Get first observer point
            features = observer_source.getFeatures()
            first_feature = next(features, None)
            
            if first_feature is None:
                raise QgsProcessingException('No observer points found')
                
            geom = first_feature.geometry()
            point = geom.asPoint()
            obs_x, obs_y = point.x(), point.y()
            
            feedback.pushInfo(f'Calculating viewshed from observer at ({obs_x}, {obs_y})...')
            feedback.pushInfo('Loading DEM...')
            
            processor = MorphometryProcessor(dem_layer.source())
            
            feedback.setProgress(20)
            
            result = processor.calculate_viewshed(
                obs_x, obs_y, 
                observer_height=obs_h, 
                target_height=target_h
            )
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result, nodata=-9999)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
