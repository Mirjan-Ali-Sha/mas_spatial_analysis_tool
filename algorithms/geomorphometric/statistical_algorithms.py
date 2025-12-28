# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/statistical_algorithms.py
Unified statistical filter tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingException
)
from ...core.dem_utils import DEMProcessor
from ...core.array_utils import focal_statistics
from osgeo import gdal

class StatisticalFilterAlgorithm(QgsProcessingAlgorithm):
    """Unified statistical filter tool (Mean, Median, Min, Max, etc.)."""
    
    INPUT = 'INPUT'
    STATISTIC = 'STATISTIC'
    RADIUS = 'RADIUS'
    OUTPUT = 'OUTPUT'
    
    STAT_OPTIONS = ['Mean', 'Median', 'Minimum', 'Maximum', 'Range', 'Standard Deviation', 'Sum']
    STAT_KEYS = ['mean', 'median', 'min', 'max', 'range', 'std', 'sum']
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return StatisticalFilterAlgorithm()
    
    def name(self):
        return 'statistical_filter'
    
    def displayName(self):
        return 'Statistical Filter'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Apply statistical filters to a raster using a moving window.
        
        Supported statistics:
        - Mean: Average value
        - Median: Median value (noise removal)
        - Minimum: Lowest value
        - Maximum: Highest value
        - Range: Max - Min
        - Standard Deviation: Local variability
        - Sum: Total value
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                'Input Raster'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.STATISTIC,
                'Statistic type',
                options=self.STAT_OPTIONS,
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RADIUS,
                'Filter radius (cells)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
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
            stat_idx = self.parameterAsEnum(parameters, self.STATISTIC, context)
            radius = self.parameterAsInt(parameters, self.RADIUS, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input raster')
            
            stat_key = self.STAT_KEYS[stat_idx]
            window_size = (radius * 2) + 1
            
            feedback.pushInfo('Loading raster...')
            processor = DEMProcessor(input_layer.source())
            
            feedback.pushInfo(f'Calculating {self.STAT_OPTIONS[stat_idx]} (Window: {window_size}x{window_size})...')
            feedback.setProgress(30)
            
            # Calculate statistics
            result = focal_statistics(
                processor.array,
                window_size=window_size,
                statistic=stat_key
            )
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, result)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
