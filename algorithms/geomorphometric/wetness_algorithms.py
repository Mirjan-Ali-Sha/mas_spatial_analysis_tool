# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/wetness_algorithms.py
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException
)
from ...core.morphometry import MorphometryProcessor
from ...core.flow_algorithms import FlowRouter
from ...core.hydro_utils import HydrologicalAnalyzer
from osgeo import gdal

class WetnessIndexAlgorithm(QgsProcessingAlgorithm):
    """Calculate Topographic Wetness Index (TWI)."""
    
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return WetnessIndexAlgorithm()
    
    def name(self):
        return 'wetness_index'
    
    def displayName(self):
        return 'Topographic Wetness Index (TWI)'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Calculate Topographic Wetness Index (TWI).
        
        TWI = ln(a / tan(slope))
        where a = specific catchment area
        
        High values indicate wet areas (channels, valleys).
        Low values indicate dry areas (ridges).
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                'Input DEM'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output TWI'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            dem_processor = MorphometryProcessor(input_layer.source())
            
            feedback.setProgress(10)
            
            # 1. Calculate Slope
            feedback.pushInfo('Calculating slope...')
            slope = dem_processor.calculate_slope(units='radians')
            
            feedback.setProgress(30)
            
            # 2. Calculate Flow Accumulation
            feedback.pushInfo('Calculating flow accumulation...')
            router = FlowRouter(dem_processor.array, dem_processor.cellsize_x)
            flow_dir = router.d8_flow_direction()
            flow_acc = router.d8_flow_accumulation(flow_dir)
            
            feedback.setProgress(60)
            
            # 3. Calculate TWI
            feedback.pushInfo('Calculating TWI...')
            hydro = HydrologicalAnalyzer(dem_processor.array, dem_processor.cellsize_x)
            twi = hydro.calculate_wetness_index(flow_acc, slope)
            
            feedback.setProgress(80)
            
            feedback.pushInfo('Saving output...')
            dem_processor.save_raster(output_path, twi)
            dem_processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
