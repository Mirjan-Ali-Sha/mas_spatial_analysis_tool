# -*- coding: utf-8 -*-
"""
algorithms/hydrological/flow_indices.py
Unified flow indices tool (TWI, SPI)
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterEnum,
    QgsProcessingException
)
from ...core.morphometry import MorphometryProcessor
from ...core.flow_algorithms import FlowRouter
from ...core.hydro_utils import HydrologicalAnalyzer
from osgeo import gdal
import numpy as np

class FlowIndicesAlgorithm(QgsProcessingAlgorithm):
    """Unified flow indices tool (TWI, SPI)."""
    
    INPUT = 'INPUT'
    INDEX_TYPE = 'INDEX_TYPE'
    OUTPUT = 'OUTPUT'
    
    INDEX_OPTIONS = ['Topographic Wetness Index (TWI)', 'Stream Power Index (SPI)', 'Sediment Transport Index (STI)']
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return FlowIndicesAlgorithm()
    
    def name(self):
        return 'flow_indices'
    
    def displayName(self):
        return 'Flow Indices'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Calculate hydrological flow indices.
        
        Indices:
        - TWI: ln(a / tan(slope)). Indicates soil moisture.
        - SPI: a * tan(slope). Indicates erosive power.
        - STI: (a / 22.13)^0.6 * (sin(slope) / 0.0896)^1.3. Indicates erosion potential.
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
                self.INDEX_TYPE,
                'Index Type',
                options=self.INDEX_OPTIONS,
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Index'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
            index_idx = self.parameterAsEnum(parameters, self.INDEX_TYPE, context)
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
            
            # 3. Calculate Index
            feedback.pushInfo(f'Calculating {self.INDEX_OPTIONS[index_idx]}...')
            hydro = HydrologicalAnalyzer(dem_processor.array, dem_processor.cellsize_x)
            
            if index_idx == 0:  # TWI
                result = hydro.calculate_wetness_index(flow_acc, slope)
            elif index_idx == 1:  # SPI
                result = hydro.calculate_stream_power_index(flow_acc, slope)
            else:  # STI
                result = hydro.calculate_sediment_transport_index(flow_acc, slope)
            
            feedback.setProgress(80)
            
            feedback.pushInfo('Saving output...')
            dem_processor.save_raster(output_path, result)
            dem_processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
