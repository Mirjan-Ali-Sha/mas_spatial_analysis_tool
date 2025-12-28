# -*- coding: utf-8 -*-
"""
algorithms/stream_network/valley_extraction.py
Unified valley extraction tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException
)
from ...core.hydro_utils import HydrologicalAnalyzer
from ...core.morphometry import MorphometryProcessor
from ...core.flow_algorithms import FlowRouter

class ValleyExtractionAlgorithm(QgsProcessingAlgorithm):
    """Unified valley extraction tool."""
    
    INPUT_DEM = 'INPUT_DEM'
    OUTPUT = 'OUTPUT'
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return ValleyExtractionAlgorithm()
    
    def name(self):
        return 'valley_extraction'
    
    def displayName(self):
        return 'Valley Extraction'
    
    def group(self):
        return 'Stream Network Analysis'
    
    def groupId(self):
        return 'stream_network'
    
    def shortHelpString(self):
        return """
        Calculate Valley Index to identify valley bottoms.
        
        Uses a combination of slope and flow accumulation:
        VI = (1 - Slope/MaxSlope) * log(Accumulation)
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM,
                'Input DEM'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Valley Index'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if dem_layer is None:
                raise QgsProcessingException('Input DEM is required')
            
            feedback.pushInfo('Loading DEM...')
            # We need multiple processors
            morph = MorphometryProcessor(dem_layer.source())
            router = FlowRouter(morph.array, morph.cellsize_x) # For accumulation
            hydro = HydrologicalAnalyzer(morph.array, morph.cellsize_x, morph.nodata)
            
            feedback.pushInfo('Calculating Slope...')
            feedback.setProgress(20)
            slope = morph.calculate_slope(units='degrees')
            
            feedback.pushInfo('Calculating Flow Accumulation...')
            feedback.setProgress(40)
            # We need flow direction first
            flow_dir = router.d8_flow_direction()
            flow_acc = router.d8_flow_accumulation(flow_dir)
            
            feedback.pushInfo('Calculating Valley Index...')
            feedback.setProgress(70)
            vi = hydro.calculate_valley_index(flow_acc, slope)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(90)
            
            morph.save_raster(output_path, vi)
            morph.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
