# -*- coding: utf-8 -*-
"""
algorithms/hydrological/depression_algorithms.py
Unified depression handling tool
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
from ...core.flow_algorithms import FlowRouter
from ...core.hydro_utils import HydrologicalAnalyzer
from osgeo import gdal

class DepressionHandlingAlgorithm(QgsProcessingAlgorithm):
    """Unified depression handling tool (Fill, Breach)."""
    
    INPUT = 'INPUT'
    METHOD = 'METHOD'
    MAX_DEPTH = 'MAX_DEPTH'
    MAX_LENGTH = 'MAX_LENGTH'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'Fill Depressions (Priority Flood)', 
        'Breach Depressions (Least Cost)',
        'Fill Single Cell Pits',
        'Breach Single Cell Pits'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return DepressionHandlingAlgorithm()
    
    def name(self):
        return 'depression_handling'
    
    def displayName(self):
        return 'Depression Handling'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Handle depressions (sinks) in DEM.
        
        Methods:
        - Fill Depressions: Raises elevation of depressions to pour point level.
        - Breach Depressions: Lowers elevation of barriers to create flow path.
        - Fill Single Cell Pits: Fills only single-cell pits.
        - Breach Single Cell Pits: Breaches only single-cell pits.
        
        TIP: First run may be slower as algorithms compile. For faster 
        performance, run with a small DEM first to warm up the system.
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
                self.MAX_DEPTH,
                'Maximum breach depth (optional)',
                type=QgsProcessingParameterNumber.Double,
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_LENGTH,
                'Maximum breach length (optional)',
                type=QgsProcessingParameterNumber.Integer,
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output DEM'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
            method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
            max_depth = self.parameterAsDouble(parameters, self.MAX_DEPTH, context)
            max_length = self.parameterAsInt(parameters, self.MAX_LENGTH, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = DEMProcessor(input_layer.source())
            
            feedback.setProgress(20)
            
            if method_idx == 0:  # Fill
                feedback.pushInfo('Filling depressions (Priority Flood)...')
                router = FlowRouter(processor.array, processor.cellsize_x)
                result = router.fill_depressions()
                
            elif method_idx == 1:  # Breach
                feedback.pushInfo('Breaching depressions (Least Cost)...')
                hydro = HydrologicalAnalyzer(processor.array, processor.cellsize_x)
                result = hydro.breach_depressions_least_cost(
                    max_depth=max_depth if max_depth > 0 else None,
                    max_length=max_length if max_length > 0 else None
                )
            
            elif method_idx == 2: # Fill Single Cell
                feedback.pushInfo('Filling single cell pits...')
                router = FlowRouter(processor.array, processor.cellsize_x)
                result = router.fill_single_cell_pits()
                
            elif method_idx == 3: # Breach Single Cell
                feedback.pushInfo('Breaching single cell pits...')
                router = FlowRouter(processor.array, processor.cellsize_x)
                result = router.breach_single_cell_pits()
            
            else:
                result = processor.array
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            # Depression handling creates float values (epsilon filling / breaching)
            # We must force Float output even if input is Integer to preserve sub-pixel adjustments.
            processor.save_raster(output_path, result, dtype=gdal.GDT_Float32)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
