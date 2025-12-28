# -*- coding: utf-8 -*-
"""
algorithms/hydrological/hydro_enforcement.py
Unified hydro-enforcement tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException
)
from ...core.dem_utils import DEMProcessor
import numpy as np

class HydroEnforcementAlgorithm(QgsProcessingAlgorithm):
    """Unified hydro-enforcement tool."""
    
    INPUT_DEM = 'INPUT_DEM'
    INPUT_STREAMS = 'INPUT_STREAMS'
    DECREMENT = 'DECREMENT'
    OUTPUT = 'OUTPUT'
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return HydroEnforcementAlgorithm()
    
    def name(self):
        return 'hydro_enforcement'
    
    def displayName(self):
        return 'Hydro-Enforcement (Burn Streams)'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Enforce flow by burning stream network into DEM.
        
        Decreases the elevation of cells coinciding with the stream network by a specified amount.
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM,
                'Input DEM'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_STREAMS,
                'Stream Raster'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DECREMENT,
                'Decrement Amount (elevation units)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
                minValue=0.0
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
            dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
            stream_layer = self.parameterAsRasterLayer(parameters, self.INPUT_STREAMS, context)
            decrement = self.parameterAsDouble(parameters, self.DECREMENT, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if dem_layer is None or stream_layer is None:
                raise QgsProcessingException('Input DEM and Stream Raster are required')
            
            feedback.pushInfo('Loading data...')
            processor = DEMProcessor(dem_layer.source())
            
            # Load streams (assuming same extent/resolution for now)
            # In a real plugin, we should align them.
            # For now, assume user provides aligned rasters.
            
            # We need to read stream raster into array.
            # DEMProcessor doesn't have a generic load_other_raster method exposed easily,
            # but we can use gdal directly or add a helper.
            # Or just use FlowRouter's load_raster if we inherit?
            # Let's just use gdal here for simplicity or add a helper to DEMProcessor.
            
            # Actually, DEMProcessor has read_array but it reads from self.ds.
            # Let's use a temporary DEMProcessor for streams.
            stream_proc = DEMProcessor(stream_layer.source())
            streams = stream_proc.array
            stream_proc.close()
            
            if streams.shape != processor.array.shape:
                raise QgsProcessingException('DEM and Stream Raster must have same dimensions')
            
            feedback.pushInfo('Burning streams...')
            feedback.setProgress(50)
            
            burned_dem = processor.array.copy()
            
            # Where streams > 0, subtract decrement
            mask = (streams > 0)
            burned_dem[mask] -= decrement
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, burned_dem)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
