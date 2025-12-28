# -*- coding: utf-8 -*-
"""
algorithms/stream_network/stream_extraction.py
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterNumber,
    QgsProcessingException
)
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter
import numpy as np

class ExtractStreamsAlgorithm(QgsProcessingAlgorithm):
    """Extract stream network from flow accumulation."""
    
    INPUT = 'INPUT'
    THRESHOLD = 'THRESHOLD'
    OUTPUT = 'OUTPUT'
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return ExtractStreamsAlgorithm()
    
    def name(self):
        return 'extract_streams'
    
    def displayName(self):
        return 'Extract Streams'
    
    def group(self):
        return 'Stream Network Analysis'
    
    def groupId(self):
        return 'stream_network'
    
    def shortHelpString(self):
        return """
        Extract stream network from flow accumulation raster.
        
        Cells with flow accumulation greater than the threshold
        are considered streams.
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                'Input Flow Accumulation'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.THRESHOLD,
                'Accumulation Threshold',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=100.0,
                minValue=0.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Stream Raster'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
            threshold = self.parameterAsDouble(parameters, self.THRESHOLD, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input raster')
            
            feedback.pushInfo('Loading Flow Accumulation raster...')
            processor = DEMProcessor(input_layer.source())
            
            # Show the expression being used (standard raster algebra syntax)
            feedback.pushInfo(f'Extracting streams using expression: "flow_acc" >= {threshold}')
            feedback.setProgress(30)
            
            # Initialize router for the extract_streams method
            router = FlowRouter(processor.array, processor.cellsize_x)
            streams = router.extract_streams(processor.array, threshold)
            
            # Statistics
            stream_pixels = np.sum(streams == 1)
            total_valid = np.sum(~np.isnan(processor.array))
            feedback.pushInfo(f'Stream pixels: {stream_pixels:,} / {total_valid:,} ({100*stream_pixels/total_valid:.2f}%)')
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            # Save as Int32 with -1 as NoData (NOT 0, so 0 values are preserved as non-stream)
            from osgeo import gdal
            processor.save_raster(output_path, streams, dtype=gdal.GDT_Int32, nodata=-9999)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
