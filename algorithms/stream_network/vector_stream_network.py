# -*- coding: utf-8 -*-
"""
algorithms/stream_network/vector_stream_network.py
Unified vector stream network tool
"""

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterEnum,
    QgsProcessingException,
    QgsFeature,
    QgsFeatureSink,
    QgsGeometry,
    QgsPointXY,
    QgsFields,
    QgsField,
    QgsWkbTypes
)
from qgis.PyQt.QtCore import QVariant
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter

class VectorStreamNetworkAlgorithm(QgsProcessingAlgorithm):
    """Unified vector stream network tool."""
    
    INPUT_STREAMS = 'INPUT_STREAMS'
    INPUT_DIR = 'INPUT_DIR'
    METHOD = 'METHOD'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'Raster Streams to Vector'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return VectorStreamNetworkAlgorithm()
    
    def name(self):
        return 'vector_stream_network'
    
    def displayName(self):
        return 'Vector Stream Network Analysis'
    
    def group(self):
        return 'Stream Network Analysis'
    
    def groupId(self):
        return 'stream_network'
    
    def shortHelpString(self):
        return """
        Perform vector operations on stream networks.
        
        - Raster Streams to Vector: Converts a raster stream network to a vector line layer, maintaining topology.
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_STREAMS,
                'Stream Raster'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DIR,
                'D8 Flow Direction'
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
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                'Output Vector Layer',
                type=QgsProcessing.TypeVectorLine
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            stream_layer = self.parameterAsRasterLayer(parameters, self.INPUT_STREAMS, context)
            dir_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DIR, context)
            method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
            
            if stream_layer is None or dir_layer is None:
                raise QgsProcessingException('Stream Raster and Flow Direction are required')
            
            feedback.pushInfo('Loading data...')
            dir_proc = DEMProcessor(dir_layer.source())
            stream_proc = DEMProcessor(stream_layer.source())
            
            if dir_proc.array.shape != stream_proc.array.shape:
                raise QgsProcessingException('Input rasters must have same dimensions')
                
            router = FlowRouter(
                dir_proc.array, 
                dir_proc.cellsize_x, 
                geotransform=dir_proc.geotransform
            )
            router.flow_dir = dir_proc.array
            streams = stream_proc.array
            
            feedback.pushInfo('Extracting stream segments...')
            feedback.setProgress(20)
            
            segments = router.extract_stream_segments(streams)
            
            feedback.pushInfo(f'Found {len(segments)} segments. Creating vector features...')
            feedback.setProgress(60)
            
            # Create fields
            fields = QgsFields()
            fields.append(QgsField('id', QVariant.Int))
            fields.append(QgsField('order', QVariant.Int))
            fields.append(QgsField('length', QVariant.Double))
            
            (sink, dest_id) = self.parameterAsSink(
                parameters,
                self.OUTPUT,
                context,
                fields,
                QgsWkbTypes.LineString,
                stream_layer.crs()
            )
            
            if sink is None:
                raise QgsProcessingException('Error creating output sink')
            
            # Add features - segments is list of (points, order) tuples
            for i, segment_data in enumerate(segments):
                if feedback.isCanceled():
                    break
                
                # Handle both old format (list) and new format (tuple)
                if isinstance(segment_data, tuple):
                    segment, order = segment_data
                else:
                    segment = segment_data
                    order = 1
                    
                feat = QgsFeature()
                feat.setFields(fields)
                
                # Create geometry
                points = [QgsPointXY(x, y) for x, y in segment]
                geom = QgsGeometry.fromPolylineXY(points)
                
                feat.setGeometry(geom)
                feat.setAttribute('id', i + 1)
                feat.setAttribute('order', order)
                feat.setAttribute('length', geom.length())
                
                sink.addFeature(feat, QgsFeatureSink.FastInsert)
                
            feedback.setProgress(100)
            
            return {self.OUTPUT: dest_id}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
