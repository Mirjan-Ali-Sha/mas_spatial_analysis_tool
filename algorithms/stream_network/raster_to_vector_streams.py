# -*- coding: utf-8 -*-
"""
algorithms/stream_network/raster_to_vector_streams.py
Raster to Vector Stream conversion with Gap Joining
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingException,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsFields,
    QgsField,
    QgsWkbTypes,
    QgsProcessing
)
from qgis.PyQt.QtCore import QVariant
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter

class RasterToVectorStreamsAlgorithm(QgsProcessingAlgorithm):
    """Convert raster streams to vector lines with gap joining."""
    
    INPUT_STREAMS = 'INPUT_STREAMS'
    INPUT_DIR = 'INPUT_DIR'
    GAP_THRESHOLD = 'GAP_THRESHOLD'
    UNIT = 'UNIT'
    OUTPUT = 'OUTPUT'
    
    UNIT_OPTIONS = ['Pixels', 'Map Units', 'Meters']
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return RasterToVectorStreamsAlgorithm()
    
    def name(self):
        return 'raster_to_vector_streams'
    
    def displayName(self):
        return 'Raster to Vector Streams (with Gap Filling)'
    
    def group(self):
        return 'Stream Network Analysis'
    
    def groupId(self):
        return 'stream_network'
    
    def shortHelpString(self):
        return """
        Converts a binary stream raster to vector lines, with optional gap filling.
        
        The tool can bridge gaps in the stream network by tracing downstream from endpoints
        and checking if they reconnect to the network within the specified threshold.
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
                'Flow Direction Raster'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.GAP_THRESHOLD,
                'Gap Threshold (0 to disable) - Endpoint to stream connection',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.UNIT,
                'Threshold Unit',
                options=self.UNIT_OPTIONS,
                defaultValue=1 # Map Units
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                'Output Vector Streams',
                type=QgsProcessing.TypeVectorLine
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            stream_layer = self.parameterAsRasterLayer(parameters, self.INPUT_STREAMS, context)
            dir_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DIR, context)
            gap_threshold = self.parameterAsDouble(parameters, self.GAP_THRESHOLD, context)
            unit_idx = self.parameterAsEnum(parameters, self.UNIT, context)
            
            if stream_layer is None or dir_layer is None:
                raise QgsProcessingException('Stream Raster and Flow Direction are required')
            
            feedback.pushInfo('Loading data...')
            dir_proc = DEMProcessor(dir_layer.source())
            stream_proc = DEMProcessor(stream_layer.source())
            
            if dir_proc.array.shape != stream_proc.array.shape:
                raise QgsProcessingException('Input rasters must have same dimensions')
            
            # Unit conversion
            # Pixels = 0, Map Units = 1, Meters = 2
            # We need threshold in Map Units for the recursive logic if using cellsize
            # The logic in FlowRouter assumes gap_threshold is in same units as cellsize/diag_dist
            
            final_threshold = gap_threshold
            
            if unit_idx == 0: # Pixels
                final_threshold = gap_threshold * dir_proc.cellsize_x
                feedback.pushInfo(f'Converted {gap_threshold} pixels to {final_threshold} map units')
            elif unit_idx == 2: # Meters
                # If CRS is Geographic (degrees), this is tricky. linearUnits typically returns meters for Projected.
                # If degrees, map units are degrees.
                crs = stream_layer.crs()
                if crs.isGeographic():
                    # Rough conversion: 1 degree approx 111320 meters
                    # This is very rough but standard for simple tools unless we use QgsDistanceArea
                    # For robust hydrological analysis, integration usually suggests projected CRS.
                    # Let's warn if geographic.
                    feedback.reportError("Warning: Using Meters with Geographic CRS is approximate.")
                    # 1 deg ~ 111km
                    final_threshold = gap_threshold / 111320.0
                else:
                    # Assume Map Units are Meters
                    final_threshold = gap_threshold
            
            router = FlowRouter(
                dir_proc.array, 
                dir_proc.cellsize_x, 
                geotransform=dir_proc.geotransform
            )
            router.flow_dir = dir_proc.array
            streams = stream_proc.array
            
            # Gap Joining
            if final_threshold > 0:
                feedback.pushInfo(f'Joining gaps with threshold {final_threshold}...')
                feedback.setProgress(20)
                # Need to implement logic in FlowRouter to join gaps without standard thresholds?
                # The user said "convert all pixels to line shape and also gap thresold ... no need of breakdown value"
                # This implies the input IS the stream raster, and we fill gaps IN the raster before vectorizing.
                
                streams = router.join_stream_gaps(streams, final_threshold)
            
            # Vectorization
            feedback.pushInfo('Extracting vector segments...')
            feedback.setProgress(50)
            
            segments = router.extract_stream_segments(streams)
            
            feedback.pushInfo(f'Found {len(segments)} segments. Saving...')
            
            # Create Sink
            fields = QgsFields()
            fields.append(QgsField('id', QVariant.Int))
            fields.append(QgsField('length', QVariant.Double))
            
            (sink, dest_id) = self.parameterAsSink(
                parameters, self.OUTPUT, context,
                fields, QgsWkbTypes.LineString, stream_layer.crs()
            )
            
            if sink is None:
                raise QgsProcessingException('Error creating output sink')
            
            count = 0
            for i, segment_data in enumerate(segments):
                if feedback.isCanceled(): break
                
                if isinstance(segment_data, tuple):
                    segment, order = segment_data
                else:
                    segment = segment_data
                
                points = [QgsPointXY(x, y) for x, y in segment]
                if len(points) < 2: continue
                
                geom = QgsGeometry.fromPolylineXY(points)
                feat = QgsFeature()
                feat.setFields(fields)
                feat.setGeometry(geom)
                feat.setAttribute('id', i + 1)
                feat.setAttribute('length', geom.length())
                
                sink.addFeature(feat)
                count += 1
                
            feedback.pushInfo(f'Created {count} features.')
            feedback.setProgress(100)
            
            return {self.OUTPUT: dest_id}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
