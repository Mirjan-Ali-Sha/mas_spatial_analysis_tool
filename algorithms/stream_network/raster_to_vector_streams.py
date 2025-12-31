# -*- coding: utf-8 -*-
"""
algorithms/stream_network/raster_to_vector_streams.py
Raster to Vector Streams (with Gap Filling)

Same functionality as Stream to Feature but with:
1. Gap filling options
2. Short stream removal
3. Endpoint snapping for proper network topology
"""

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingException,
    QgsFeature,
    QgsFeatureSink,
    QgsGeometry,
    QgsPointXY,
    QgsFields,
    QgsField,
    QgsWkbTypes,
    QgsSpatialIndex,
    QgsRectangle
)
from qgis.PyQt.QtCore import QVariant
import numpy as np
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter
import math


class RasterToVectorStreamsAlgorithm(QgsProcessingAlgorithm):
    """Convert raster streams to vector with gap filling and connected topology."""
    
    INPUT_STREAMS = 'INPUT_STREAMS'
    INPUT_DIR = 'INPUT_DIR'
    INPUT_ACC = 'INPUT_ACC'
    GAP_THRESHOLD = 'GAP_THRESHOLD'
    MIN_LENGTH = 'MIN_LENGTH'
    REMOVE_ISOLATED = 'REMOVE_ISOLATED'
    SNAP_DISTANCE = 'SNAP_DISTANCE'
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
        Convert raster stream network to vector lines with proper topology.
        
        Same base functionality as 'Stream to Feature' but with:
        - Optional gap filling (flow direction based)
        - Short stream removal
        - Endpoint snapping for connected network topology
        
        Parameters:
        - Gap Threshold: Maximum distance to bridge gaps (0 = no gap filling)
        - Min Stream Length: Remove streams shorter than this (0 = keep all)
        - Snap Distance: Distance to snap nearby endpoints for connectivity
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_STREAMS,
                'Stream Raster (binary or stream order)'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DIR,
                'D8 Flow Direction Raster'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_ACC,
                'Flow Accumulation Raster (optional, for isolated removal)',
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.GAP_THRESHOLD,
                'Gap Threshold (0 = no gap filling)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0,
                minValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_LENGTH,
                'Minimum Stream Length (0 = keep all)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0,
                minValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.REMOVE_ISOLATED,
                'Remove isolated streams (requires Flow Accumulation)',
                defaultValue=False
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SNAP_DISTANCE,
                'Endpoint Snap Distance (for connectivity, 0 = 1.5x cell size)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0,
                minValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.UNIT,
                'Distance Unit',
                options=self.UNIT_OPTIONS,
                defaultValue=1  # Map Units
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
            # Get parameters
            stream_layer = self.parameterAsRasterLayer(parameters, self.INPUT_STREAMS, context)
            dir_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DIR, context)
            acc_layer = self.parameterAsRasterLayer(parameters, self.INPUT_ACC, context)
            gap_threshold = self.parameterAsDouble(parameters, self.GAP_THRESHOLD, context)
            min_length = self.parameterAsDouble(parameters, self.MIN_LENGTH, context)
            remove_isolated = self.parameterAsBool(parameters, self.REMOVE_ISOLATED, context)
            snap_distance = self.parameterAsDouble(parameters, self.SNAP_DISTANCE, context)
            unit_idx = self.parameterAsEnum(parameters, self.UNIT, context)
            
            if stream_layer is None or dir_layer is None:
                raise QgsProcessingException('Stream Raster and Flow Direction are required')
            
            # Load data
            feedback.pushInfo('Loading data...')
            dir_proc = DEMProcessor(dir_layer.source())
            stream_proc = DEMProcessor(stream_layer.source())
            
            if dir_proc.array.shape != stream_proc.array.shape:
                raise QgsProcessingException('Input rasters must have same dimensions')
            
            acc_array = None
            if acc_layer:
                acc_proc = DEMProcessor(acc_layer.source())
                acc_array = acc_proc.array
                feedback.pushInfo('Flow Accumulation loaded')
            
            # Unit conversion
            cellsize = dir_proc.cellsize_x
            crs = stream_layer.crs()
            
            if unit_idx == 0:  # Pixels
                gap_threshold_map = gap_threshold * cellsize
                min_length_map = min_length * cellsize
                snap_distance_map = snap_distance * cellsize if snap_distance > 0 else cellsize * 1.5
            elif unit_idx == 1:  # Map Units
                gap_threshold_map = gap_threshold
                min_length_map = min_length
                snap_distance_map = snap_distance if snap_distance > 0 else cellsize * 1.5
            else:  # Meters
                if crs.isGeographic():
                    gap_threshold_map = gap_threshold / 111320.0
                    min_length_map = min_length / 111320.0
                    snap_distance_map = snap_distance / 111320.0 if snap_distance > 0 else cellsize * 1.5
                    feedback.reportError("Warning: Using Meters with Geographic CRS is approximate.")
                else:
                    gap_threshold_map = gap_threshold
                    min_length_map = min_length
                    snap_distance_map = snap_distance if snap_distance > 0 else cellsize * 1.5
            
            # Create router
            router = FlowRouter(
                dir_proc.array,
                dir_proc.cellsize_x,
                geotransform=dir_proc.geotransform
            )
            router.flow_dir = dir_proc.array
            streams = stream_proc.array.copy()
            
            original_count = np.sum(streams > 0)
            feedback.pushInfo(f'Original stream cells: {original_count}')
            
            # Step 1: Gap filling (optional)
            if gap_threshold_map > 0:
                feedback.pushInfo('Step 1: Flow-direction gap filling...')
                feedback.setProgress(10)
                streams = router.join_stream_gaps(streams, gap_threshold_map)
                gap_count = np.sum(streams > 0)
                feedback.pushInfo(f'After gap filling: {gap_count} cells (+{gap_count - original_count} added)')
            
            # Step 2: Remove short streams (optional)
            if min_length_map > 0:
                feedback.pushInfo('Step 2: Removing short streams...')
                feedback.setProgress(25)
                streams = router.remove_short_streams(streams, min_length_map)
                short_count = np.sum(streams > 0)
                feedback.pushInfo(f'After removing short streams: {short_count} cells')
            
            # Step 3: Remove isolated streams (optional)
            if remove_isolated and acc_array is not None:
                feedback.pushInfo('Step 3: Removing isolated streams...')
                feedback.setProgress(35)
                streams = router.remove_isolated_streams(streams, acc_array)
                iso_count = np.sum(streams > 0)
                feedback.pushInfo(f'After removing isolated streams: {iso_count} cells')
            
            # Step 4: Extract stream segments (same as Stream to Feature)
            feedback.pushInfo('Step 4: Extracting stream segments...')
            feedback.setProgress(45)
            
            segments = router.extract_stream_segments(streams)
            feedback.pushInfo(f'Extracted {len(segments)} segments')
            
            # Step 5: Snap endpoints for connectivity (using spatial index for performance)
            feedback.pushInfo('Step 5: Snapping endpoints for connectivity...')
            feedback.setProgress(60)
            
            segments = self._snap_endpoints(segments, snap_distance_map, feedback)
            feedback.pushInfo(f'After snapping: {len(segments)} segments')
            
            # Step 6: Create output
            feedback.pushInfo('Step 6: Creating vector features...')
            feedback.setProgress(80)
            
            fields = QgsFields()
            fields.append(QgsField('id', QVariant.Int))
            fields.append(QgsField('order', QVariant.Int))
            fields.append(QgsField('length', QVariant.Double))
            
            (sink, dest_id) = self.parameterAsSink(
                parameters, self.OUTPUT, context,
                fields, QgsWkbTypes.LineString, stream_layer.crs()
            )
            
            if sink is None:
                raise QgsProcessingException('Error creating output')
            
            count = 0
            for i, segment_data in enumerate(segments):
                if feedback.isCanceled():
                    break
                
                if isinstance(segment_data, tuple):
                    pts, order = segment_data
                else:
                    pts = segment_data
                    order = 1
                
                if len(pts) < 2:
                    continue
                
                points = [QgsPointXY(x, y) for x, y in pts]
                geom = QgsGeometry.fromPolylineXY(points)
                
                # Skip very short segments
                if min_length_map > 0 and geom.length() < min_length_map * 0.1:
                    continue
                
                feat = QgsFeature()
                feat.setFields(fields)
                feat.setGeometry(geom)
                feat.setAttribute('id', count + 1)
                feat.setAttribute('order', order)
                feat.setAttribute('length', geom.length())
                
                sink.addFeature(feat, QgsFeatureSink.FastInsert)
                count += 1
            
            feedback.pushInfo(f'Created {count} vector features')
            feedback.setProgress(100)
            
            return {self.OUTPUT: dest_id}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
    
    def _snap_endpoints(self, segments, snap_dist, feedback):
        """Snap nearby endpoints together for network connectivity.
        
        Uses spatial index for O(n log n) performance.
        """
        if not segments or snap_dist <= 0:
            return segments
        
        # Extract all endpoints with their segment index and type
        endpoints = []  # (x, y, seg_idx, end_type)
        
        for i, seg_data in enumerate(segments):
            pts = seg_data[0] if isinstance(seg_data, tuple) else seg_data
            if len(pts) >= 2:
                endpoints.append((pts[0][0], pts[0][1], i, 'start'))
                endpoints.append((pts[-1][0], pts[-1][1], i, 'end'))
        
        if len(endpoints) < 2:
            return segments
        
        # Build spatial index
        point_features = []
        for idx, (x, y, seg_idx, end_type) in enumerate(endpoints):
            feat = QgsFeature(idx)
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
            point_features.append(feat)
        
        spatial_idx = QgsSpatialIndex()
        spatial_idx.addFeatures(point_features)
        
        # Find snap groups (endpoints that should share the same coordinate)
        snap_groups = []  # List of lists of endpoint indices
        used = set()
        
        for idx, (x, y, seg_idx, end_type) in enumerate(endpoints):
            if idx in used:
                continue
            
            # Find nearby endpoints
            rect = QgsRectangle(x - snap_dist, y - snap_dist, x + snap_dist, y + snap_dist)
            nearby = spatial_idx.intersects(rect)
            
            group = []
            for other_idx in nearby:
                if other_idx in used:
                    continue
                ox, oy, _, _ = endpoints[other_idx]
                dist = math.sqrt((x - ox)**2 + (y - oy)**2)
                if dist < snap_dist:
                    group.append(other_idx)
                    used.add(other_idx)
            
            if len(group) > 1:
                snap_groups.append(group)
        
        # Convert segments to mutable lists
        new_segments = []
        for seg_data in segments:
            if isinstance(seg_data, tuple):
                pts, order = seg_data
                new_segments.append([list(pts), order])
            else:
                new_segments.append([list(seg_data), 1])
        
        # Apply snapping - use centroid of each group
        for group in snap_groups:
            # Calculate centroid
            sum_x, sum_y = 0, 0
            for idx in group:
                x, y, _, _ = endpoints[idx]
                sum_x += x
                sum_y += y
            cx = sum_x / len(group)
            cy = sum_y / len(group)
            
            # Update all endpoints in group to centroid
            for idx in group:
                _, _, seg_idx, end_type = endpoints[idx]
                if end_type == 'start':
                    new_segments[seg_idx][0][0] = (cx, cy)
                else:
                    new_segments[seg_idx][0][-1] = (cx, cy)
        
        # Convert back to tuple format
        return [(pts, order) for pts, order in new_segments]
