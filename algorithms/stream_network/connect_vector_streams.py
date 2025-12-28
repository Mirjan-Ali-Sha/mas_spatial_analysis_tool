# -*- coding: utf-8 -*-
"""
algorithms/stream_network/connect_vector_streams.py
Connect Vector Stream Network (Join Gaps)
"""

import math
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingException,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsWkbTypes,
    QgsProcessing,
    QgsSpatialIndex,
    QgsDistanceArea
)
from ...core.dem_utils import DEMProcessor

class ConnectVectorStreamsAlgorithm(QgsProcessingAlgorithm):
    """Connect vector streams by bridging gaps data."""
    
    INPUT_STREAMS = 'INPUT_STREAMS'
    FLOW_DIR = 'FLOW_DIR'
    GAP_THRESHOLD = 'GAP_THRESHOLD'
    GAP_UNIT = 'GAP_UNIT'
    REMOVE_SMALL = 'REMOVE_SMALL'
    SMALL_THRESHOLD = 'SMALL_THRESHOLD'
    SMALL_UNIT = 'SMALL_UNIT'
    OUTPUT = 'OUTPUT'
    
    UNIT_OPTIONS = ['Map Units', 'Meters']
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return ConnectVectorStreamsAlgorithm()
    
    def name(self):
        return 'connect_vector_streams'
    
    def displayName(self):
        return 'Connect Vector Streams (Join Gaps)'
    
    def group(self):
        return 'Stream Network Analysis'
    
    def groupId(self):
        return 'stream_network'
    
    def shortHelpString(self):
        return """
        Joins gaps between stream lines within a specified threshold.
        
        Optional: Can remove small stream segments before processing to reduce noise.
        Optional: Uses Flow Direction to prioritize downstream connections.
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_STREAMS,
                'Input Vector Streams',
                types=[QgsProcessing.TypeVectorLine]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.FLOW_DIR,
                'Flow Direction Raster (Optional)',
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.GAP_THRESHOLD,
                'Gap Threshold (Map Units)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
                minValue=0.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.GAP_UNIT,
                'Gap Threshold Unit',
                options=self.UNIT_OPTIONS,
                defaultValue=0 # Map Units
            )
        )
        
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.REMOVE_SMALL,
                'Remove Small Streams before connecting',
                defaultValue=False
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SMALL_THRESHOLD,
                'Small Stream Threshold (Map Units)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=100.0,
                minValue=0.0,
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.SMALL_UNIT,
                'Small Stream Threshold Unit',
                options=self.UNIT_OPTIONS,
                defaultValue=0 # Map Units
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                'Connected Streams',
                type=QgsProcessing.TypeVectorLine
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            source = self.parameterAsSource(parameters, self.INPUT_STREAMS, context)
            flow_dir_layer = self.parameterAsRasterLayer(parameters, self.FLOW_DIR, context)
            gap_threshold = self.parameterAsDouble(parameters, self.GAP_THRESHOLD, context)
            gap_unit = self.parameterAsEnum(parameters, self.GAP_UNIT, context)
            remove_small = self.parameterAsBool(parameters, self.REMOVE_SMALL, context)
            small_threshold = self.parameterAsDouble(parameters, self.SMALL_THRESHOLD, context)
            small_unit = self.parameterAsEnum(parameters, self.SMALL_UNIT, context)
            
            if source is None:
                raise QgsProcessingException('Input streams required')
            
            # Helper for unit conversion
            def get_map_units(value, unit_idx, crs):
                if unit_idx == 0: # Map Units
                    return value
                elif unit_idx == 1: # Meters
                    if crs.isGeographic():
                        # Rough conversion for thresholding
                        # 1 deg ~ 111320m
                        return value / 111320.0
                    else:
                        return value
                return value

            # Flow Direction process (for connectivity check only)
            flow_proc = None
            if flow_dir_layer:
                flow_proc = DEMProcessor(flow_dir_layer.source())

            crs = source.sourceCrs()
            
            # Convert Thresholds to Map Units for processing
            try:
                final_gap_threshold = get_map_units(gap_threshold, gap_unit, crs)
                final_small_threshold = get_map_units(small_threshold, small_unit, crs)
                
                if gap_unit == 1 and crs.isGeographic():
                     feedback.reportError("Warning: Using Meters with Geographic CRS is approximate.")
            except Exception as e:
                raise QgsProcessingException(str(e))
            
            feedback.pushInfo(f'Gap Threshold: {final_gap_threshold} map units')
            if remove_small:
                feedback.pushInfo(f'Small Threshold: {final_small_threshold} map units')

            
            feedback.pushInfo('Loading features...')
            
            # Load all features
            features = list(source.getFeatures())
            # Load all features
            features = list(source.getFeatures())
            
            # Optional Cleaning
            if remove_small:
                feedback.pushInfo(f'Removing streams < {final_small_threshold}...')
                cleaned_features = []
                for f in features:
                    if f.geometry() and f.geometry().length() >= final_small_threshold:
                        cleaned_features.append(f)
                features = cleaned_features
                feedback.pushInfo(f'{len(features)} streams remaining after cleaning.')
                
            # Create Spatial Index
            index = QgsSpatialIndex()
            feature_map = {}
            for f in features:
                index.addFeature(f)
                feature_map[f.id()] = f
                
            # Create output sink
            (sink, dest_id) = self.parameterAsSink(
                parameters, self.OUTPUT, context,
                source.fields(), QgsWkbTypes.LineString, crs
            )
            
            # Flow Direction helper
            flow_data = None
            gt = None
            cs_x = None
            cs_y = None
            if flow_dir_layer:
                proc = DEMProcessor(flow_dir_layer.source())
                flow_data = proc.array
                gt = proc.geotransform
                cs_x = proc.cellsize_x
                
            # Helper: Get endpoints
            def get_endpoints(geom):
                if geom.isMultipart():
                    lines = geom.asMultiPolyline()
                    pts = []
                    for l in lines:
                        if l:
                            pts.append(l[0])
                            pts.append(l[-1])
                    return pts
                else:
                    l = geom.asPolyline()
                    if l:
                        return [l[0], l[-1]]
                    return []

            # 1. Add all original features
            for f in features:
                sink.addFeature(f)
                
            # 2. Find connections
            feedback.pushInfo('Connecting gaps...')
            new_segments = []
            
            # To avoid cycles or double connections, track processed?
            # Simple approach: Iterate all endpoints, find nearest neighbor. 
            # If distance < threshold AND (optional) flow matches, create segment.
            
            count = 0 
            total = len(features)
            
            for f in features:
                if feedback.isCanceled(): break
                
                geom = f.geometry()
                endpoints = get_endpoints(geom)
                
                for pt in endpoints:
                    # Search nearest
                    # neighbor_ids = index.nearestNeighbor(pt, 2) # 2 because self is one
                    # Range search is better
                    neighbor_ids = index.nearestNeighbor(pt, 5) # Check a few
                    
                    best_target = None
                    min_dist = float('inf')
                    
                    for nid in neighbor_ids:
                        if nid == f.id(): continue
                        
                        target_feat = feature_map[nid]
                        target_geom = target_feat.geometry()
                        
                        # Distance from pt to geometry
                        # We want to connect to endpoints usually, but ANY point on line is OK?
                        # User said "join gaps between each line joints".
                        # Usually means endpoint to endpoint, or endpoint to line.
                        
                        dist = target_geom.distance(QgsGeometry.fromPointXY(pt))
                        
                        
                        if dist < final_gap_threshold and dist < min_dist:
                            # Candidate found
                            # Check Flow Direction
                            valid_flow = True
                            if flow_data is not None:
                                # Check if pt flows towards target
                                # Get flow direction at pt
                                try:
                                    c = int((pt.x() - gt[0]) / gt[1])
                                    r = int((pt.y() - gt[3]) / gt[5])
                                    if 0 <= r < flow_data.shape[0] and 0 <= c < flow_data.shape[1]:
                                        d = flow_data[r, c]
                                        # Convert d to angle (approx)
                                        # N=64, NE=128, E=1, SE=2, S=4, SW=8, W=16, NW=32
                                        # Just check if moving one step in 'd' gets closer to target?
                                        # Vector from pt to closest point on target
                                        closest_pt = target_geom.closestVertex(pt)[0] # Vertex or nearest point?
                                        # nearestPoint() returns QgsGeometry
                                        closest_geom_pt = target_geom.nearestPoint(QgsGeometry.fromPointXY(pt)).asPoint()
                                        
                                        dx = closest_geom_pt.x() - pt.x()
                                        dy = closest_geom_pt.y() - pt.y()
                                        angle = math.degrees(math.atan2(dy, dx)) # -180 to 180. 0 is East.
                                        
                                        # Map flow codes to angles
                                        # E=1 (0), SE=2 (-45), S=4 (-90), SW=8 (-135), W=16 (180), NW=32 (135), N=64 (90), NE=128 (45)
                                        code_map = {
                                            1: 0, 2: -45, 4: -90, 8: -135, 16: 180, 32: 135, 64: 90, 128: 45
                                        }
                                        
                                        if d in code_map:
                                            flow_angle = code_map[d]
                                            diff = abs(angle - flow_angle)
                                            if diff > 180: diff = 360 - diff
                                            
                                            if diff > 90: # Flows away
                                                valid_flow = False
                                except:
                                    pass # Ignore flow check errors
                            
                            if valid_flow:
                                min_dist = dist
                                best_target = target_geom.nearestPoint(QgsGeometry.fromPointXY(pt)).asPoint()

                    if best_target and min_dist > 0: # >0 to avoid self-snapped endpoints (cycles)
                         # Create segment
                         # Avoid duplicates?
                         seg_geom = QgsGeometry.fromPolylineXY([pt, best_target])
                         
                         # Check if this segment already exists (approx)?
                         # For now, just add. Duplicate clean up is another step.
                         
                         feat_new = QgsFeature()
                         feat_new.setGeometry(seg_geom)
                         # Set fields? Nulls likely.
                         new_segments.append(feat_new)
                
                count += 1
                if count % 100 == 0:
                    feedback.setProgress(int(count/total * 100))
            
            # Add new bridge segments
            feedback.pushInfo(f'Adding {len(new_segments)} connection segments...')
            for f in new_segments:
                sink.addFeature(f)
                
            return {self.OUTPUT: dest_id}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
