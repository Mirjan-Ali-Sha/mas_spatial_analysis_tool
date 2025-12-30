# -*- coding: utf-8 -*-
"""
algorithms/stream_network/connect_vector_streams.py
Connect Vector Stream Network (Join Gaps) - Restructured Algorithm

New approach:
1. Extend all line endpoints by threshold distance
2. Find and break lines at intersection points
3. Delete very small line segments (optional, enabled by default)
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
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
    QgsFeatureRequest
)
import math


class ConnectVectorStreamsAlgorithm(QgsProcessingAlgorithm):
    """Connect vector streams by extending lines, breaking at crossings, and cleaning."""
    
    INPUT_STREAMS = 'INPUT_STREAMS'
    EXTEND_DISTANCE = 'EXTEND_DISTANCE'
    DISTANCE_UNIT = 'DISTANCE_UNIT'
    DELETE_SMALL = 'DELETE_SMALL'
    SMALL_THRESHOLD = 'SMALL_THRESHOLD'
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
        Connects gaps between stream lines using an extend-break-clean approach.
        
        Algorithm steps:
        1. Extend all line endpoints by the specified distance
        2. Find intersection points where extended lines cross
        3. Break lines at intersection points to create proper junctions
        4. Optionally delete very small line segments (cleaning step)
        
        Parameters:
        - Extend Distance: How far to extend line endpoints
        - Delete Small Lines: Remove segments shorter than threshold (recommended)
        - Small Line Threshold: Minimum length to keep (default: half of extend distance)
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
            QgsProcessingParameterNumber(
                self.EXTEND_DISTANCE,
                'Extend Distance (threshold for gap joining)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
                minValue=0.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DISTANCE_UNIT,
                'Distance Unit',
                options=self.UNIT_OPTIONS,
                defaultValue=0  # Map Units
            )
        )
        
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.DELETE_SMALL,
                'Delete small line segments after processing',
                defaultValue=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SMALL_THRESHOLD,
                'Minimum segment length to keep (0 = use half of extend distance)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
                optional=True
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
            extend_dist = self.parameterAsDouble(parameters, self.EXTEND_DISTANCE, context)
            unit_idx = self.parameterAsEnum(parameters, self.DISTANCE_UNIT, context)
            delete_small = self.parameterAsBool(parameters, self.DELETE_SMALL, context)
            small_threshold = self.parameterAsDouble(parameters, self.SMALL_THRESHOLD, context)
            
            if source is None:
                raise QgsProcessingException('Input streams required')
            
            crs = source.sourceCrs()
            
            # Unit conversion
            if unit_idx == 1 and crs.isGeographic():
                # Convert meters to degrees (approximate)
                extend_dist = extend_dist / 111320.0
                small_threshold = small_threshold / 111320.0
                feedback.reportError("Warning: Using Meters with Geographic CRS is approximate.")
            
            # Default small threshold to half of extend distance
            if small_threshold <= 0:
                small_threshold = extend_dist / 2.0
            
            feedback.pushInfo(f'Extend distance: {extend_dist} map units')
            feedback.pushInfo(f'Small threshold: {small_threshold} map units')
            
            # Create output sink
            (sink, dest_id) = self.parameterAsSink(
                parameters, self.OUTPUT, context,
                source.fields(), QgsWkbTypes.LineString, crs
            )
            
            if sink is None:
                raise QgsProcessingException('Failed to create output')
            
            # Step 1: Load all features and extend their endpoints
            feedback.pushInfo('Step 1: Extending line endpoints...')
            features = list(source.getFeatures())
            total = len(features)
            
            extended_geometries = []
            
            for i, feat in enumerate(features):
                if feedback.isCanceled():
                    break
                
                geom = feat.geometry()
                if geom.isNull() or geom.isEmpty():
                    continue
                
                extended_geom = self._extend_line_both_ends(geom, extend_dist)
                extended_geometries.append({
                    'original_feat': feat,
                    'extended_geom': extended_geom
                })
                
                feedback.setProgress(int((i / total) * 25))
            
            feedback.pushInfo(f'Extended {len(extended_geometries)} lines')
            
            # Step 2: Find all intersection points
            feedback.pushInfo('Step 2: Finding intersection points...')
            all_intersections = []
            
            for i in range(len(extended_geometries)):
                if feedback.isCanceled():
                    break
                
                geom_i = extended_geometries[i]['extended_geom']
                
                for j in range(i + 1, len(extended_geometries)):
                    geom_j = extended_geometries[j]['extended_geom']
                    
                    if geom_i.intersects(geom_j):
                        intersection = geom_i.intersection(geom_j)
                        
                        if not intersection.isNull() and not intersection.isEmpty():
                            # Extract points from intersection
                            pts = self._extract_points_from_geometry(intersection)
                            for pt in pts:
                                all_intersections.append({
                                    'point': pt,
                                    'line_indices': [i, j]
                                })
                
                feedback.setProgress(25 + int((i / len(extended_geometries)) * 25))
            
            feedback.pushInfo(f'Found {len(all_intersections)} intersection points')
            
            # Step 3: Split lines at intersection points
            feedback.pushInfo('Step 3: Breaking lines at intersections...')
            all_segments = []
            
            for i, item in enumerate(extended_geometries):
                if feedback.isCanceled():
                    break
                
                geom = item['extended_geom']
                original_feat = item['original_feat']
                
                # Collect all intersection points for this line
                split_points = []
                for isect in all_intersections:
                    if i in isect['line_indices']:
                        split_points.append(isect['point'])
                
                if split_points:
                    # Split the geometry at these points
                    segments = self._split_line_at_points(geom, split_points)
                    for seg in segments:
                        all_segments.append({
                            'geometry': seg,
                            'original_feat': original_feat
                        })
                else:
                    # No splits needed
                    all_segments.append({
                        'geometry': geom,
                        'original_feat': original_feat
                    })
                
                feedback.setProgress(50 + int((i / len(extended_geometries)) * 25))
            
            feedback.pushInfo(f'Created {len(all_segments)} segments')
            
            # Step 4: Optionally delete small segments
            if delete_small:
                feedback.pushInfo(f'Step 4: Removing segments shorter than {small_threshold}...')
                filtered_segments = []
                removed_count = 0
                
                for seg in all_segments:
                    if seg['geometry'].length() >= small_threshold:
                        filtered_segments.append(seg)
                    else:
                        removed_count += 1
                
                feedback.pushInfo(f'Removed {removed_count} small segments')
                all_segments = filtered_segments
            
            # Step 5: Write output features
            feedback.pushInfo('Writing output features...')
            
            for i, seg in enumerate(all_segments):
                if feedback.isCanceled():
                    break
                
                feat_out = QgsFeature()
                feat_out.setGeometry(seg['geometry'])
                
                # Copy attributes from original feature if fields exist
                if source.fields().count() > 0:
                    feat_out.setFields(source.fields())
                    try:
                        feat_out.setAttributes(seg['original_feat'].attributes())
                    except:
                        pass
                
                sink.addFeature(feat_out)
                
                feedback.setProgress(75 + int((i / len(all_segments)) * 25))
            
            feedback.pushInfo(f'Output: {len(all_segments)} features')
            
            return {self.OUTPUT: dest_id}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
    
    def _extend_line_both_ends(self, geom, distance):
        """Extend a line geometry on both ends by the specified distance."""
        if geom.isMultipart():
            # Handle multipart
            lines = geom.asMultiPolyline()
            extended_lines = []
            for line in lines:
                extended = self._extend_polyline(line, distance)
                extended_lines.append(extended)
            return QgsGeometry.fromMultiPolylineXY(extended_lines)
        else:
            line = geom.asPolyline()
            extended = self._extend_polyline(line, distance)
            return QgsGeometry.fromPolylineXY(extended)
    
    def _extend_polyline(self, points, distance):
        """Extend a polyline (list of QgsPointXY) on both ends."""
        if len(points) < 2:
            return points
        
        # Extend start
        p0, p1 = points[0], points[1]
        dx = p0.x() - p1.x()
        dy = p0.y() - p1.y()
        length = math.sqrt(dx * dx + dy * dy)
        
        if length > 0:
            new_start = QgsPointXY(
                p0.x() + (dx / length) * distance,
                p0.y() + (dy / length) * distance
            )
        else:
            new_start = p0
        
        # Extend end
        pn, pn1 = points[-1], points[-2]
        dx = pn.x() - pn1.x()
        dy = pn.y() - pn1.y()
        length = math.sqrt(dx * dx + dy * dy)
        
        if length > 0:
            new_end = QgsPointXY(
                pn.x() + (dx / length) * distance,
                pn.y() + (dy / length) * distance
            )
        else:
            new_end = pn
        
        # Create extended polyline
        extended = [new_start] + list(points) + [new_end]
        return extended
    
    def _extract_points_from_geometry(self, geom):
        """Extract QgsPointXY objects from a geometry (point, multipoint, or vertices)."""
        points = []
        
        if geom.type() == QgsWkbTypes.PointGeometry:
            if geom.isMultipart():
                for pt in geom.asMultiPoint():
                    points.append(pt)
            else:
                points.append(geom.asPoint())
        elif geom.type() == QgsWkbTypes.LineGeometry:
            # For line intersections, use the centroid or all vertices
            if geom.isMultipart():
                for line in geom.asMultiPolyline():
                    for pt in line:
                        points.append(pt)
            else:
                for pt in geom.asPolyline():
                    points.append(pt)
        
        return points
    
    def _split_line_at_points(self, geom, split_points):
        """Split a line geometry at the given points."""
        if geom.isMultipart():
            lines = geom.asMultiPolyline()
        else:
            lines = [geom.asPolyline()]
        
        all_segments = []
        
        for line in lines:
            if len(line) < 2:
                continue
            
            # Create temporary geometry for distance calculations
            line_geom = QgsGeometry.fromPolylineXY(line)
            
            # Calculate distances along line for each split point
            splits_with_dist = []
            for pt in split_points:
                pt_geom = QgsGeometry.fromPointXY(pt)
                dist_along = line_geom.lineLocatePoint(pt_geom)
                if dist_along > 0 and dist_along < line_geom.length():
                    splits_with_dist.append((dist_along, pt))
            
            # Sort by distance
            splits_with_dist.sort(key=lambda x: x[0])
            
            if not splits_with_dist:
                # No valid splits on this line
                all_segments.append(line_geom)
                continue
            
            # Split the line
            prev_dist = 0
            for dist, pt in splits_with_dist:
                if dist > prev_dist:
                    # Extract segment from prev_dist to dist
                    seg = self._extract_line_segment(line_geom, prev_dist, dist)
                    if seg and not seg.isEmpty():
                        all_segments.append(seg)
                    prev_dist = dist
            
            # Add final segment
            if prev_dist < line_geom.length():
                seg = self._extract_line_segment(line_geom, prev_dist, line_geom.length())
                if seg and not seg.isEmpty():
                    all_segments.append(seg)
        
        return all_segments
    
    def _extract_line_segment(self, line_geom, start_dist, end_dist):
        """Extract a segment of a line between two distances along it."""
        try:
            # Get interpolated points at start and end distances
            start_pt = line_geom.interpolate(start_dist)
            end_pt = line_geom.interpolate(end_dist)
            
            if start_pt.isNull() or end_pt.isNull():
                return None
            
            # Simple approach: create line from start to end point
            # For more accuracy, we should include intermediate vertices
            pts = [start_pt.asPoint(), end_pt.asPoint()]
            
            return QgsGeometry.fromPolylineXY(pts)
        except:
            return None
