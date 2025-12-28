# -*- coding: utf-8 -*-
"""
algorithms/hydrological/snap_pour_points.py
Unified snap pour points tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSink,
    QgsProcessingException,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsFields,
    QgsField,
    QgsWkbTypes,
    QgsProcessing,
    QgsProcessingParameterMapLayer
)
from qgis.PyQt.QtCore import QVariant
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter

class SnapPourPointsAlgorithm(QgsProcessingAlgorithm):
    """Unified snap pour points tool."""
    
    INPUT_POUR_POINTS = 'INPUT_POUR_POINTS'
    POUR_POINT_FIELD = 'POUR_POINT_FIELD'
    INPUT_ACC = 'INPUT_ACC'
    SNAP_DIST = 'SNAP_DIST'
    OUTPUT = 'OUTPUT'
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return SnapPourPointsAlgorithm()
    
    def name(self):
        return 'snap_pour_points'
    
    def displayName(self):
        return 'Snap Pour Points'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Snap pour points to the cell with highest flow accumulation within a specified distance.
        
        Use Case:
        Corrects manual digitizing errors by moving points to the actual stream channel.
        Accepts either Raster or Feature pour point data as input.
        """
    
    def initAlgorithm(self, config=None):
        # Single input for both Raster and Vector (Points)
        self.addParameter(
            QgsProcessingParameterMapLayer(
                self.INPUT_POUR_POINTS,
                'Input raster or feature pour point data',
                types=[QgsProcessing.TypeVectorPoint, QgsProcessing.TypeRaster]
            )
        )
        
        # Optional field selection (mostly for vectors)
        from qgis.core import QgsProcessingParameterField
        self.addParameter(
            QgsProcessingParameterField(
                self.POUR_POINT_FIELD,
                'Pour point field (optional)',
                parentLayerParameterName=self.INPUT_POUR_POINTS,
                type=QgsProcessingParameterField.Any,
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_ACC,
                'Flow Accumulation Raster'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SNAP_DIST,
                'Snap Distance (map units)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
                minValue=0.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                'Snapped Points'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            pour_layer = self.parameterAsLayer(parameters, self.INPUT_POUR_POINTS, context)
            acc_layer = self.parameterAsRasterLayer(parameters, self.INPUT_ACC, context)
            snap_dist = self.parameterAsDouble(parameters, self.SNAP_DIST, context)
            field_name = self.parameterAsString(parameters, self.POUR_POINT_FIELD, context)
            
            if pour_layer is None or acc_layer is None:
                raise QgsProcessingException('Input Pour Points and Flow Accumulation are required')
            
            feedback.pushInfo('Loading accumulation data...')
            processor = DEMProcessor(acc_layer.source())
            
            router = FlowRouter(
                processor.array, 
                processor.cellsize_x,
                geotransform=processor.geotransform
            )
            
            points = []
            original_features = [] 
            source_crs = pour_layer.crs()
            source_fields = QgsFields()
            
            from qgis.core import QgsMapLayer, QgsVectorLayer, QgsRasterLayer
            
            if isinstance(pour_layer, QgsVectorLayer):
                # Handle Vector Input
                feedback.pushInfo('Input is Vector Layer.')
                source_fields = pour_layer.fields()
                features = pour_layer.getFeatures()
                
                for feat in features:
                    geom = feat.geometry()
                    pts = []
                    if geom.isMultipart():
                        pts = geom.asMultiPoint()
                    else:
                        pts = [geom.asPoint()]
                    
                    for pt in pts:
                        points.append((pt.x(), pt.y()))
                        original_features.append(feat) # Store original features to copy attributes
                        
            elif isinstance(pour_layer, QgsRasterLayer):
                # Handle Raster Input
                feedback.pushInfo('Input is Raster Layer.')
                # Create default fields for output
                source_fields.append(QgsField("id", QVariant.Int))
                source_fields.append(QgsField("value", QVariant.Double))
                
                points_processor = DEMProcessor(pour_layer.source())
                
                import numpy as np
                valid_mask = ~np.isnan(points_processor.array)
                if points_processor.nodata is not None:
                    valid_mask &= (points_processor.array != points_processor.nodata)
                valid_mask &= (points_processor.array != 0)
                
                y_indices, x_indices = np.where(valid_mask)
                count = len(y_indices)
                feedback.pushInfo(f'Found {count} pour point cells.')
                
                gt = points_processor.geotransform
                
                for i in range(count):
                    r = y_indices[i]
                    c = x_indices[i]
                    val = points_processor.array[r, c]
                    
                    # Center of pixel
                    x = gt[0] + (c + 0.5) * gt[1] + (r + 0.5) * gt[2]
                    y = gt[3] + (c + 0.5) * gt[4] + (r + 0.5) * gt[5]
                    
                    points.append((x, y))
                    
                    # Create dictionary-like object or dummy feature to store value
                    feat = QgsFeature(source_fields)
                    feat.setAttribute("id", i+1)
                    feat.setAttribute("value", float(val))
                    original_features.append(feat)

            # Snap
            feedback.pushInfo(f'Snapping {len(points)} points...')
            feedback.setProgress(20)
            snapped_coords = router.snap_pour_points(points, router.dem, snap_dist)
            
            # Output
            feedback.pushInfo('Creating output...')
            (sink, dest_id) = self.parameterAsSink(
                parameters, self.OUTPUT, context,
                source_fields, QgsWkbTypes.Point, source_crs
            )
            
            if sink is None:
                raise QgsProcessingException('Error creating output sink')
            
            for i, (x, y) in enumerate(snapped_coords):
                if feedback.isCanceled(): break
                feat = QgsFeature(original_features[i]) if isinstance(original_features[i], QgsFeature) else original_features[i]
                feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                sink.addFeature(feat)
                
            feedback.setProgress(100)
            return {self.OUTPUT: dest_id}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
