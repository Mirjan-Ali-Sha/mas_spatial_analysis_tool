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
    QgsProcessing
)
from qgis.PyQt.QtCore import QVariant
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter

class SnapPourPointsAlgorithm(QgsProcessingAlgorithm):
    """Unified snap pour points tool."""
    
    INPUT_POINTS = 'INPUT_POINTS'
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
        
        Useful for ensuring pour points are located on the digital stream network before watershed delineation.
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_POINTS,
                'Input Pour Points',
                types=[QgsProcessing.TypeVectorPoint]
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
            points_source = self.parameterAsSource(parameters, self.INPUT_POINTS, context)
            acc_layer = self.parameterAsRasterLayer(parameters, self.INPUT_ACC, context)
            snap_dist = self.parameterAsDouble(parameters, self.SNAP_DIST, context)
            
            if points_source is None or acc_layer is None:
                raise QgsProcessingException('Input Points and Flow Accumulation are required')
            
            feedback.pushInfo('Loading data...')
            processor = DEMProcessor(acc_layer.source())
            
            # Initialize router with acc array as "dem"
            # We pass geotransform so snap_pour_points can convert coordinates
            router = FlowRouter(
                processor.array, 
                processor.cellsize_x,
                geotransform=processor.geotransform
            )
            
            # We pass router.dem (which is acc) to snap_pour_points
            # snap_pour_points expects flow_acc array
            
            # Get points
            points = []
            features = points_source.getFeatures()
            original_features = [] # Keep to preserve attributes?
            
            for feat in features:
                geom = feat.geometry()
                if geom.isMultipart():
                    pts = geom.asMultiPoint()
                    for pt in pts:
                        points.append((pt.x(), pt.y()))
                        original_features.append(feat)
                else:
                    pt = geom.asPoint()
                    points.append((pt.x(), pt.y()))
                    original_features.append(feat)
            
            feedback.pushInfo(f'Snapping {len(points)} points...')
            feedback.setProgress(20)
            
            # Snap
            # We pass router.dem as flow_acc because we loaded acc_layer into it
            snapped_coords = router.snap_pour_points(points, router.dem, snap_dist)
            
            feedback.pushInfo('Creating output...')
            feedback.setProgress(80)
            
            # Create sink
            fields = points_source.fields()
            (sink, dest_id) = self.parameterAsSink(
                parameters,
                self.OUTPUT,
                context,
                fields,
                QgsWkbTypes.Point,
                points_source.sourceCrs()
            )
            
            if sink is None:
                raise QgsProcessingException('Error creating output sink')
            
            # Add features
            for i, (x, y) in enumerate(snapped_coords):
                if feedback.isCanceled():
                    break
                    
                feat = QgsFeature(original_features[i])
                feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                sink.addFeature(feat)
                
            feedback.setProgress(100)
            
            return {self.OUTPUT: dest_id}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
