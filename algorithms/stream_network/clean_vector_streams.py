# -*- coding: utf-8 -*-
"""
algorithms/stream_network/clean_vector_streams.py
Clean Vector Stream Network (Remove short segments)
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingException,
    QgsFeature,
    QgsGeometry,
    QgsWkbTypes,
    QgsProcessing
)

class CleanVectorStreamsAlgorithm(QgsProcessingAlgorithm):
    """Clean vector streams by removing short segments."""
    
    INPUT_STREAMS = 'INPUT_STREAMS'
    MIN_LENGTH = 'MIN_LENGTH'
    UNIT = 'UNIT'
    OUTPUT = 'OUTPUT'
    
    UNIT_OPTIONS = ['Map Units', 'Meters']
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return CleanVectorStreamsAlgorithm()
    
    def name(self):
        return 'clean_vector_streams'
    
    def displayName(self):
        return 'Clean Vector Streams'
    
    def group(self):
        return 'Stream Network Analysis'
    
    def groupId(self):
        return 'stream_network'
    
    def shortHelpString(self):
        return """
        Removes stream segments that are shorter than the specified threshold.
        
        Use this tool to clean up "noisy" vector stream networks, often resulting from
        raster-to-vector conversion or merging operations.
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
                self.MIN_LENGTH,
                'Minimum Length',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=100.0,
                minValue=0.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.UNIT,
                'Length Unit',
                options=self.UNIT_OPTIONS,
                defaultValue=0 # Map Units
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                'Cleaned Streams',
                type=QgsProcessing.TypeVectorLine
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            source = self.parameterAsSource(parameters, self.INPUT_STREAMS, context)
            min_length = self.parameterAsDouble(parameters, self.MIN_LENGTH, context)
            unit_idx = self.parameterAsEnum(parameters, self.UNIT, context)
            
            if source is None:
                raise QgsProcessingException('Input Streams vector layer is required')
            
            # Unit Handling
            # If Map Units (0), use directly.
            # If Meters (1) and CRS is Geographic, convert?
            # QgsGeometry.length() returns map units.
            
            check_meters = (unit_idx == 1)
            crs = source.sourceCrs()
            
            feedback.pushInfo('Cleaning streams...')
            
            (sink, dest_id) = self.parameterAsSink(
                parameters, self.OUTPUT, context,
                source.fields(), QgsWkbTypes.LineString, crs
            )
            
            if sink is None:
                raise QgsProcessingException('Error creating output sink')
            
            features = source.getFeatures()
            total = source.featureCount()
            count = 0
            kept = 0
            
            # For meter calculation on geographic CRS
            from qgis.core import QgsDistanceArea
            da = QgsDistanceArea()
            da.setSourceCrs(crs, context.transformContext())
            
            for feat in features:
                if feedback.isCanceled(): break
                
                geom = feat.geometry()
                if geom.isNull(): continue
                
                length = 0.0
                if check_meters:
                     length = da.measureLength(geom)
                else:
                     length = geom.length()
                     
                if length >= min_length:
                    sink.addFeature(feat)
                    kept += 1
                    
                count += 1
                if count % 100 == 0:
                    feedback.setProgress(int(count/total * 100))
            
            feedback.pushInfo(f'Kept {kept} streams out of {count}. Removed {count - kept}.')
            return {self.OUTPUT: dest_id}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
