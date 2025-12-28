# -*- coding: utf-8 -*-
"""
algorithms/hydrological/watershed_delineation.py
Watershed delineation from pour points
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterBoolean,
    QgsProcessingException,
    QgsProcessing
)
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter
from ...core.hydro_utils import HydrologicalAnalyzer
import numpy as np
from osgeo import gdal


class WatershedDelineationAlgorithm(QgsProcessingAlgorithm):
    """Delineate watersheds from pour points."""
    
    INPUT_DEM = 'INPUT_DEM'
    INPUT_POUR_POINTS = 'INPUT_POUR_POINTS'
    FILL_DEPRESSIONS = 'FILL_DEPRESSIONS'
    OUTPUT_WATERSHEDS = 'OUTPUT_WATERSHEDS'
    OUTPUT_FLOW_DIR = 'OUTPUT_FLOW_DIR'
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return WatershedDelineationAlgorithm()
    
    def name(self):
        return 'watershed_delineation'
    
    def displayName(self):
        return 'Watershed Delineation'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Delineate watersheds from pour points.
        
        Traces upslope from each pour point to identify all cells
        that drain to that point.
        
        Parameters:
        - INPUT_DEM: Digital elevation model
        - INPUT_POUR_POINTS: Point layer with pour point locations
        - FILL_DEPRESSIONS: Fill depressions before routing
        - OUTPUT_WATERSHEDS: Output watershed raster (unique ID per watershed)
        - OUTPUT_FLOW_DIR: Optional flow direction raster
        
        Each watershed receives a unique ID starting from 1.
        """
    
    def initAlgorithm(self, config=None):
        # Input DEM
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM,
                'Input DEM'
            )
        )
        
        # Pour points
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_POUR_POINTS,
                'Pour points',
                [QgsProcessing.TypeVectorPoint]
            )
        )
        
        # Fill depressions
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FILL_DEPRESSIONS,
                'Fill depressions',
                defaultValue=True
            )
        )
        
        # Output watersheds
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_WATERSHEDS,
                'Output watersheds'
            )
        )
        
        # Optional flow direction output
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_FLOW_DIR,
                'Output flow direction',
                optional=True
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        """Process algorithm."""
        try:
            # Get parameters
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
            pour_points = self.parameterAsSource(parameters, self.INPUT_POUR_POINTS, context)
            fill_deps = self.parameterAsBool(parameters, self.FILL_DEPRESSIONS, context)
            output_watersheds_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_WATERSHEDS, context)
            output_flow_dir_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_FLOW_DIR, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            if pour_points is None:
                raise QgsProcessingException('Invalid pour points')
            
            feedback.pushInfo('Loading DEM...')
            processor = DEMProcessor(input_layer.source())
            
            feedback.setProgress(10)
            
            # Convert pour points to raster coordinates
            feedback.pushInfo('Processing pour points...')
            pour_point_coords = self._extract_pour_point_coords(
                pour_points, processor.geotransform
            )
            
            if len(pour_point_coords) == 0:
                raise QgsProcessingException('No valid pour points found')
            
            feedback.pushInfo(f'Found {len(pour_point_coords)} pour points')
            feedback.setProgress(20)
            
            # Initialize flow router
            router = FlowRouter(processor.array, processor.cellsize_x)
            
            # Fill depressions if requested
            if fill_deps:
                feedback.pushInfo('Filling depressions...')
                filled_dem = router.fill_depressions()
                router.dem = filled_dem
                feedback.setProgress(40)
            
            # Calculate flow direction
            feedback.pushInfo('Calculating flow direction...')
            flow_dir = router.d8_flow_direction()
            feedback.setProgress(60)
            
            # Delineate watersheds
            feedback.pushInfo('Delineating watersheds...')
            watersheds = np.zeros_like(processor.array, dtype=np.int32)
            
            hydro = HydrologicalAnalyzer(
                router.dem, processor.cellsize_x
            )
            
            for watershed_id, (row, col) in enumerate(pour_point_coords, start=1):
                feedback.pushInfo(f'Processing watershed {watershed_id}...')
                
                # Delineate this watershed
                watershed_mask = hydro.delineate_watershed_from_point(
                    flow_dir, row, col
                )
                
                # Assign ID (don't overwrite existing watersheds)
                watersheds[watershed_mask == 1] = watershed_id
            
            feedback.setProgress(80)
            
            # Save watersheds
            feedback.pushInfo('Saving watersheds...')
            processor.save_raster(
                output_watersheds_path,
                watersheds,
                dtype=gdal.GDT_Int32,
                nodata=-9999
            )
            
            # Save flow direction if requested
            if output_flow_dir_path:
                feedback.pushInfo('Saving flow direction...')
                processor.save_raster(
                    output_flow_dir_path,
                    flow_dir,
                    dtype=gdal.GDT_Int32
                )
            
            processor.close()
            
            feedback.pushInfo('Watershed delineation complete!')
            feedback.setProgress(100)
            
            results = {self.OUTPUT_WATERSHEDS: output_watersheds_path}
            if output_flow_dir_path:
                results[self.OUTPUT_FLOW_DIR] = output_flow_dir_path
            
            return results
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
    
    def _extract_pour_point_coords(self, point_source, geotransform):
        """Convert pour points to raster coordinates.
        
        Args:
            point_source: QGIS point feature source
            geotransform: GDAL geotransform
            
        Returns:
            list: List of (row, col) tuples
        """
        coords = []
        
        # Extract geotransform parameters
        x_origin = geotransform[0]
        y_origin = geotransform[3]
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]
        
        # Iterate through pour points
        for feature in point_source.getFeatures():
            geom = feature.geometry()
            point = geom.asPoint()
            
            # Convert to raster coordinates
            col = int((point.x() - x_origin) / pixel_width)
            row = int((point.y() - y_origin) / pixel_height)
            
            coords.append((row, col))
        
        return coords
