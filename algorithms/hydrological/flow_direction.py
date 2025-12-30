# -*- coding: utf-8 -*-
"""
algorithms/hydrological/flow_direction.py
Native D8 Flow Direction implementation
"""

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsProcessingException
)
from osgeo import gdal
import numpy as np
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter

class FlowDirectionAlgorithm(QgsProcessingAlgorithm):
    """Unified Flow Direction tool (D8, D-Inf, Rho8)."""
    
    INPUT = 'INPUT'
    METHOD = 'METHOD'
    FORCE_EDGE_OUTWARD = 'FORCE_EDGE_OUTWARD'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'D8 (Deterministic 8-neighbor)', 
        'D-Infinity (Tarboton 1997)', 
        'Rho8 (Stochastic 8-neighbor)',
        'MFD (Multiple Flow Direction)'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return FlowDirectionAlgorithm()
    
    def name(self):
        return 'flow_direction'
    
    def displayName(self):
        return 'Flow Direction'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Calculate flow direction from DEM.
        
        Methods:
        - D8: Routes flow to steepest downslope neighbor (1-128 codes).
        - D-Infinity: Routes flow to single angle (radians).
        - Rho8: Stochastic D8 to break parallel lines.
        
        Options:
        - Force edge cells outward: Ensures all cells at the DEM boundary
          drain to the edge (standard GIS behavior).
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                'Input DEM'
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
            QgsProcessingParameterBoolean(
                self.FORCE_EDGE_OUTWARD,
                'Force all edge cells to flow outwards (optional)',
                defaultValue=False
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output flow direction'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
            method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
            force_edge = self.parameterAsBool(parameters, self.FORCE_EDGE_OUTWARD, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = DEMProcessor(input_layer.source())
            
            feedback.setProgress(20)
            
            # Initialize router
            router = FlowRouter(processor.array, processor.cellsize_x)
            
            feedback.pushInfo(f'Calculating {self.METHOD_OPTIONS[method_idx]}...')
            
            if method_idx == 0: # D8
                flow_dir = router.d8_flow_direction()
                dtype = gdal.GDT_Int32
            elif method_idx == 1: # D-Inf
                flow_dir = router.dinf_flow_direction()
                dtype = gdal.GDT_Float32
            elif method_idx == 2: # Rho8
                flow_dir = router.rho8_flow_direction()
                dtype = gdal.GDT_Int32
            elif method_idx == 3: # MFD
                # MFD calculates flow proportions, not single direction
                # For direction output, we use D8 but note MFD is primarily for accumulation
                feedback.pushInfo('Note: MFD is primarily for flow accumulation. Using D8 for direction output.')
                flow_dir = router.d8_flow_direction()
                dtype = gdal.GDT_Int32
            
            # Force edge cells to flow outward if requested
            if force_edge and method_idx in [0, 2]:  # Only for D8-based methods
                feedback.pushInfo('Forcing edge cells to flow outward...')
                rows, cols = flow_dir.shape
                
                # D8 direction codes: E=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128
                # Top edge flows North (64)
                flow_dir[0, 1:-1] = 64
                # Bottom edge flows South (4)
                flow_dir[rows-1, 1:-1] = 4
                # Left edge flows West (16)
                flow_dir[1:-1, 0] = 16
                # Right edge flows East (1)
                flow_dir[1:-1, cols-1] = 1
                # Corners
                flow_dir[0, 0] = 32  # NW corner flows NW
                flow_dir[0, cols-1] = 128  # NE corner flows NE
                flow_dir[rows-1, 0] = 8  # SW corner flows SW
                flow_dir[rows-1, cols-1] = 2  # SE corner flows SE
            
            feedback.setProgress(80)
            
            feedback.pushInfo('Saving output...')
            processor.save_raster(output_path, flow_dir, dtype=dtype)
            processor.close()
            
            # Apply D8 Flow Direction symbology
            feedback.pushInfo('Applying Flow Direction symbology...')
            try:
                from ...core.symbology_utils import apply_flow_direction_symbology
                from qgis.core import QgsRasterLayer, QgsProject
                
                # Check if QGIS will load this layer (user checked "Open output file")
                will_load = context.willLoadLayerOnCompletion(output_path)
                
                if will_load:
                    feedback.pushInfo('Loading output with symbology...')
                    
                    # Create the layer ourselves
                    import os
                    layer_name = os.path.splitext(os.path.basename(output_path))[0]
                    styled_layer = QgsRasterLayer(output_path, layer_name)
                    
                    if styled_layer.isValid():
                        # Apply symbology
                        apply_flow_direction_symbology(styled_layer)
                        feedback.pushInfo('D8 Flow Direction symbology applied (8 directional colors)')
                        
                        # Add to project
                        QgsProject.instance().addMapLayer(styled_layer)
                        feedback.pushInfo('Styled layer added to project')
                        
                        # Tell QGIS NOT to load this layer again
                        context.setLayersToLoadOnCompletion({})
                    else:
                        feedback.pushInfo('Warning: Could not load styled layer')
                        

            except Exception as e:
                feedback.pushInfo(f'Note: Could not apply symbology: {e}')
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error processing flow direction: {str(e)}')

