# -*- coding: utf-8 -*-
"""
algorithms/stream_network/stream_ordering.py
Unified stream ordering tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingException
)
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter
from osgeo import gdal
import numpy as np

class StreamOrderingAlgorithm(QgsProcessingAlgorithm):
    """Unified stream ordering tool (Strahler, Shreve, Horton, Hack)."""
    
    INPUT_DEM = 'INPUT_DEM'
    INPUT_STREAMS = 'INPUT_STREAMS'
    INPUT_FLOW_DIR = 'INPUT_FLOW_DIR'
    METHOD = 'METHOD'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'Strahler Order', 
        'Shreve Magnitude', 
        'Horton Order', 
        'Hack Order', 
        'Topological Order'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return StreamOrderingAlgorithm()
    
    def name(self):
        return 'stream_ordering'
    
    def displayName(self):
        return 'Stream Ordering'
    
    def group(self):
        return 'Stream Network Analysis'
    
    def groupId(self):
        return 'stream_network'
    
    def shortHelpString(self):
        return """
        Calculate stream order/magnitude.
        
        Methods:
        - Strahler: Hierarchical ordering (1+1=2, 1+2=2).
        - Shreve: Magnitude summation (1+1=2, 1+2=3).
        - Horton: Strahler with main stem continuity.
        - Hack: Main stream length order.
        - Topological: Unique topological ID.
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_STREAMS,
                'Input Stream Raster'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_FLOW_DIR,
                'Input Flow Direction (D8)'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                'Ordering Method',
                options=self.METHOD_OPTIONS,
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Stream Order'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            streams_layer = self.parameterAsRasterLayer(parameters, self.INPUT_STREAMS, context)
            flow_dir_layer = self.parameterAsRasterLayer(parameters, self.INPUT_FLOW_DIR, context)
            method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if streams_layer is None or flow_dir_layer is None:
                raise QgsProcessingException('Invalid inputs')
            
            feedback.pushInfo('Loading data...')
            # We use DEMProcessor just to load arrays, we don't need DEM values specifically
            # but we need consistent dimensions
            s_proc = DEMProcessor(streams_layer.source())
            fd_proc = DEMProcessor(flow_dir_layer.source())
            
            streams = s_proc.array
            flow_dir = fd_proc.array
            
            if streams.shape != flow_dir.shape:
                raise QgsProcessingException('Input rasters must have same dimensions')
            
            feedback.pushInfo(f'Calculating {self.METHOD_OPTIONS[method_idx]}...')
            feedback.setProgress(30)
            
            # Initialize router (dummy DEM)
            router = FlowRouter(np.zeros_like(streams), s_proc.cellsize_x)
            
            if method_idx == 0:  # Strahler
                result = router.strahler_order(flow_dir, streams)
            elif method_idx == 1:  # Shreve
                result = router.shreve_order(flow_dir, streams)
            elif method_idx == 2: # Horton
                feedback.pushInfo("Calculating Horton Order...")
                result = router.horton_order(flow_dir, streams)
            elif method_idx == 3: # Hack
                feedback.pushInfo("Calculating Hack Order...")
                result = router.hack_order(flow_dir, streams)
            elif method_idx == 4: # Topological
                # Topological order is just a unique ID per link, sorted?
                # Or just Stream Link ID?
                # Usually "Topological Order" refers to a sorting where upstream < downstream.
                # Let's return Stream Link IDs as a proxy for now, or implement a specific topological sort ID.
                # Since we have assign_stream_link_ids, let's use that.
                feedback.pushInfo("Calculating Topological Order (Link IDs)...")
                result = router.assign_stream_link_ids(flow_dir, streams)
            else:
                result = router.strahler_order(flow_dir, streams)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            s_proc.save_raster(output_path, result, dtype=gdal.GDT_Int32, nodata=-9999)
            s_proc.close()
            fd_proc.close()
            
            # Apply stream order symbology
            feedback.pushInfo('Applying Stream Order symbology...')
            try:
                from qgis.core import QgsRasterLayer, QgsProject
                from ...core.symbology_utils import apply_stream_order_symbology
                import os
                
                # Check if QGIS will load this layer (user checked "Open output file")
                will_load = context.willLoadLayerOnCompletion(output_path)
                
                if will_load:
                    feedback.pushInfo('Loading output with symbology...')
                    
                    # Create the layer ourselves
                    layer_name = os.path.splitext(os.path.basename(output_path))[0]
                    styled_layer = QgsRasterLayer(output_path, layer_name)
                    
                    if styled_layer.isValid():
                        # Apply symbology
                        apply_stream_order_symbology(styled_layer)
                        feedback.pushInfo('Stream Order symbology applied (color palette)')
                        
                        # Add to project

                        QgsProject.instance().addMapLayer(styled_layer)
                        feedback.pushInfo('Styled layer added to project')
                        
                        # Tell QGIS NOT to load this layer again
                        context.setLayersToLoadOnCompletion({})
                    else:
                        feedback.pushWarning('Could not load styled layer')
                        

            except Exception as e:
                feedback.pushWarning(f'Could not apply symbology: {str(e)}')
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
