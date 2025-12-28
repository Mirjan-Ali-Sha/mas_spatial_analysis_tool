# -*- coding: utf-8 -*-
from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon
import os

class MasGeospatialProvider(QgsProcessingProvider):
    """MAS Spatial Analysis Tool provider."""
    
    def __init__(self):
        """Initialize provider."""
        super().__init__()
        super().__init__()

    def loadAlgorithms(self):
        """Load algorithms."""
        from qgis.core import QgsMessageLog, Qgis
        
        # Helper to safely load algorithm
        def load_alg(module_path, class_name):
            try:
                import importlib
                module = importlib.import_module(module_path, package='mas_spatial_analysis_tool')
                alg_class = getattr(module, class_name)
                self.addAlgorithm(alg_class())
            except Exception as e:
                QgsMessageLog.logMessage(f"Failed to load {class_name}: {str(e)}", "MAS Spatial Analysis Tool", Qgis.Critical)

        # Geomorphometric
        load_alg('.algorithms.geomorphometric.hillshade_algorithms', 'HillshadeAlgorithm')
        load_alg('.algorithms.geomorphometric.slope_algorithms', 'SlopeAlgorithm')
        load_alg('.algorithms.geomorphometric.slope_algorithms', 'AspectAlgorithm')
        load_alg('.algorithms.geomorphometric.curvature_algorithms', 'CurvatureAlgorithm')
        load_alg('.algorithms.geomorphometric.roughness_algorithms', 'RoughnessAlgorithm')
        load_alg('.algorithms.geomorphometric.position_algorithms', 'TPIAlgorithm')
        load_alg('.algorithms.geomorphometric.feature_detection', 'FeatureDetectionAlgorithm')
        load_alg('.algorithms.geomorphometric.hypsometric_analysis', 'HypsometricAnalysisAlgorithm')
        load_alg('.algorithms.geomorphometric.visibility_algorithms', 'VisibilityAlgorithm')
        load_alg('.algorithms.geomorphometric.directional_analysis', 'DirectionalAnalysisAlgorithm')
        load_alg('.algorithms.geomorphometric.openness', 'OpennessAlgorithm')
        
        # Hydrological
        load_alg('.algorithms.hydrological.flow_direction', 'FlowDirectionAlgorithm')
        load_alg('.algorithms.hydrological.flow_accumulation', 'FlowAccumulationAlgorithm')
        load_alg('.algorithms.hydrological.watershed_delineation', 'WatershedDelineationAlgorithm')
        load_alg('.algorithms.hydrological.depression_algorithms', 'DepressionHandlingAlgorithm')
        load_alg('.algorithms.hydrological.flow_indices', 'FlowIndicesAlgorithm')
        load_alg('.algorithms.hydrological.flow_routing_extended', 'FlowRoutingAlgorithm')
        load_alg('.algorithms.hydrological.flow_distance', 'FlowDistanceAlgorithm')
        load_alg('.algorithms.hydrological.basin_analysis', 'BasinAnalysisAlgorithm')
        load_alg('.algorithms.hydrological.flow_path_statistics', 'FlowPathStatisticsAlgorithm')
        load_alg('.algorithms.hydrological.sink_analysis', 'SinkAnalysisAlgorithm')
        load_alg('.algorithms.hydrological.hydro_enforcement', 'HydroEnforcementAlgorithm')
        load_alg('.algorithms.hydrological.snap_pour_points', 'SnapPourPointsAlgorithm')
        load_alg('.algorithms.hydrological.flow_length', 'FlowLengthAlgorithm')
        load_alg('.algorithms.hydrological.dem_quality', 'DemQualityAlgorithm')
        load_alg('.algorithms.hydrological.hillslopes', 'HillslopesAlgorithm')
        
        # Stream Network
        load_alg('.algorithms.stream_network.stream_extraction', 'ExtractStreamsAlgorithm')
        load_alg('.algorithms.stream_network.stream_ordering', 'StreamOrderingAlgorithm')
        load_alg('.algorithms.stream_network.stream_link_analysis', 'StreamLinkAlgorithm')
        load_alg('.algorithms.stream_network.stream_network_analysis', 'StreamNetworkAnalysisAlgorithm')
        load_alg('.algorithms.stream_network.vector_stream_network', 'VectorStreamNetworkAlgorithm')
        load_alg('.algorithms.stream_network.stream_cleaning', 'StreamCleaningAlgorithm')
        load_alg('.algorithms.stream_network.valley_extraction', 'ValleyExtractionAlgorithm')
        load_alg('.algorithms.stream_network.join_stream_links', 'JoinStreamLinksAlgorithm')

        # Structured Hydrology
        load_alg('.algorithms.hydrology_struct.structured_tools', 'StructBasinAlgorithm')
        load_alg('.algorithms.hydrology_struct.structured_tools', 'StructFillAlgorithm')
        load_alg('.algorithms.hydrology_struct.structured_tools', 'StructFlowAccumulationAlgorithm')
        load_alg('.algorithms.hydrology_struct.structured_tools', 'StructFlowDirectionAlgorithm')
        load_alg('.algorithms.hydrology_struct.structured_tools', 'StructFlowDistanceAlgorithm')
        load_alg('.algorithms.hydrology_struct.structured_tools', 'StructFlowLengthAlgorithm')
        load_alg('.algorithms.hydrology_struct.structured_tools', 'StructSinkAlgorithm')
        load_alg('.algorithms.hydrology_struct.structured_tools', 'StructSnapPourPointAlgorithm')
        load_alg('.algorithms.hydrology_struct.structured_tools', 'StructStreamLinkAlgorithm')
        load_alg('.algorithms.hydrology_struct.structured_tools', 'StructStreamOrderAlgorithm')
        load_alg('.algorithms.hydrology_struct.structured_tools', 'StructStreamToFeatureAlgorithm')
        load_alg('.algorithms.hydrology_struct.structured_tools', 'StructWatershedAlgorithm')

        # # Structured Hydrology Old
        # load_alg('.algorithms.hydrology_struct_old.structured_tools_old', 'StructBasinAlgorithm')
        # load_alg('.algorithms.hydrology_struct_old.structured_tools_old', 'StructFillAlgorithm')
        # load_alg('.algorithms.hydrology_struct_old.structured_tools_old', 'StructFlowAccumulationAlgorithm')
        # load_alg('.algorithms.hydrology_struct_old.structured_tools_old', 'StructFlowDirectionAlgorithm')
        # load_alg('.algorithms.hydrology_struct_old.structured_tools_old', 'StructFlowDistanceAlgorithm')
        # load_alg('.algorithms.hydrology_struct_old.structured_tools_old', 'StructFlowLengthAlgorithm')
        # load_alg('.algorithms.hydrology_struct_old.structured_tools_old', 'StructSinkAlgorithm')
        # load_alg('.algorithms.hydrology_struct_old.structured_tools_old', 'StructSnapPourPointAlgorithm')
        # load_alg('.algorithms.hydrology_struct_old.structured_tools_old', 'StructStreamLinkAlgorithm')
        # load_alg('.algorithms.hydrology_struct_old.structured_tools_old', 'StructStreamOrderAlgorithm')
        # load_alg('.algorithms.hydrology_struct_old.structured_tools_old', 'StructStreamToFeatureAlgorithm')
        # load_alg('.algorithms.hydrology_struct_old.structured_tools_old', 'StructWatershedAlgorithm')

    def id(self):
        """Return provider ID."""
        return 'mas_spatial_analysis_tool'

    def name(self):
        """Return provider name."""
        return self.tr('MAS Spatial Analysis Tool')

    def icon(self):
        """Return provider icon."""
        return QIcon(os.path.join(os.path.dirname(__file__), 'icons', 'mas_icon.svg'))
