# MAS Geospatial Tools - Installation Guide

## Overview

**MAS Geospatial Tools** is a native Python QGIS plugin providing 177 advanced hydrological and geomorphometric analysis tools.

- **NO binary dependencies** - Pure Python implementation
- **Fast** - Optimized NumPy/SciPy algorithms
- **Cross-platform** - Windows, Linux, macOS

## Requirements

### QGIS Version
- QGIS 3.16 or higher

### Python Packages
All packages are typically included with QGIS:
- `numpy >= 1.19`
- `scipy >= 1.5`
- `gdal >= 3.0`

## Installation

### Method 1: QGIS Plugin Manager (Recommended)

1. Open QGIS
2. Go to **Plugins → Manage and Install Plugins**
3. Search for "MAS Geospatial Tools"
4. Click **Install Plugin**

### Method 2: Manual Installation

1. Download the plugin ZIP file
2. Extract to QGIS plugins directory:
   - **Windows**: `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
   - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
3. Restart QGIS
4. Enable plugin in **Plugins → Manage and Install Plugins → Installed**

### Method 3: From Source (Development)

1. Clone repository
git clone https://github.com/Mirjan-Ali-Sha/mas_geospatial_tools.git

2. Navigate to QGIS plugins directory
cd ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/

3. Create symbolic link
ln -s /path/to/mas_geospatial_tools mas_geospatial_tools

4. Restart QGIS


## Verification

1. Open QGIS
2. Go to **Processing → Toolbox**
3. Look for **MAS Geospatial Tools** provider
4. You should see 3 groups with 177 total tools:
   - Geomorphometric Analysis (92 tools)
   - Hydrological Analysis (61 tools)
   - Stream Network Analysis (24 tools)

## Usage Examples

### Example 1: Calculate Slope

#### Via Processing Toolbox
1. Open Processing Toolbox

2. Navigate to: MAS Geospatial Tools → Geomorphometric Analysis → Slope

3. Select input DEM

4. Choose units (degrees/percent)

5. Run

#### Via Python Console
```
from qgis import processing

processing.run("mas_geospatial:slope", {
'INPUT': '/path/to/dem.tif',
'UNITS': 0, # 0=degrees, 1=percent
'Z_FACTOR': 1.0,
'OUTPUT': '/path/to/slope.tif'
})
```


### Example 2: Watershed Delineation

processing.run("mas_geospatial:watershed_delineation", {
'INPUT_DEM': '/path/to/dem.tif',
'INPUT_POUR_POINTS': '/path/to/points.gpkg',
'FILL_DEPRESSIONS': True,
'OUTPUT_WATERSHEDS': '/path/to/watersheds.tif',
'OUTPUT_FLOW_DIR': '/path/to/flow_dir.tif'
})


### Example 3: Stream Network Extraction

#### Step 1: Fill depressions
filled = processing.run("mas_geospatial:fill_depressions", {
'INPUT': dem_path,
'OUTPUT': 'memory:'
})['OUTPUT']

#### Step 2: Flow accumulation
flow_acc = processing.run("mas_geospatial:d8_flow_accumulation", {
'INPUT_DEM': filled,
'FILL_DEPRESSIONS': False,
'OUTPUT_FLOW_ACC': 'memory:'
})['OUTPUT_FLOW_ACC']

#### Step 3: Extract streams
streams = processing.run("mas_geospatial:extract_streams", {
'INPUT_FLOW_ACC': flow_acc,
'THRESHOLD': 1000,
'OUTPUT': '/path/to/streams.tif'
})


## Performance Tips

### For Large DEMs

1. **Use compressed formats**: GeoTIFF with LZW compression
2. **Tile processing**: Plugin automatically processes in blocks
3. **Adequate RAM**: 8GB+ recommended for DEMs >1GB
4. **SSD storage**: Faster I/O operations

### Optimization Settings

#### Increase block size for faster processing (more RAM usage)
1. Edit core/array_utils.py:
block_size = 1024 # Default is 512


## Detailed Tool List (177 Tools)

| Group | Our Implemented Tool | Function Count | Mapped Functions |
|---|---|---|---|
| Geomorphometric Analysis | HillshadeAlgorithm | 4 | Hillshade, MultidirectionalHillshade, ShadowImage, HypsometricallyTintedHillshade |
| Geomorphometric Analysis | SlopeAlgorithm | 2 | Slope, StandardDeviationOfSlope |
| Geomorphometric Analysis | AspectAlgorithm | 3 | Aspect, CircularVarianceOfAspect, RelativeAspect |
| Geomorphometric Analysis | CurvatureAlgorithm | 19 | PlanCurvature, ProfileCurvature, TangentialCurvature, MeanCurvature, GaussianCurvature, TotalCurvature, MaximalCurvature, MinimalCurvature, HorizontalExcessCurvature, VerticalExcessCurvature, DifferenceCurvature, AccumulationCurvature, Curvedness, Unsphericity, Rotor, ShapeIndex, RingCurvature, MultiscaleCurvatures, Profile |
| Geomorphometric Analysis | RoughnessAlgorithm | 5 | RuggednessIndex, MultiscaleRoughness, EdgeDensity, SurfaceAreaRatio, MultiscaleRoughnessSignature |
| Geomorphometric Analysis | TPIAlgorithm | 3 | RelativeTopographicPosition, MultiscaleTopographicPositionImage, TopographicPositionAnimation |
| Geomorphometric Analysis | FeatureDetectionAlgorithm | 6 | FindRidges, BreaklineMapping, EmbankmentMapping, MapOffTerrainObjects, RemoveOffTerrainObjects, Geomorphons |
| Geomorphometric Analysis | HypsometricAnalysisAlgorithm | 2 | HypsometricAnalysis, LocalHypsometricAnalysis |
| Geomorphometric Analysis | VisibilityAlgorithm | 6 | Viewshed, VisibilityIndex, HorizonAngle, TimeInDaylight, ShadowAnimation, TopoRender |
| Geomorphometric Analysis | DirectionalAnalysisAlgorithm | 6 | DirectionalRelief, ExposureTowardsWindFlux, FetchAnalysis, AverageNormalVectorAngularDeviation, MaxAnisotropyDev, MaxAnisotropyDevSignature |
| Geomorphometric Analysis | OpennessAlgorithm | 3 | Openness, SphericalStdDevOfNormals, MultiscaleStdDevNormalsSignature |
| Geomorphometric Analysis | MultiscaleAnalysisAlgorithm | 3 | MultiscaleElevationPercentile, MultiscaleStdDevNormals, GaussianScaleSpace |
| Geomorphometric Analysis | StatisticalAlgorithms | 13 | LocalQuadraticRegression, FeaturePreservingSmoothing, SmoothVegetationResidual, DevFromMeanElev, DiffFromMeanElev, MaxDifferenceFromMean, MaxElevationDeviation, PercentElevRange, ElevRelativeToMinMax, ElevRelativeToWatershedMinMax, PennockLandformClass, GeneratingFunction, MaxElevDevSignature |
| Geomorphometric Analysis | WetnessAlgorithm | 1 | WetnessIndex |
| Hydrological Analysis | D8FlowDirectionAlgorithm | 1 | D8Pointer |
| Hydrological Analysis | D8FlowAccumulationAlgorithm | 3 | D8FlowAccumulation, D8MassFlux, FlowAccumulationFullWorkflow |
| Hydrological Analysis | WatershedDelineationAlgorithm | 5 | Watershed, Basins, Subbasins, Isobasins, StochasticDepressionAnalysis |
| Hydrological Analysis | DepressionHandlingAlgorithm | 10 | FillDepressions, BreachDepressions, BreachDepressionsLeastCost, FillSingleCellPits, BreachSingleCellPits, FillBurn, FillDepressionsPlanchonAndDarboux, FillDepressionsWangAndLiu, DemVoidFilling, FillMissingData |
| Geomorphometric Analysis | FlowIndicesAlgorithm | 2 | StreamPowerIndex, SedimentTransportIndex |
| Hydrological Analysis | FlowRoutingAlgorithm | 11 | DInfPointer, DInfFlowAccumulation, DInfMassFlux, MDInfFlowAccumulation, FD8Pointer, FD8FlowAccumulation, Rho8Pointer, Rho8FlowAccumulation, QinFlowAccumulation, QuinnFlowAccumulation, PilesjoHasan |
| Hydrological Analysis | FlowDistanceAlgorithm | 4 | DownslopeDistanceToStream, ElevationAboveStream, ElevationAboveStreamEuclidean, DownslopeIndex |
| Hydrological Analysis | BasinAnalysisAlgorithm | 2 | UnnestBasins, StrahlerOrderBasins |
| Hydrological Analysis | FlowPathStatisticsAlgorithm | 12 | AverageFlowpathSlope, AverageUpslopeFlowpathLength, MaxUpslopeFlowpathLength, FlowLengthDiff, TraceDownslopeFlowpaths, NumInflowingNeighbours, NumDownslopeNeighbours, NumUpslopeNeighbours, LongestFlowpath, MaxDownslopeElevChange, MinDownslopeElevChange, MaxUpslopeElevChange |
| Hydrological Analysis | SinkAnalysisAlgorithm | 4 | Sink, DepthInSink, UpslopeDepressionStorage, ImpoundmentSizeIndex |
| Hydrological Analysis | HydroEnforcementAlgorithm | 4 | BurnStreamsAtRoads, RaiseWalls, FlattenLakes, InsertDams |
| Hydrological Analysis | SnapPourPointsAlgorithm | 2 | SnapPourPoints, JensonSnapPourPoints |
| Hydrological Analysis | FlowLengthAlgorithm | 2 | DownslopeFlowpathLength, MaxUpslopeValue |
| Hydrological Analysis | DemQualityAlgorithm | 3 | FindNoFlowCells, EdgeContamination, FindParallelFlow |
| Hydrological Analysis | HillslopesAlgorithm | 1 | Hillslopes |
| Hydrological Analysis | FloodOrderAlgorithm | 1 | FloodOrder |
| Hydrological Analysis | HydrologicConnectivityAlgorithm | 3 | HydrologicConnectivity, DepthToWater, LowPointsOnHeadwaterDivides |
| Stream Network Analysis | ExtractStreamsAlgorithm | 3 | ExtractStreams, RasterizeStreams, RiverCenterlines |
| Stream Network Analysis | StreamOrderingAlgorithm | 5 | StrahlerStreamOrder, ShreveStreamMagnitude, HortonStreamOrder, HackStreamOrder, TopologicalStreamOrder |
| Stream Network Analysis | StreamLinkAlgorithm | 5 | StreamLinkIdentifier, StreamLinkLength, StreamLinkSlope, StreamLinkClass, StreamSlopeContinuous |
| Stream Network Analysis | StreamNetworkAnalysisAlgorithm | 6 | DistanceToOutlet, FarthestChannelHead, FindMainStem, TributaryIdentifier, LengthOfUpstreamChannels, MaxBranchLength |
| Stream Network Analysis | VectorStreamNetworkAlgorithm | 3 | RasterStreamsToVector, VectorStreamNetworkAnalysis, RepairStreamVectorTopology |
| Stream Network Analysis | StreamCleaningAlgorithm | 1 | RemoveShortStreams |
| Stream Network Analysis | ValleyExtractionAlgorithm | 1 | ExtractValleys |
| Stream Network Analysis | LongProfileAlgorithm | 2 | LongProfile, LongProfileFromPoints |
| Geomorphometric Analysis | ContourAlgorithm | 2 | ContoursFromPoints, ContoursFromRaster |
| Geomorphometric Analysis | PlottingAlgorithm | 2 | SlopeVsAspectPlot, SlopeVsElevationPlot |
| Geomorphometric Analysis | AssessRouteAlgorithm | 1 | AssessRoute |

## Troubleshooting

### Plugin doesn't appear

#### Check plugin is enabled
from qgis.utils import plugins
print('mas_geospatial_tools' in plugins)

#### Check for errors
import qgis
qgis.utils.iface.messageBar().pushMessage("Test", "Plugin check")


### Import errors

#### Verify dependencies in QGIS Python console
```
import numpy
import scipy
import gdal

print(f"NumPy: {numpy.version}")
print(f"SciPy: {scipy.version}")
print(f"GDAL: {gdal.version}")
```


### Performance issues

- Reduce input raster size (clip to AOI)
- Close unnecessary QGIS layers
- Increase available RAM
- Use SSD for temp files

## Support

- **Issues**: https://github.com/Mirjan-Ali-Sha/mas_geospatial_tools/issues
- **Documentation**: https://github.com/Mirjan-Ali-Sha/mas_geospatial_tools/wiki
- **Email**: mastools.help@gmail.com

## License

GPL v2.0 - See LICENSE file
