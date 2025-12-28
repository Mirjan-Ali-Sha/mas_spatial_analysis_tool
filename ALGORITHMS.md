# MAS Spatial Analysis Tool - Algorithm Documentation

**Version**: 1.0  
**Plugin**: MAS Spatial Analysis Tool for QGIS  
**Author**: MAS Geospatial  

---

## Table of Contents

1. [Geomorphometric Analysis](#1-geomorphometric-analysis)
   - [Hillshade](#11-hillshade)
   - [Slope](#12-slope)
   - [Aspect](#13-aspect)
   - [Curvature](#14-curvature)
   - [Roughness](#15-roughness)
   - [Topographic Position Index (TPI)](#16-topographic-position-index-tpi)
   - [Feature Detection](#17-feature-detection)
   - [Hypsometric Analysis](#18-hypsometric-analysis)
   - [Visibility Analysis](#19-visibility-analysis)
   - [Directional Analysis](#110-directional-analysis)
   - [Openness](#111-openness)

2. [Hydrological Analysis](#2-hydrological-analysis)
   - [Flow Direction](#21-flow-direction)
   - [Flow Accumulation](#22-flow-accumulation)
   - [Watershed Delineation](#23-watershed-delineation)
   - [Depression Handling](#24-depression-handling)
   - [Flow Indices](#25-flow-indices)
   - [Flow Routing Extended](#26-flow-routing-extended)
   - [Flow Distance](#27-flow-distance)
   - [Basin Analysis](#28-basin-analysis)
   - [Flow Path Statistics](#29-flow-path-statistics)
   - [Sink Analysis](#210-sink-analysis)
   - [Hydro Enforcement](#211-hydro-enforcement)
   - [Snap Pour Points](#212-snap-pour-points)
   - [Flow Length](#213-flow-length)
   - [DEM Quality](#214-dem-quality)
   - [Hillslopes](#215-hillslopes)

3. [Stream Network Analysis](#3-stream-network-analysis)
   - [Extract Streams](#31-extract-streams)
   - [Stream Ordering](#32-stream-ordering)
   - [Stream Link Analysis](#33-stream-link-analysis)
   - [Stream Network Analysis](#34-stream-network-analysis)
   - [Vector Stream Network](#35-vector-stream-network)
   - [Stream Cleaning](#36-stream-cleaning)
   - [Valley Extraction](#37-valley-extraction)
   - [Join Stream Links](#38-join-stream-links)

4. [Structured Hydrology](#4-structured-hydrology)

5. [References](#5-references)

---

# 1. Geomorphometric Analysis

Algorithms for analyzing terrain morphology and surface characteristics.

---

## 1.1 Hillshade

**Algorithm**: `HillshadeAlgorithm`  
**Category**: Terrain Visualization

### Description
Creates a shaded relief raster by simulating illumination of the terrain surface from a specified light source.

### Mathematical Formula
```
Hillshade = 255 × [(cos(θz) × cos(S)) + (sin(θz) × sin(S) × cos(θa - A))]
```

Where:
- **θz** = Zenith angle of illumination (90° - Altitude)
- **θa** = Azimuth angle of illumination
- **S** = Slope angle
- **A** = Aspect angle

### Parameters
| Parameter | Type   | Default  | Description                           |
| --------- | ------ | -------- | ------------------------------------- |
| Input DEM | Raster | Required | Digital Elevation Model               |
| Azimuth   | Float  | 315°     | Direction of light source (0=N, 90=E) |
| Altitude  | Float  | 45°      | Height of light source above horizon  |
| Z Factor  | Float  | 1.0      | Vertical exaggeration factor          |

### Output
- Hillshade raster (0-255, Byte)

### References
- Burrough, P.A. & McDonnell, R.A. (1998). *Principles of Geographical Information Systems*

---

## 1.2 Slope

**Algorithm**: `SlopeAlgorithm`  
**Category**: Surface Derivatives

### Description
Calculates the rate of change of elevation for each cell in a DEM.

### Mathematical Formula
**Horn's Method (3×3 neighborhood)**:
```
dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 × cellsize)
dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 × cellsize)

Slope (radians) = arctan(√[(dz/dx)² + (dz/dy)²])
Slope (degrees) = Slope (radians) × 180/π
Slope (percent) = tan(Slope) × 100
```

Where the 3×3 kernel is:
```
| a | b | c |
| d | e | f |
| g | h | i |
```

### Parameters
| Parameter | Type   | Default  | Description                            |
| --------- | ------ | -------- | -------------------------------------- |
| Input DEM | Raster | Required | Digital Elevation Model                |
| Unit      | Enum   | Degrees  | Output units (Degrees/Percent/Radians) |

### Output
- Slope raster (Float32)

### References
- Horn, B.K.P. (1981). "Hill Shading and the Reflectance Map". *Proceedings of the IEEE*, 69(1): 14-47.

---

## 1.3 Aspect

**Algorithm**: `AspectAlgorithm`  
**Category**: Surface Derivatives

### Description
Calculates the downslope direction of the maximum rate of change for each cell.

### Mathematical Formula
```
Aspect (radians) = arctan2(dz/dy, -dz/dx)

If Aspect < 0:
    Aspect = 2π + Aspect
    
Aspect (degrees) = Aspect × 180/π
```

**Convention**: 
- 0° = North, 90° = East, 180° = South, 270° = West
- Flat areas = -1 (no aspect)

### Parameters
| Parameter | Type   | Default  | Description             |
| --------- | ------ | -------- | ----------------------- |
| Input DEM | Raster | Required | Digital Elevation Model |

### Output
- Aspect raster (Float32, 0-360 or -1)

### References
- Burrough, P.A. (1986). *Principles of Geographical Information Systems for Land Resources Assessment*

---

## 1.4 Curvature

**Algorithm**: `CurvatureAlgorithm`  
**Category**: Surface Derivatives

### Description
Calculates the curvature of the surface. Profile curvature affects flow acceleration/deceleration; plan curvature affects flow convergence/divergence.

### Mathematical Formulas

**Second Derivatives**:
```
d²z/dx² = (d - 2e + f) / cellsize²
d²z/dy² = (b - 2e + h) / cellsize²
d²z/dxdy = (a - c - g + i) / (4 × cellsize²)
```

**Curvature Types**:
```
Profile Curvature = -[(p² × r) + (2pqt) + (q² × s)] / [(p² + q²) × √(1 + p² + q²)³]

Plan Curvature = -[(q² × r) - (2pqt) + (p² × s)] / [(p² + q²)^(3/2)]

Total Curvature = d²z/dx² + d²z/dy²
```

Where: p = dz/dx, q = dz/dy, r = d²z/dx², s = d²z/dy², t = d²z/dxdy

### Parameters
| Parameter      | Type   | Default  | Description             |
| -------------- | ------ | -------- | ----------------------- |
| Input DEM      | Raster | Required | Digital Elevation Model |
| Curvature Type | Enum   | Total    | Profile/Plan/Total      |

### Output
- Curvature raster (Float32)
- Positive = convex, Negative = concave

### References
- Zevenbergen, L.W. & Thorne, C.R. (1987). "Quantitative analysis of land surface topography". *Earth Surface Processes and Landforms*, 12: 47-56.

---

## 1.5 Roughness

**Algorithm**: `RoughnessAlgorithm`  
**Category**: Surface Texture

### Description
Measures terrain irregularity as the standard deviation of elevation within a moving window.

### Mathematical Formula
```
Roughness = √[Σ(zi - z̄)² / n]
```
Where:
- zi = elevation of each cell in window
- z̄ = mean elevation in window
- n = number of cells in window

### Parameters
| Parameter   | Type   | Default  | Description                        |
| ----------- | ------ | -------- | ---------------------------------- |
| Input DEM   | Raster | Required | Digital Elevation Model            |
| Window Size | Int    | 3        | Size of moving window (odd number) |

### Output
- Roughness raster (Float32)

### References
- Riley, S.J., DeGloria, S.D. & Elliot, R. (1999). "A terrain ruggedness index that quantifies topographic heterogeneity". *Intermountain Journal of Sciences*, 5(1-4): 23-27.

---

## 1.6 Topographic Position Index (TPI)

**Algorithm**: `TPIAlgorithm`  
**Category**: Landform Classification

### Description
Compares elevation of each cell to the mean elevation of surrounding cells within a specified radius.

### Mathematical Formula
```
TPI = z0 - z̄
```
Where:
- z0 = elevation of central cell
- z̄ = mean elevation of neighborhood

**Classification**:
- TPI > 0: Ridges, hilltops
- TPI ≈ 0: Flat areas, mid-slopes
- TPI < 0: Valleys, depressions

### Parameters
| Parameter    | Type   | Default  | Description             |
| ------------ | ------ | -------- | ----------------------- |
| Input DEM    | Raster | Required | Digital Elevation Model |
| Inner Radius | Int    | 0        | Inner annulus radius    |
| Outer Radius | Int    | 10       | Outer annulus radius    |

### Output
- TPI raster (Float32)

### References
- Weiss, A.D. (2001). "Topographic Position and Landforms Analysis". ESRI User Conference.

---

## 1.7 Feature Detection

**Algorithm**: `FeatureDetectionAlgorithm`  
**Category**: Landform Identification

### Description
Detects geomorphological features such as peaks, ridges, passes, channels, and pits.

### Method
Combines curvature, slope, and relative elevation to classify landforms:
- **Peaks**: High TPI, low curvature
- **Ridges**: High TPI, linear
- **Passes**: Saddle points
- **Channels**: Low TPI, concave curvature
- **Pits**: Lowest TPI, closed depression

### Parameters
| Parameter    | Type   | Default  | Description               |
| ------------ | ------ | -------- | ------------------------- |
| Input DEM    | Raster | Required | Digital Elevation Model   |
| Feature Type | Enum   | All      | Type of feature to detect |

### Output
- Feature classification raster (Int32)

### References
- Wood, J. (1996). *The geomorphological characterisation of digital elevation models*. PhD Thesis, University of Leicester.

---

## 1.8 Hypsometric Analysis

**Algorithm**: `HypsometricAnalysisAlgorithm`  
**Category**: Basin Characterization

### Description
Calculates the hypsometric curve and integral for terrain characterization.

### Mathematical Formula
**Hypsometric Integral**:
```
HI = (mean_elevation - min_elevation) / (max_elevation - min_elevation)
```

**Interpretation**:
- HI > 0.6: Young, uneroded terrain
- HI 0.4-0.6: Mature terrain
- HI < 0.4: Old, highly eroded terrain

### Parameters
| Parameter | Type   | Default  | Description             |
| --------- | ------ | -------- | ----------------------- |
| Input DEM | Raster | Required | Digital Elevation Model |
| Mask      | Raster | Optional | Basin boundary mask     |

### Output
- Hypsometric integral value
- Hypsometric curve data

### References
- Strahler, A.N. (1952). "Hypsometric (area-altitude) analysis of erosional topology". *Geological Society of America Bulletin*, 63(11): 1117-1142.

---

## 1.9 Visibility Analysis

**Algorithm**: `VisibilityAlgorithm`  
**Category**: Viewshed Analysis

### Description
Determines which cells are visible from one or more observer points.

### Method
Line-of-sight analysis using ray tracing:
1. Cast rays from observer to each cell
2. Compare elevation along ray to terrain
3. Mark cells as visible/not visible

### Parameters
| Parameter       | Type   | Default  | Description                 |
| --------------- | ------ | -------- | --------------------------- |
| Input DEM       | Raster | Required | Digital Elevation Model     |
| Observer Points | Vector | Required | Point locations             |
| Observer Height | Float  | 1.75     | Height above ground         |
| Target Height   | Float  | 0        | Height of targets           |
| Max Distance    | Float  | ∞        | Maximum visibility distance |

### Output
- Visibility raster (Int32: 0=not visible, 1=visible)

### References
- Wang, J., Robinson, G.J. & White, K. (1996). "A Fast Solution to Local Viewshed Computation Using Grid-Based Digital Elevation Models". *Photogrammetric Engineering & Remote Sensing*, 62(10): 1157-1164.

---

## 1.10 Directional Analysis

**Algorithm**: `DirectionalAnalysisAlgorithm`  
**Category**: Terrain Orientation

### Description
Analyzes terrain characteristics in specified directions.

### Parameters
| Parameter | Type   | Default  | Description             |
| --------- | ------ | -------- | ----------------------- |
| Input DEM | Raster | Required | Digital Elevation Model |
| Direction | Float  | 0°       | Analysis direction      |

### Output
- Directional derivative raster (Float32)

---

## 1.11 Openness

**Algorithm**: `OpennessAlgorithm`  
**Category**: Terrain Exposure

### Description
Calculates positive and negative openness, measuring terrain exposure and enclosure.

### Mathematical Formula
**Positive Openness** (sky exposure):
```
O+ = mean(zenith_angles) for all directions
```

**Negative Openness** (ground enclosure):
```
O- = mean(nadir_angles) for all directions
```

### Parameters
| Parameter  | Type   | Default  | Description             |
| ---------- | ------ | -------- | ----------------------- |
| Input DEM  | Raster | Required | Digital Elevation Model |
| Radius     | Int    | 10       | Search radius in cells  |
| Directions | Int    | 8        | Number of directions    |

### Output
- Openness raster (Float32, 0-90°)

### References
- Yokoyama, R., Shirasawa, M. & Pike, R.J. (2002). "Visualizing Topography by Openness: A New Application of Image Processing to Digital Elevation Models". *Photogrammetric Engineering & Remote Sensing*, 68(3): 257-265.

---

# 2. Hydrological Analysis

Algorithms for water flow modeling and watershed analysis.

---

## 2.1 Flow Direction

**Algorithm**: `FlowDirectionAlgorithm`  
**Category**: Flow Routing

### Description
Calculates the direction of flow from each cell using D8 (deterministic 8-direction) algorithm.

### D8 Algorithm
```
Flow Direction = direction of steepest descent among 8 neighbors

Drop = (z_center - z_neighbor) / distance
```

**Direction Encoding (ArcGIS/ESRI Standard)**:
```
 32 | 64 | 128
----+----+----
 16 |  X |  1
----+----+----
  8 |  4 |  2
```

### Tie-Breaking
When multiple neighbors have equal steepest descent:
1. Use the first direction found (clockwise from East)
2. For flat areas, flow toward lowest elevation neighbor

### Parameters
| Parameter | Type   | Default  | Description           |
| --------- | ------ | -------- | --------------------- |
| Input DEM | Raster | Required | Filled DEM (no sinks) |

### Output
- Flow Direction raster (Int32): 1, 2, 4, 8, 16, 32, 64, 128
- Symbology: 8-color directional scheme

### References
- Jenson, S.K. & Domingue, J.O. (1988). "Extracting Topographic Structure from Digital Elevation Data for Geographic Information System Analysis". *Photogrammetric Engineering and Remote Sensing*, 54(11): 1593-1600.
- O'Callaghan, J.F. & Mark, D.M. (1984). "The extraction of drainage networks from digital elevation data". *Computer Vision, Graphics and Image Processing*, 28: 323-344.

---

## 2.2 Flow Accumulation

**Algorithm**: `FlowAccumulationAlgorithm`  
**Category**: Flow Routing

### Description
Calculates the accumulated flow to each cell as the number of upstream cells draining through it.

### Algorithm (Recursive)
```
For each cell (i,j):
    Flow_Acc[i,j] = 1 + Σ Flow_Acc[upstream_cells]
```

### Implementation (Efficient)
1. Sort cells by elevation (highest first)
2. Process in order, propagating accumulation downstream

### Parameters
| Parameter            | Type   | Default  | Description              |
| -------------------- | ------ | -------- | ------------------------ |
| Input Flow Direction | Raster | Required | D8 flow direction        |
| Weight Grid          | Raster | Optional | Cell weights (default=1) |

### Output
- Flow Accumulation raster (Float32)
- High values = stream channels

### Interpretation
```
Contributing Area (m²) = Flow_Acc × Cell_Area
```

### References
- Tarboton, D.G. (1997). "A new method for the determination of flow directions and upslope areas in grid digital elevation models". *Water Resources Research*, 33(2): 309-319.

---

## 2.3 Watershed Delineation

**Algorithm**: `WatershedDelineationAlgorithm`  
**Category**: Basin Analysis

### Description
Delineates watershed boundaries for specified pour points.

### Algorithm
1. Identify pour point(s)
2. Trace upstream using flow direction
3. Mark all cells draining to pour point

```
For each cell:
    If cell flows to pour_point (directly or indirectly):
        watershed[cell] = pour_point_id
```

### Parameters
| Parameter        | Type   | Default  | Description                |
| ---------------- | ------ | -------- | -------------------------- |
| Input DEM        | Raster | Required | Digital Elevation Model    |
| Pour Points      | Vector | Required | Outlet point locations     |
| Fill Depressions | Bool   | True     | Fill sinks before analysis |

### Output
- Watershed raster (Int32): unique ID per watershed
- NoData: -9999

### References
- Martz, L.W. & Garbrecht, J. (1992). "Numerical definition of drainage network and subcatchment areas from digital elevation models". *Computers & Geosciences*, 18(6): 747-761.

---

## 2.4 Depression Handling

**Algorithm**: `DepressionHandlingAlgorithm`  
**Category**: DEM Preprocessing

### Description
Fills or breaches depressions (sinks) in DEMs to ensure continuous flow paths.

### Fill Method (Priority-Flood Algorithm)
1. Initialize boundary cells as seed points
2. Process cells in priority order (lowest elevation first)
3. Raise sink cells to their pour point elevation

```
For each sink cell:
    filled_elevation = max(original_elevation, pour_point_elevation + epsilon)
```

### Breach Method
Instead of filling, lower the pour point to create drainage path.

### Parameters
| Parameter | Type   | Default  | Description                     |
| --------- | ------ | -------- | ------------------------------- |
| Input DEM | Raster | Required | Digital Elevation Model         |
| Method    | Enum   | Fill     | Fill or Breach                  |
| Epsilon   | Float  | 0.001    | Minimum gradient for flat areas |

### Output
- Filled DEM raster (Float32)

### References
- Wang, L. & Liu, H. (2006). "An efficient method for identifying and filling surface depressions in digital elevation models for hydrologic analysis and modelling". *International Journal of Geographical Information Science*, 20(2): 193-213.
- Barnes, R., Lehman, C. & Mulla, D. (2014). "Priority-Flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models". *Computers & Geosciences*, 62: 117-127.

---

## 2.5 Flow Indices

**Algorithm**: `FlowIndicesAlgorithm`  
**Category**: Hydrological Indices

### Description
Calculates various hydrological indices including Topographic Wetness Index (TWI) and Stream Power Index (SPI).

### Topographic Wetness Index (TWI)
```
TWI = ln(a / tan(β))
```
Where:
- a = specific catchment area (flow_acc × cellsize)
- β = local slope (radians)

### Stream Power Index (SPI)
```
SPI = a × tan(β)
```

### Parameters
| Parameter         | Type   | Default  | Description             |
| ----------------- | ------ | -------- | ----------------------- |
| Input DEM         | Raster | Required | Digital Elevation Model |
| Flow Accumulation | Raster | Required | Flow accumulation       |
| Index Type        | Enum   | TWI      | TWI, SPI, or Both       |

### Output
- Index raster (Float32)

### Interpretation
- High TWI: Potential saturation zones
- High SPI: High erosion potential

### References
- Beven, K.J. & Kirkby, M.J. (1979). "A physically based, variable contributing area model of basin hydrology". *Hydrological Sciences Bulletin*, 24(1): 43-69.
- Moore, I.D., Grayson, R.B. & Ladson, A.R. (1991). "Digital terrain modelling: a review of hydrological, geomorphological, and biological applications". *Hydrological Processes*, 5(1): 3-30.

---

## 2.6 Flow Routing Extended

**Algorithm**: `FlowRoutingAlgorithm`  
**Category**: Advanced Flow Routing

### Description
Extended flow routing with multiple algorithms including D-Infinity.

### D-Infinity Algorithm
Direction is a continuous angle (0-360°) based on steepest slope among 8 triangular facets.

```
θ = atan2(s2, s1)  where s1, s2 are facet slopes
```

### Parameters
| Parameter | Type   | Default  | Description             |
| --------- | ------ | -------- | ----------------------- |
| Input DEM | Raster | Required | Digital Elevation Model |
| Algorithm | Enum   | D8       | D8, D-Infinity, MFD     |

### Output
- Flow direction raster

### References
- Tarboton, D.G. (1997). "A new method for the determination of flow directions and upslope areas in grid digital elevation models". *Water Resources Research*, 33(2): 309-319.

---

## 2.7 Flow Distance

**Algorithm**: `FlowDistanceAlgorithm`  
**Category**: Flow Path Analysis

### Description
Calculates the distance along flow paths from each cell to the nearest stream or outlet.

### Algorithm
```
Upstream distance: Distance from cell to watershed divide
Downstream distance: Distance from cell to stream/outlet
```

### Parameters
| Parameter      | Type   | Default    | Description       |
| -------------- | ------ | ---------- | ----------------- |
| Flow Direction | Raster | Required   | D8 flow direction |
| Streams        | Raster | Optional   | Stream network    |
| Direction      | Enum   | Downstream | Up or Downstream  |

### Output
- Distance raster (Float32, map units)

---

## 2.8 Basin Analysis

**Algorithm**: `BasinAnalysisAlgorithm`  
**Category**: Basin Characterization

### Description
Delineates all drainage basins from a flow direction raster.

### Algorithm
Identifies all cells that drain to the edge of the DEM as separate basins.

### Output
- Basin raster (Int32): unique ID per basin

---

## 2.9 Flow Path Statistics

**Algorithm**: `FlowPathStatisticsAlgorithm`  
**Category**: Flow Analysis

### Description
Calculates statistics along flow paths.

### Metrics
- Maximum flow path length
- Mean flow path length
- Flow path slope

### Output
- Statistics raster (Float32)

---

## 2.10 Sink Analysis

**Algorithm**: `SinkAnalysisAlgorithm`  
**Category**: DEM Quality

### Description
Identifies and characterizes sinks (depressions) in the DEM.

### Output
- Sink raster (Int32): sink ID
- Sink depth, volume statistics

---

## 2.11 Hydro Enforcement

**Algorithm**: `HydroEnforcementAlgorithm`  
**Category**: DEM Conditioning

### Description
Enforces hydrology by burning streams into the DEM.

### Method
```
DEM_enforced = DEM - (burn_depth × stream_mask)
```

### Parameters
| Parameter  | Type   | Default  | Description             |
| ---------- | ------ | -------- | ----------------------- |
| Input DEM  | Raster | Required | Digital Elevation Model |
| Streams    | Vector | Required | Stream network          |
| Burn Depth | Float  | 10       | Depth to lower streams  |

### Output
- Enforced DEM (Float32)

### References
- Olivera, F. (2001). "Cutting over the hydrological problems in GIS". *Texas A&M University*.

---

## 2.12 Snap Pour Points

**Algorithm**: `SnapPourPointsAlgorithm`  
**Category**: Point Processing

### Description
Snaps pour points to the highest flow accumulation cell within a specified radius.

### Algorithm
```
For each pour point:
    Search within snap_distance
    Move to cell with maximum flow_accumulation
```

### Parameters
| Parameter         | Type   | Default  | Description             |
| ----------------- | ------ | -------- | ----------------------- |
| Pour Points       | Vector | Required | Input points            |
| Flow Accumulation | Raster | Required | Flow accumulation       |
| Snap Distance     | Float  | 100      | Snap radius (map units) |

### Output
- Snapped pour points (Vector)

---

## 2.13 Flow Length

**Algorithm**: `FlowLengthAlgorithm`  
**Category**: Flow Path Analysis

### Description
Calculates the length of flow path from each cell.

### Types
- **Upstream**: Distance to watershed divide
- **Downstream**: Distance to outlet

### Formula
```
Flow_Length = Σ(cell_distances along flow path)

Cell distance = cellsize (cardinal) or cellsize×√2 (diagonal)
```

### Output
- Flow length raster (Float32, map units)

---

## 2.14 DEM Quality

**Algorithm**: `DemQualityAlgorithm`  
**Category**: Quality Assessment

### Description
Assesses DEM quality for hydrological analysis.

### Metrics
- Sink count and density
- Flat area percentage
- Slope statistics

### Output
- Quality report and assessment raster

---

## 2.15 Hillslopes

**Algorithm**: `HillslopesAlgorithm`  
**Category**: Landscape Units

### Description
Delineates hillslope units based on flow to stream channels.

### Output
- Hillslope raster with left/right bank classification

---

# 3. Stream Network Analysis

Algorithms for stream network extraction and characterization.

---

## 3.1 Extract Streams

**Algorithm**: `ExtractStreamsAlgorithm`  
**Category**: Stream Definition

### Description
Extracts stream network from flow accumulation using a threshold.

### Mathematical Formula
```
Stream = 1  if Flow_Accumulation >= Threshold
Stream = 0  otherwise
```

This is equivalent to ArcGIS Raster Calculator:
```
"flow_acc" >= threshold
```

### Parameters
| Parameter         | Type   | Default  | Description               |
| ----------------- | ------ | -------- | ------------------------- |
| Flow Accumulation | Raster | Required | Flow accumulation         |
| Threshold         | Float  | 100      | Minimum contributing area |

### Output
- Binary stream raster (Int32: 0=non-stream, 1=stream)
- NoData: -9999

### Threshold Selection
```
Threshold (cells) = Contributing_Area / Cell_Area

Common values:
- 100 = detailed network
- 1000 = moderate detail
- 5000+ = major channels only
```

### References
- Montgomery, D.R. & Dietrich, W.E. (1988). "Where do channels begin?". *Nature*, 336: 232-234.

---

## 3.2 Stream Ordering

**Algorithm**: `StreamOrderingAlgorithm`  
**Category**: Network Classification

### Description
Assigns hierarchical order values to stream segments.

### Strahler Order
```
- Headwaters (no upstream tributaries) = Order 1
- When two streams of same order meet: Order + 1
- When streams of different orders meet: max(Order)
```

Example:
```
1 + 1 = 2
2 + 1 = 2
2 + 2 = 3
3 + 2 = 3
```

### Shreve Magnitude
```
Magnitude = Sum of all upstream first-order streams
```

### Parameters
| Parameter      | Type   | Default  | Description           |
| -------------- | ------ | -------- | --------------------- |
| Stream Raster  | Raster | Required | Binary stream network |
| Flow Direction | Raster | Required | D8 flow direction     |
| Method         | Enum   | Strahler | Strahler/Shreve       |

### Output
- Stream order raster (Int32)
- Symbology: 10-color gradient (blue→red)

### References
- Strahler, A.N. (1957). "Quantitative analysis of watershed geomorphology". *Transactions of the American Geophysical Union*, 38(6): 913-920.
- Shreve, R.L. (1966). "Statistical law of stream numbers". *Journal of Geology*, 74: 17-37.

---

## 3.3 Stream Link Analysis

**Algorithm**: `StreamLinkAlgorithm`  
**Category**: Network Segmentation

### Description
Assigns unique IDs to stream links (segments between junctions).

### Algorithm
1. Identify junction points (>1 upstream tributary)
2. Trace streams between junctions
3. Assign unique ID to each segment

### Output
- Stream link raster (Int32)

---

## 3.4 Stream Network Analysis

**Algorithm**: `StreamNetworkAnalysisAlgorithm`  
**Category**: Network Metrics

### Description
Analyzes stream network properties.

### Metrics
- Total stream length
- Drainage density
- Stream frequency
- Bifurcation ratio

### Output
- Analysis raster and statistics

---

## 3.5 Vector Stream Network

**Algorithm**: `VectorStreamNetworkAlgorithm`  
**Category**: Vectorization

### Description
Converts stream raster to vector polylines with attributes.

### Attributes
| Field  | Type  | Description    |
| ------ | ----- | -------------- |
| id     | Int   | Segment ID     |
| order  | Int   | Stream order   |
| length | Float | Segment length |

### Output
- Stream polyline shapefile

---

## 3.6 Stream Cleaning

**Algorithm**: `StreamCleaningAlgorithm`  
**Category**: Network Correction

### Description
Removes short stream segments and artifacts.

### Operations
- Remove segments below minimum length
- Clean isolated pixels
- Smooth stream paths

### Parameters
| Parameter     | Type   | Default  | Description            |
| ------------- | ------ | -------- | ---------------------- |
| Stream Raster | Raster | Required | Stream network         |
| Min Length    | Float  | 100      | Minimum segment length |

### Output
- Cleaned stream raster

---

## 3.7 Valley Extraction

**Algorithm**: `ValleyExtractionAlgorithm`  
**Category**: Landform Extraction

### Description
Extracts valley bottoms using geomorphological indices.

### Method
Combines:
- Topographic Position Index (negative)
- Curvature (concave)
- Flow accumulation (high)

### Output
- Valley raster (Int32)

---

## 3.8 Join Stream Links

**Algorithm**: `JoinStreamLinksAlgorithm`  
**Category**: Network Cleaning

### Description
Joins gaps in stream networks by tracing downstream along flow direction.

### Algorithm
1. Find stream endpoints (cells that flow to non-stream)
2. Trace downstream following flow direction
3. If another stream found within threshold, fill the gap
4. Follows natural slope/downstream direction

### Two Modes

**Mode 1: Flow Accumulation (default)**
```
Stream = 1 if flow_acc >= break_down_value
Then join gaps within threshold
```

**Mode 2: Stream Raster**
```
Use existing stream raster (0/1)
Join gaps within threshold
```

### Parameters
| Parameter         | Type   | Default  | Description                    |
| ----------------- | ------ | -------- | ------------------------------ |
| Use Stream Raster | Bool   | False    | Use existing stream raster     |
| Input Raster      | Raster | Required | Flow Acc or Stream Raster      |
| Flow Direction    | Raster | Required | D8 flow direction              |
| Break Down Value  | Float  | 100      | Stream threshold (Mode 1 only) |
| Gap Threshold     | Float  | 5        | Max gap to join (pixels)       |
| Threshold Type    | Enum   | Pixels   | Pixels or Map Units            |

### Output
- Joined stream raster (Int32: 0/1)
- Statistics: endpoints found, gaps joined

---

# 4. Structured Hydrology

Wrapper algorithms for structured hydrological workflows.

| Algorithm                       | Description           |
| ------------------------------- | --------------------- |
| StructBasinAlgorithm            | Basin delineation     |
| StructFillAlgorithm             | Fill depressions      |
| StructFlowAccumulationAlgorithm | Flow accumulation     |
| StructFlowDirectionAlgorithm    | Flow direction        |
| StructFlowDistanceAlgorithm     | Flow distance         |
| StructFlowLengthAlgorithm       | Flow length           |
| StructSinkAlgorithm             | Sink analysis         |
| StructSnapPourPointAlgorithm    | Snap pour points      |
| StructStreamLinkAlgorithm       | Stream links          |
| StructStreamOrderAlgorithm      | Stream ordering       |
| StructStreamToFeatureAlgorithm  | Raster to vector      |
| StructWatershedAlgorithm        | Watershed delineation |

These provide simplified workflows by internally chaining required preprocessing steps.

---

# 5. References

## Books
1. Burrough, P.A. & McDonnell, R.A. (1998). *Principles of Geographical Information Systems*. Oxford University Press.
2. Wilson, J.P. & Gallant, J.C. (2000). *Terrain Analysis: Principles and Applications*. John Wiley & Sons.
3. Hengl, T. & Reuter, H.I. (2009). *Geomorphometry: Concepts, Software, Applications*. Elsevier.

## Key Journal Articles

### Flow Direction & Accumulation
- Jenson, S.K. & Domingue, J.O. (1988). "Extracting Topographic Structure from Digital Elevation Data". *Photogrammetric Engineering and Remote Sensing*, 54(11): 1593-1600.
- O'Callaghan, J.F. & Mark, D.M. (1984). "The extraction of drainage networks from digital elevation data". *Computer Vision, Graphics and Image Processing*, 28: 323-344.
- Tarboton, D.G. (1997). "A new method for the determination of flow directions and upslope areas in grid digital elevation models". *Water Resources Research*, 33(2): 309-319.

### Depression Filling
- Wang, L. & Liu, H. (2006). "An efficient method for identifying and filling surface depressions in digital elevation models". *IJGIS*, 20(2): 193-213.
- Barnes, R., Lehman, C. & Mulla, D. (2014). "Priority-Flood: An optimal depression-filling algorithm". *Computers & Geosciences*, 62: 117-127.

### Stream Networks
- Strahler, A.N. (1957). "Quantitative analysis of watershed geomorphology". *Transactions AGU*, 38(6): 913-920.
- Shreve, R.L. (1966). "Statistical law of stream numbers". *Journal of Geology*, 74: 17-37.

### Terrain Derivatives
- Horn, B.K.P. (1981). "Hill Shading and the Reflectance Map". *Proceedings of the IEEE*, 69(1): 14-47.
- Zevenbergen, L.W. & Thorne, C.R. (1987). "Quantitative analysis of land surface topography". *Earth Surface Processes and Landforms*, 12: 47-56.

### Hydrological Indices
- Beven, K.J. & Kirkby, M.J. (1979). "A physically based, variable contributing area model of basin hydrology". *Hydrological Sciences Bulletin*, 24(1): 43-69.

---

*Document generated for MAS Spatial Analysis Tool QGIS Plugin*  
*Last updated: December 2024*
