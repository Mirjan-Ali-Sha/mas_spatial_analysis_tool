# MAS Spatial Analysis Tool

![QGIS](https://img.shields.io/badge/QGIS-3.0+-green.svg)
![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)
![Version](https://img.shields.io/badge/Version-1.1.0-orange.svg)

**Professional-grade terrain analysis toolkit for QGIS** featuring hydrological, geomorphometric, and stream network analysis tools with Numba JIT acceleration.

---

## 🚀 Features

- **45+ Analysis Algorithms** across 4 categories
- **Numba JIT Acceleration** - 10-100x faster processing
- **Industry-Standard** - D8 flow direction encoding (1,2,4,8,16,32,64,128)
- **Automatic Symbology** - Flow direction, stream order, flow accumulation
- **Pure Python** - No binary dependencies
- **Cross-Platform** - Windows, Linux, macOS

---

## 📦 Installation

### Method 1: QGIS Plugin Manager (Recommended)

1. Open QGIS
2. Go to **Plugins → Manage and Install Plugins**
3. Search for **"MAS Spatial Analysis Tool"**
4. Click **Install**

### Method 2: Manual Installation

1. Download the [latest release](https://github.com/Mirjan-Ali-Sha/mas_spatial_analysis_tool/releases/)
2. Extract to QGIS plugins directory:
   - **Windows**: `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
   - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
3. Restart QGIS
4. Enable in **Plugins → Manage and Install Plugins → Installed**

---

## 🛠️ How to Use

1. **Open Processing Toolbox**: `Ctrl+Alt+T` or `Processing → Toolbox`
2. **Expand** "MAS Spatial Analysis Tool"
3. **Select** algorithm category
4. **Double-click** tool to run

### Typical Hydrological Workflow

```
DEM → Fill Depressions → Flow Direction → Flow Accumulation → Extract Streams
```

---

## 📊 Algorithm Categories

### 1. Hydrological Analysis (15 tools)

| Tool                      | Description                                   |
| ------------------------- | --------------------------------------------- |
| **Flow Direction**        | D8 flow direction with standard encoding      |
| **Flow Accumulation**     | Upstream contributing area                    |
| **Watershed Delineation** | Basin boundaries from pour points             |
| **Depression Handling**   | Fill/breach sinks for continuous flow         |
| **Flow Indices**          | TWI (Topographic Wetness), SPI (Stream Power) |
| **Flow Distance**         | Distance to streams/outlets                   |
| **Flow Length**           | Upstream/downstream path lengths              |
| **Basin Analysis**        | Automatic basin delineation                   |
| **Sink Analysis**         | Identify and characterize sinks               |
| **Hydro Enforcement**     | Burn streams into DEM                         |
| **Snap Pour Points**      | Snap to highest accumulation                  |
| **Hillslopes**            | Left/right bank classification                |

### 2. Geomorphometric Analysis (11 tools)

| Tool                     | Description                                |
| ------------------------ | ------------------------------------------ |
| **Hillshade**            | Shaded relief visualization                |
| **Slope**                | Rate of elevation change (degrees/percent) |
| **Aspect**               | Downslope direction (0-360°)               |
| **Curvature**            | Profile, plan, and total curvature         |
| **Roughness**            | Surface irregularity (std dev)             |
| **TPI**                  | Topographic Position Index                 |
| **Openness**             | Sky exposure and enclosure                 |
| **Visibility**           | Viewshed analysis                          |
| **Feature Detection**    | Ridges, valleys, peaks, pits               |
| **Hypsometric Analysis** | Area-altitude relationships                |
| **Directional Analysis** | Terrain orientation metrics                |

### 3. Stream Network Analysis (8 tools)

| Tool                        | Description                                 |
| --------------------------- | ------------------------------------------- |
| **Extract Streams**         | Binary stream raster from flow accumulation |
| **Stream Ordering**         | Strahler and Shreve methods                 |
| **Stream Link Analysis**    | Segment identification and metrics          |
| **Stream Network Analysis** | Network topology and statistics             |
| **Vector Stream Network**   | Raster to polyline conversion               |
| **Stream Cleaning**         | Remove artifacts and short segments         |
| **Valley Extraction**       | Identify valley bottoms                     |
| **Join Stream Links**       | Fill gaps in stream networks                |

### 4. Hydrology - Structured Workflow (12 tools)

Organized workflow tools following standard hydrology analysis patterns:

- Basin, Fill, Flow Direction, Flow Accumulation
- Flow Distance, Flow Length, Sink, Snap Pour Point
- Stream Link, Stream Order, Stream to Feature, Watershed

---

## 💻 Python Usage

```python
from qgis import processing

# Fill depressions
processing.run("mas_spatial_analysis_tool:fill_depressions", {
    'INPUT': '/path/to/dem.tif',
    'OUTPUT': '/path/to/filled.tif'
})

# Flow direction
processing.run("mas_spatial_analysis_tool:flow_direction", {
    'INPUT': '/path/to/filled.tif',
    'OUTPUT': '/path/to/flow_dir.tif'
})

# Flow accumulation
processing.run("mas_spatial_analysis_tool:flow_accumulation", {
    'INPUT_FLOW_DIR': '/path/to/flow_dir.tif',
    'OUTPUT': '/path/to/flow_acc.tif'
})

# Extract streams
processing.run("mas_spatial_analysis_tool:extract_streams", {
    'INPUT': '/path/to/flow_acc.tif',
    'THRESHOLD': 1500,
    'OUTPUT': '/path/to/streams.tif'
})
```

---

## ⚡ Performance

| Feature                 | Benefit                       |
| ----------------------- | ----------------------------- |
| **Numba JIT**           | Compiled loops run at C speed |
| **NumPy Vectorization** | Efficient array operations    |
| **Block Processing**    | Handle large DEMs in chunks   |
| **Caching**             | Faster repeated runs          |

### Tips for Large DEMs

- Use compressed GeoTIFF (LZW)
- Clip to area of interest
- 8GB+ RAM recommended for DEMs >1GB
- Use SSD for temp files

---

## 📋 Technical Specifications

| Specification   | Value                  |
| --------------- | ---------------------- |
| QGIS Version    | 3.0 - 3.99             |
| Python          | 3.x (QGIS bundled)     |
| NoData Value    | -9999                  |
| Flow Direction  | D8 (Standard encoding) |
| Stream Ordering | Strahler, Shreve       |

### D8 Flow Direction Encoding

```
 32 | 64 | 128
----+----+----
 16 |  X |  1
----+----+----
  8 |  4 |  2
```

---

## 🔧 Troubleshooting

### Plugin Doesn't Appear

1. Check QGIS version (3.0+ required)
2. Verify plugin is enabled in Plugin Manager
3. Check for import errors in Python console

### Performance Issues

- Reduce input raster size (clip to AOI)
- Close unnecessary layers
- Increase available RAM
- Use SSD for temp files

---

## 📝 Changelog

### v1.1.0 (2024-12-28)
- **NEW**: Join Stream Links algorithm
- **NEW**: Flow Direction Method selector
- **IMPROVED**: Extract Streams with expression display
- **IMPROVED**: Standardized NoData (-9999)
- **IMPROVED**: Professional symbology

### v1.0.0 (2024-11-15)
- Initial release
- 45+ algorithms
- Numba acceleration

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Mirjan-Ali-Sha/mas_spatial_analysis_tool/issues)
- **Wiki**: [Documentation](https://github.com/Mirjan-Ali-Sha/mas_spatial_analysis_tool/wiki)
- **Email**: mastools.help@gmail.com

---

## 📄 License

GNU General Public License v3.0

---

## 👤 Author

**Mirjan Ali Sha**  
[GitHub](https://github.com/Mirjan-Ali-Sha) | mastools.help@gmail.com
