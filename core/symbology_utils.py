# -*- coding: utf-8 -*-
"""
core/symbology_utils.py
Professional-style symbology utilities for hydrology output layers
"""

from qgis.core import (
    QgsRasterLayer,
    QgsColorRampShader,
    QgsSingleBandPseudoColorRenderer,
    QgsRasterShader,
    QgsSingleBandGrayRenderer,
    QgsRasterBandStats,
    QgsStyle
)
from qgis.PyQt.QtGui import QColor


def apply_flow_direction_symbology(layer: QgsRasterLayer):
    """
    Apply professional flow direction symbology.
    Uses categorical colors for 8 D8 direction codes.
    
    D8 codes: E=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128
    """
    if not layer or not layer.isValid():
        return False
    
    from qgis.core import QgsPalettedRasterRenderer
    
    # D8 direction color scheme (professional-style) - using QgsPalettedRasterRenderer.Class
    classes = [
        QgsPalettedRasterRenderer.Class(1, QColor(255, 0, 0), 'E'),        # East - Red
        QgsPalettedRasterRenderer.Class(2, QColor(255, 127, 0), 'SE'),     # Southeast - Orange
        QgsPalettedRasterRenderer.Class(4, QColor(255, 255, 0), 'S'),      # South - Yellow
        QgsPalettedRasterRenderer.Class(8, QColor(127, 255, 0), 'SW'),     # Southwest - Yellow-Green
        QgsPalettedRasterRenderer.Class(16, QColor(0, 255, 0), 'W'),       # West - Green
        QgsPalettedRasterRenderer.Class(32, QColor(0, 255, 255), 'NW'),    # Northwest - Cyan
        QgsPalettedRasterRenderer.Class(64, QColor(0, 127, 255), 'N'),     # North - Light Blue
        QgsPalettedRasterRenderer.Class(128, QColor(127, 0, 255), 'NE'),   # Northeast - Purple
    ]
    
    # Create paletted renderer
    renderer = QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes)
    layer.setRenderer(renderer)
    
    # Force refresh
    layer.emitStyleChanged()
    layer.triggerRepaint()
    
    return True


def apply_flow_accumulation_symbology(layer: QgsRasterLayer, log_scale: bool = True):
    """
    Apply professional flow accumulation symbology.
    
    Uses 2 discrete classes for stream extraction visualization:
    - Class 1: Non-stream cells (light/white)
    - Class 2: Stream cells (dark blue)
    
    Breaking point is calculated as 0.044% of max value
    (based on standard stream extraction ratios).
    """
    if not layer or not layer.isValid():
        return False
    
    # Get statistics from actual data
    provider = layer.dataProvider()
    stats = provider.bandStatistics(1, QgsRasterBandStats.All)
    actual_min = stats.minimumValue
    actual_max = stats.maximumValue
    
    # Calculate dynamic breaking point
    # Based on standard stream extraction ratio: 1500 / 3394716 ≈ 0.000442 (0.044%)
    break_ratio = 1500.0 / 3394716.0
    break_point = actual_max * break_ratio
    
    # Round to nice number
    if break_point > 100:
        break_point = round(break_point, -1)  # Round to nearest 10
    elif break_point > 10:
        break_point = round(break_point)
    
    # Create blue gradient shader with 2 simple classes
    shader = QgsRasterShader()
    color_ramp_shader = QgsColorRampShader()
    
    # Use Discrete classification for range categories
    color_ramp_shader.setColorRampType(QgsColorRampShader.Discrete)
    
    # Define 2 class breaks
    # Very light (nearly white) for non-streams, dark blue for streams
    items = [
        QgsColorRampShader.ColorRampItem(
            break_point, QColor(245, 250, 255), f'≤ {int(break_point):,}'
        ),
        QgsColorRampShader.ColorRampItem(
            actual_max, QColor(8, 48, 107), f'{int(break_point):,} - {int(actual_max):,}'
        ),
    ]
    
    color_ramp_shader.setColorRampItemList(items)
    shader.setRasterShaderFunction(color_ramp_shader)
    
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
    
    # Set min/max to actual data range
    renderer.setClassificationMin(actual_min)
    renderer.setClassificationMax(actual_max)
    
    layer.setRenderer(renderer)
    
    # Force refresh
    layer.emitStyleChanged()
    layer.triggerRepaint()
    
    return True


def apply_stream_order_symbology(layer: QgsRasterLayer):
    """
    Apply symbology for stream order output.
    
    Uses distinct color palette from cool to warm colors:
    - Order 1 (headwaters): Light blue
    - Order 2-3: Green tones
    - Order 4-5: Yellow tones
    - Order 6-7: Orange tones
    - Order 8+: Red tones
    
    Higher order streams (larger rivers) get more prominent colors.
    """
    if not layer or not layer.isValid():
        return False
    
    from qgis.core import QgsPalettedRasterRenderer
    
    # Get actual max order from data
    provider = layer.dataProvider()
    stats = provider.bandStatistics(1, QgsRasterBandStats.All)
    max_order = int(stats.maximumValue)
    
    # Color palette for stream orders (up to 10)
    # Blues for low orders -> Greens -> Yellows -> Oranges -> Reds for high orders
    order_colors = [
        QColor(158, 202, 225),   # Order 1: Light blue (headwaters)
        QColor(66, 146, 198),    # Order 2: Medium blue
        QColor(49, 163, 84),     # Order 3: Green
        QColor(116, 196, 118),   # Order 4: Light green
        QColor(255, 255, 178),   # Order 5: Light yellow
        QColor(254, 204, 92),    # Order 6: Yellow-orange
        QColor(253, 141, 60),    # Order 7: Orange
        QColor(240, 59, 32),     # Order 8: Red-orange
        QColor(189, 0, 38),      # Order 9: Dark red
        QColor(128, 0, 38),      # Order 10+: Deep red
    ]
    
    # Create classes for each order
    classes = []
    for order in range(1, max_order + 1):
        color_idx = min(order - 1, len(order_colors) - 1)
        classes.append(
            QgsPalettedRasterRenderer.Class(order, order_colors[color_idx], f'Order {order}')
        )
    
    # Create paletted renderer
    renderer = QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes)
    layer.setRenderer(renderer)
    
    # Force refresh
    layer.emitStyleChanged()
    layer.triggerRepaint()
    
    
    return True


def apply_watershed_symbology(layer: QgsRasterLayer):
    """
    Apply professional watershed/basin symbology.
    Uses random categorical colors for unique basin IDs.
    """
    if not layer or not layer.isValid():
        return False
    
    # Get unique values (limited for performance)
    stats = layer.dataProvider().bandStatistics(1, QgsRasterBandStats.All)
    min_val = int(stats.minimumValue)
    max_val = int(stats.maximumValue)
    
    # Generate distinct colors
    import random
    random.seed(42)  # Consistent colors
    
    shader = QgsRasterShader()
    color_ramp_shader = QgsColorRampShader()
    color_ramp_shader.setColorRampType(QgsColorRampShader.Exact)
    
    items = []
    num_colors = min(max_val - min_val + 1, 256)  # Limit to 256 colors
    
    for i in range(num_colors):
        val = min_val + i
        if val == 0:
            # NoData/background
            items.append(QgsColorRampShader.ColorRampItem(0, QColor(255, 255, 255, 0), 'NoData'))
        else:
            # Generate HSV color for good distribution
            hue = (i * 137) % 360  # Golden angle for distribution
            color = QColor.fromHsv(hue, 180 + random.randint(0, 75), 180 + random.randint(0, 75))
            items.append(QgsColorRampShader.ColorRampItem(val, color, f'Basin {val}'))
    
    color_ramp_shader.setColorRampItemList(items)
    shader.setRasterShaderFunction(color_ramp_shader)
    
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
    layer.setRenderer(renderer)
    layer.triggerRepaint()
    
    return True


def apply_stream_symbology(layer: QgsRasterLayer):
    """
    Apply professional stream symbology.
    Blue for stream cells, transparent for non-stream.
    """
    if not layer or not layer.isValid():
        return False
    
    shader = QgsRasterShader()
    color_ramp_shader = QgsColorRampShader()
    color_ramp_shader.setColorRampType(QgsColorRampShader.Exact)
    
    items = [
        QgsColorRampShader.ColorRampItem(0, QColor(255, 255, 255, 0), 'No Stream'),
        QgsColorRampShader.ColorRampItem(1, QColor(0, 77, 168), 'Stream'),
    ]
    
    color_ramp_shader.setColorRampItemList(items)
    shader.setRasterShaderFunction(color_ramp_shader)
    
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
    layer.setRenderer(renderer)
    layer.triggerRepaint()
    
    return True


def apply_dem_symbology(layer: QgsRasterLayer):
    """
    Apply terrain/elevation symbology using terrain color ramp.
    """
    if not layer or not layer.isValid():
        return False
    
    stats = layer.dataProvider().bandStatistics(1, QgsRasterBandStats.All)
    min_val = stats.minimumValue
    max_val = stats.maximumValue
    
    shader = QgsRasterShader()
    color_ramp_shader = QgsColorRampShader()
    color_ramp_shader.setColorRampType(QgsColorRampShader.Interpolated)
    
    # Terrain color ramp (green to brown to white)
    range_val = max_val - min_val
    items = [
        QgsColorRampShader.ColorRampItem(min_val, QColor(0, 97, 0), f'{min_val:.0f}'),
        QgsColorRampShader.ColorRampItem(min_val + range_val * 0.25, QColor(148, 176, 64), ''),
        QgsColorRampShader.ColorRampItem(min_val + range_val * 0.5, QColor(222, 214, 163), ''),
        QgsColorRampShader.ColorRampItem(min_val + range_val * 0.75, QColor(168, 119, 69), ''),
        QgsColorRampShader.ColorRampItem(max_val, QColor(255, 255, 255), f'{max_val:.0f}'),
    ]
    
    color_ramp_shader.setColorRampItemList(items)
    shader.setRasterShaderFunction(color_ramp_shader)
    
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
    layer.setRenderer(renderer)
    layer.triggerRepaint()
    
    return True


def apply_slope_symbology(layer: QgsRasterLayer):
    """
    Apply professional slope symbology.
    
    Uses classified color ramp from green (flat) to red (steep):
    - Green: 0-5° (flat/gentle)
    - Yellow-Green: 5-15° (moderate)
    - Yellow: 15-25° (moderately steep)
    - Orange: 25-35° (steep)
    - Red: 35-45° (very steep)
    - Dark Red: 45°+ (extremely steep)
    """
    if not layer or not layer.isValid():
        return False
    
    # Get statistics from actual data
    provider = layer.dataProvider()
    stats = provider.bandStatistics(1, QgsRasterBandStats.All)
    actual_min = max(0, stats.minimumValue)  # Slope can't be negative
    actual_max = stats.maximumValue
    
    # Create shader with classified colors
    shader = QgsRasterShader()
    color_ramp_shader = QgsColorRampShader()
    color_ramp_shader.setColorRampType(QgsColorRampShader.Interpolated)
    
    # Professional slope classification (9 classes)
    # Using equal intervals based on the actual data range
    range_val = actual_max - actual_min
    num_classes = 9
    
    # Color scheme: dark green -> light green -> yellow -> orange -> red -> brown
    colors = [
        QColor(38, 115, 0),      # Dark green
        QColor(56, 168, 0),      # Green
        QColor(121, 192, 0),     # Yellow-green
        QColor(210, 225, 0),     # Yellow
        QColor(255, 255, 0),     # Bright yellow
        QColor(255, 191, 0),     # Orange-yellow
        QColor(255, 127, 0),     # Orange
        QColor(230, 76, 0),      # Red-orange
        QColor(168, 56, 0),      # Brown-red
    ]
    
    items = []
    for i in range(num_classes):
        val = actual_min + (range_val * (i + 1) / num_classes)
        prev_val = actual_min + (range_val * i / num_classes) if i > 0 else actual_min
        label = f'{prev_val:.2f} - {val:.2f}'
        items.append(QgsColorRampShader.ColorRampItem(val, colors[i], label))
    
    color_ramp_shader.setColorRampItemList(items)
    shader.setRasterShaderFunction(color_ramp_shader)
    
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
    renderer.setClassificationMin(actual_min)
    renderer.setClassificationMax(actual_max)
    
    layer.setRenderer(renderer)
    layer.emitStyleChanged()
    layer.triggerRepaint()
    
    return True


def apply_aspect_symbology(layer: QgsRasterLayer):
    """
    Apply professional aspect symbology.
    
    Uses categorical colors for 8 cardinal/intercardinal directions:
    - Flat (-1): Gray
    - North (0-22.5, 337.5-360): Red
    - Northeast (22.5-67.5): Orange
    - East (67.5-112.5): Yellow
    - Southeast (112.5-157.5): Green
    - South (157.5-202.5): Cyan
    - Southwest (202.5-247.5): Blue
    - West (247.5-292.5): Purple
    - Northwest (292.5-337.5): Magenta
    """
    if not layer or not layer.isValid():
        return False
    
    # Create shader with discrete classification
    shader = QgsRasterShader()
    color_ramp_shader = QgsColorRampShader()
    color_ramp_shader.setColorRampType(QgsColorRampShader.Discrete)
    
    # Professional aspect color scheme
    # Each item defines the upper bound of the category
    items = [
        QgsColorRampShader.ColorRampItem(-0.5, QColor(178, 178, 178), 'Flat (-1)'),
        QgsColorRampShader.ColorRampItem(22.5, QColor(255, 0, 0), 'North (0-22.5)'),
        QgsColorRampShader.ColorRampItem(67.5, QColor(255, 170, 0), 'Northeast (22.5-67.5)'),
        QgsColorRampShader.ColorRampItem(112.5, QColor(255, 255, 0), 'East (67.5-112.5)'),
        QgsColorRampShader.ColorRampItem(157.5, QColor(85, 255, 0), 'Southeast (112.5-157.5)'),
        QgsColorRampShader.ColorRampItem(202.5, QColor(0, 255, 255), 'South (157.5-202.5)'),
        QgsColorRampShader.ColorRampItem(247.5, QColor(0, 85, 255), 'Southwest (202.5-247.5)'),
        QgsColorRampShader.ColorRampItem(292.5, QColor(170, 0, 255), 'West (247.5-292.5)'),
        QgsColorRampShader.ColorRampItem(337.5, QColor(255, 0, 255), 'Northwest (292.5-337.5)'),
        QgsColorRampShader.ColorRampItem(360, QColor(255, 0, 0), 'North (337.5-360)'),
    ]
    
    color_ramp_shader.setColorRampItemList(items)
    shader.setRasterShaderFunction(color_ramp_shader)
    
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
    renderer.setClassificationMin(-1)
    renderer.setClassificationMax(360)
    
    layer.setRenderer(renderer)
    layer.emitStyleChanged()
    layer.triggerRepaint()
    
    return True

