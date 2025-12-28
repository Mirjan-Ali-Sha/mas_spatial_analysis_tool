# -*- coding: utf-8 -*-
"""
MAS Spatial Analysis Tool Plugin
Advanced hydrological and geomorphometric analysis for QGIS

Author: MAS (Based on WhiteboxTools integration)
License: GPL v2+
"""

__author__ = 'MAS'
__date__ = '2025-11-30'
__copyright__ = '(C) 2025, MAS'
__version__ = '1.0.0'


def classFactory(iface):
    """Load MasGeospatialPlugin class from file mas_plugin."""
    from .mas_plugin import MasGeospatialPlugin
    return MasGeospatialPlugin(iface)
