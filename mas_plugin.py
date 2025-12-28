# -*- coding: utf-8 -*-

import os
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsApplication

from processing.core.ProcessingConfig import ProcessingConfig
from mas_spatial_analysis_tool.mas_provider import MasGeospatialProvider


class MasGeospatialPlugin:
    """QGIS Plugin Implementation for MAS Spatial Analysis Tool."""

    def __init__(self, iface):
        """Initialize the plugin.
        
        Args:
            iface: A QGIS interface instance
        """
        self.iface = iface
        self.provider = None
        self.plugin_dir = os.path.dirname(__file__)

    def initProcessing(self):
        """Initialize Processing provider."""
        self.provider = MasGeospatialProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        try:
            self.initProcessing()
            
            # Check for Numba (optional performance enhancement)
            self._check_numba_dependency()
            
        except Exception as e:
            from qgis.core import QgsMessageLog, Qgis
            QgsMessageLog.logMessage(f"MAS Plugin Init Error: {str(e)}", "MAS Spatial Analysis Tool", Qgis.Critical)
            self.iface.messageBar().pushMessage("MAS Plugin", f"Init Error: {str(e)}", level=Qgis.Critical)
            raise e
    
    def _check_numba_dependency(self):
        """Check if Numba is available and offer installation if not."""
        try:
            import numba
            from qgis.core import QgsMessageLog, Qgis
            QgsMessageLog.logMessage(f"Numba {numba.__version__} is available - using accelerated algorithms", 
                                    "MAS Spatial Analysis Tool", Qgis.Info)
        except ImportError:
            # Numba not available - offer installation
            from qgis.PyQt.QtCore import QTimer
            # Delay the dialog slightly so QGIS finishes loading
            QTimer.singleShot(2000, self._show_numba_dialog)
    
    def _show_numba_dialog(self):
        """Show Numba installation dialog (delayed)."""
        try:
            from .dependency_installer import check_and_install_numba
            check_and_install_numba(self.iface)
        except Exception as e:
            from qgis.core import QgsMessageLog, Qgis
            QgsMessageLog.logMessage(f"Dependency check error: {str(e)}", "MAS Spatial Analysis Tool", Qgis.Warning)

    def unload(self):
        """Remove the plugin menu item and icon from QGIS GUI."""
        try:
            if self.provider:
                QgsApplication.processingRegistry().removeProvider(self.provider)
        except RuntimeError:
            # Provider might have been deleted already
            pass

    def tr(self, message):
        """Get the translation for a string using Qt translation API."""
        return QCoreApplication.translate('MasGeospatialPlugin', message)

