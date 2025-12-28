# -*- coding: utf-8 -*-
"""
algorithms/geomorphometric/curvature_algorithms.py
Complete curvature analysis suite
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterEnum,
    QgsProcessingException
)
from ...core.morphometry import MorphometryProcessor
import os


class BaseCurvatureAlgorithm(QgsProcessingAlgorithm):
    """Base class for curvature algorithms."""
    
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    
    def __init__(self):
        super().__init__()
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                'Input DEM'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                f'Output {self.displayName()}'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        
        if input_layer is None:
            raise QgsProcessingException('Invalid input DEM')
        
        try:
            feedback.pushInfo(f'Loading DEM...')
            processor = MorphometryProcessor(input_layer.source())
            
            feedback.pushInfo(f'Calculating {self.displayName().lower()}...')
            feedback.setProgress(30)
            
            curvature = processor.calculate_curvature(
                curvature_type=self.curvature_type
            )
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, curvature)
            processor.close()
            
            feedback.pushInfo(f'{self.displayName()} calculation complete!')
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')


class PlanCurvatureAlgorithm(BaseCurvatureAlgorithm):
    """Plan curvature (horizontal curvature)."""
    
    curvature_type = 'plan'
    
    def createInstance(self):
        return PlanCurvatureAlgorithm()
    
    def name(self):
        return 'plan_curvature'
    
    def displayName(self):
        return 'Plan Curvature'
    
    def shortHelpString(self):
        return """
        Calculate plan curvature (horizontal curvature).
        
        Plan curvature measures the rate of change of aspect,
        indicating convergence (positive) or divergence (negative) of flow.
        
        Positive values: Convergent terrain (ridges)
        Negative values: Divergent terrain (valleys)
        Zero: Flat or linear slope
        """


class ProfileCurvatureAlgorithm(BaseCurvatureAlgorithm):
    """Profile curvature (vertical curvature)."""
    
    curvature_type = 'profile'
    
    def createInstance(self):
        return ProfileCurvatureAlgorithm()
    
    def name(self):
        return 'profile_curvature'
    
    def displayName(self):
        return 'Profile Curvature'
    
    def shortHelpString(self):
        return """
        Calculate profile curvature (vertical curvature).
        
        Profile curvature measures the rate of change of slope,
        indicating flow acceleration (positive) or deceleration (negative).
        
        Positive values: Convex slope (acceleration)
        Negative values: Concave slope (deceleration)
        Zero: Constant slope
        """


class TangentialCurvatureAlgorithm(BaseCurvatureAlgorithm):
    """Tangential curvature."""
    
    curvature_type = 'tangential'
    
    def createInstance(self):
        return TangentialCurvatureAlgorithm()
    
    def name(self):
        return 'tangential_curvature'
    
    def displayName(self):
        return 'Tangential Curvature'
    
    def shortHelpString(self):
        return """
        Calculate tangential curvature.
        
        Tangential curvature describes the geometry of contour lines,
        perpendicular to the slope direction.
        """


class TotalCurvatureAlgorithm(BaseCurvatureAlgorithm):
    """Total curvature (Laplacian)."""
    
    curvature_type = 'total'
    
    def createInstance(self):
        return TotalCurvatureAlgorithm()
    
    def name(self):
        return 'total_curvature'
    
    def displayName(self):
        return 'Total Curvature'
    
    def shortHelpString(self):
        return """
        Calculate total curvature (Laplacian).
        
        Total curvature is the second derivative of elevation,
        measuring overall terrain convexity or concavity.
        """


class MeanCurvatureAlgorithm(BaseCurvatureAlgorithm):
    """Mean curvature."""
    
    curvature_type = 'mean'
    
    def createInstance(self):
        return MeanCurvatureAlgorithm()
    
    def name(self):
        return 'mean_curvature'
    
    def displayName(self):
        return 'Mean Curvature'
    
    def shortHelpString(self):
        return """
        Calculate mean curvature.
        
        Mean curvature is the average of the two principal curvatures,
        describing the overall bending of the surface.
        """


class CurvatureAlgorithm(QgsProcessingAlgorithm):
    """Unified Curvature Analysis Tool."""
    
    INPUT = 'INPUT'
    TYPE = 'TYPE'
    OUTPUT = 'OUTPUT'
    
    TYPES = [
        'Plan', 'Profile', 'Tangential', 'Total', 'Mean', 'Gaussian', 
        'Maximal', 'Minimal', 'Horizontal Excess', 'Vertical Excess', 
        'Difference', 'Accumulation', 'Curvedness', 'Unsphericity', 
        'Rotor', 'Shape Index', 'Ring'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return CurvatureAlgorithm()
    
    def name(self):
        return 'curvature'
    
    def displayName(self):
        return 'Curvature'
    
    def group(self):
        return 'Geomorphometric Analysis'
    
    def groupId(self):
        return 'geomorphometric'
    
    def shortHelpString(self):
        return """
        Calculate various curvature types from a DEM.
        
        Supported types:
        - Plan: Horizontal curvature (convergence/divergence)
        - Profile: Vertical curvature (acceleration/deceleration)
        - Tangential: Geometry of contour lines
        - Total: Laplacian (convexity/concavity)
        - Mean: Average of principal curvatures
        - Gaussian: Product of principal curvatures
        - Maximal/Minimal: Maximum/Minimum principal curvature
        - Shape Index: Scale-invariant measure of shape
        - Curvedness: Magnitude of curvature
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
                self.TYPE,
                'Curvature Type',
                options=self.TYPES,
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Curvature'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
            type_idx = self.parameterAsEnum(parameters, self.TYPE, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            # Map display name to internal name
            type_map = {
                'Plan': 'plan',
                'Profile': 'profile',
                'Tangential': 'tangential',
                'Total': 'total',
                'Mean': 'mean',
                'Gaussian': 'gaussian',
                'Maximal': 'maximal',
                'Minimal': 'minimal',
                'Horizontal Excess': 'horizontal_excess',
                'Vertical Excess': 'vertical_excess',
                'Difference': 'difference',
                'Accumulation': 'accumulation',
                'Curvedness': 'curvedness',
                'Unsphericity': 'unsphericity',
                'Rotor': 'rotor',
                'Shape Index': 'shape_index',
                'Ring': 'ring'
            }
            
            selected_type = self.TYPES[type_idx]
            curv_type = type_map.get(selected_type, 'total')
            
            feedback.pushInfo('Loading DEM...')
            processor = MorphometryProcessor(input_layer.source())
            
            feedback.pushInfo(f'Calculating {curv_type} curvature...')
            feedback.setProgress(30)
            
            curvature = processor.calculate_curvature(curvature_type=curv_type)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            processor.save_raster(output_path, curvature)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
