#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
package_plugin.py
Creates a clean ZIP package of the MAS Spatial Analysis Tool plugin for distribution.

Usage:
    python package_plugin.py

Output:
    release/mas_spatial_analysis_tool_<version>.zip
"""

import os
import zipfile
import sys
from datetime import datetime

def get_version_from_metadata(plugin_dir):
    """Extract version number from metadata.txt"""
    metadata_path = os.path.join(plugin_dir, 'metadata.txt')
    version = "0.0.0"
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('version='):
                    version = line.split('=')[1].strip()
                    break
    except Exception as e:
        print(f"Warning: Could not read version from metadata.txt: {e}")
    
    return version

def package_plugin():
    """Create a clean ZIP package of the plugin for distribution."""
    
    # Plugin directory is the current directory
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_name = os.path.basename(plugin_dir)
    
    # Get version from metadata.txt
    version = get_version_from_metadata(plugin_dir)
    
    # Create release folder if it doesn't exist
    release_dir = os.path.join(plugin_dir, 'release')
    os.makedirs(release_dir, exist_ok=True)
    
    # Output zip file with version number
    zip_filename = f"{plugin_name}_{version}.zip"
    zip_path = os.path.join(release_dir, zip_filename)
    
    print("=" * 60)
    print(f"MAS Spatial Analysis Tool - Plugin Packager")
    print("=" * 60)
    print(f"Plugin: {plugin_name}")
    print(f"Version: {version}")
    print(f"Source: {plugin_dir}")
    print(f"Output: release/{zip_filename}")
    print("-" * 60)
    
    # Files and directories to exclude from packaging
    excludes = [
        # Python cache
        '__pycache__',
        '.pytest_cache',
        
        # IDE/Editor
        '.git',
        '.idea',
        '.vscode',
        '.gitignore',
        '.gitattributes',
        
        # Package output
        'release',
        'package_plugin.py',
        
        # Development/Test files
        'compare_flow_dir.py',
        'plugin_structure.txt',
        'verify_all_tools.py',
        'debug_core_imports.py',
        'check_imports.py',
        'test_algorithms.py',
        'test_imports.py',
        'qgis_diagnostic.py',
        'generate_readme_table.py',
        'verify_flow_optimizations.py',
        
        # Documentation/Notes (not needed for distribution)
        'LOG_INSTRUCTIONS.txt',
        'function_list.txt',
        'CHANGELOG.md',
        'CONTRIBUTING.md',
        'ALGORITHMS.md',
        'ALGORITHM_LIST.md',
        
        # Obsolete/Old files
        'mas_algorithm.py',
        'mas_utils.py',
        'structured_tools_new_old.py',
        
        # Old version folders
        'hydrology_struct_old',
        'tests',
        'test',
        'docs',
        
        # Temporary files
        '*.tmp',
        '*.bak',
        '*.orig',
    ]
    
    # File extensions to exclude
    exclude_exts = ['.pyc', '.pyo', '.pyd', '.zip', '.log', '.tmp', '.bak']
    
    # Statistics
    file_count = 0
    total_size = 0
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(plugin_dir):
                # Get relative path for exclusion checking
                rel_root = os.path.relpath(root, plugin_dir)
                
                # Remove excluded directories (modifies in-place for os.walk)
                dirs[:] = [d for d in dirs if d not in excludes and not d.startswith('.')]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, plugin_dir)
                    
                    # Skip excluded files
                    if file in excludes:
                        continue
                    
                    # Skip files in excluded paths
                    if any(rel_path.startswith(ex) or rel_path == ex for ex in excludes):
                        continue
                    
                    # Skip excluded extensions
                    _, ext = os.path.splitext(file)
                    if ext in exclude_exts:
                        continue
                    
                    # Skip hidden files
                    if file.startswith('.'):
                        continue
                    
                    # Add to zip with proper archive name (plugin_name/path)
                    arcname = os.path.join(plugin_name, rel_path)
                    zipf.write(file_path, arcname)
                    
                    # Update stats
                    file_size = os.path.getsize(file_path)
                    file_count += 1
                    total_size += file_size
                    
                    print(f"  + {rel_path}")
        
        # Get zip file size
        zip_size = os.path.getsize(zip_path)
        
        print("-" * 60)
        print(f"Packaging complete!")
        print(f"  Files: {file_count}")
        print(f"  Original size: {total_size / 1024:.1f} KB")
        print(f"  Compressed size: {zip_size / 1024:.1f} KB")
        print(f"  Compression ratio: {(1 - zip_size/total_size)*100:.1f}%")
        print("-" * 60)
        print(f"Output: {zip_path}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == '__main__':
    success = package_plugin()
    sys.exit(0 if success else 1)
