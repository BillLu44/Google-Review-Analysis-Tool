# /training/training_utils.py
# Author: Bill Lu
# Description: Utility functions for training scripts, including config loading and model directory management.

import os
import json
from pathlib import Path
from datetime import datetime


def load_config() -> dict:
    """
    Loads project-level config.json and adds training-specific settings.

    Returns:
        dict: configuration dict with keys 'models_dir' and optionally 'sentiment_base'.
    """
    # Get project root (parent of training directory)
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config.json'
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config.json: {e}")
    
    # Add models directory path
    config['models_dir'] = str(project_root / 'models')
    
    # Ensure models directory exists
    os.makedirs(config['models_dir'], exist_ok=True)
    
    return config


def ensure_model_dir(models_dir: str, versioned_name: str) -> str:
    """
    Ensures the models directory exists and returns the full path for a versioned model.
    
    Args:
        models_dir: Base models directory
        versioned_name: Model filename with version (e.g., "fusion_v1.pkl")
    
    Returns:
        str: Full path to the versioned model file
    """
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, versioned_name)


def write_model_symlink(models_dir: str, target: str, symlink_name: str) -> None:
    """
    Creates or updates a symlink to point to the target model file.
    
    Args:
        models_dir: Base models directory
        target: Full path to the target model file
        symlink_name: Name of the symlink (e.g., "fusion_latest.pkl")
    """
    symlink_path = os.path.join(models_dir, symlink_name)
    target_basename = os.path.basename(target)
    
    # Remove existing symlink if it exists
    if os.path.exists(symlink_path) or os.path.islink(symlink_path):
        os.unlink(symlink_path)
    
    # Create new symlink (relative path)
    os.symlink(target_basename, symlink_path)
    print(f"Created symlink: {symlink_name} -> {target_basename}")
