# /training/training_utils.py
# Author: Bill Lu
# Description: Utility functions for training scripts, including config loading and model directory management.

import os
import json
from datetime import datetime


def load_config() -> dict:
    """
    Loads project-level config.json and adds training-specific settings.

    Returns:
        dict: configuration dict with keys 'models_dir' and optionally 'sentiment_base'.
    """
    root = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(root, 'config.json')
    with open(config_path, 'r') as cf:
        cfg = json.load(cf)

    # Model storage directory
    models_dir = cfg.get('training', {}).get('models_dir', os.path.join(root, 'models'))
    os.makedirs(models_dir, exist_ok=True)
    cfg['models_dir'] = models_dir

    # Base sentiment model for fine-tuning
    cfg['models']['sentiment_base'] = cfg['models']['sentiment']
    return cfg


def ensure_model_dir(models_dir: str, versioned_name: str) -> str:
    """
    Ensures the path for the versioned model file exists.

    Returns the full path for the versioned model.
    """
    path = os.path.join(models_dir, versioned_name)
    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    return path


def write_model_symlink(models_dir: str, target: str, symlink_name: str) -> None:
    """
    Creates or updates a symlink pointing to the target model.
    """
    symlink_path = os.path.join(models_dir, symlink_name)
    try:
        if os.path.islink(symlink_path) or os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(os.path.basename(target), symlink_path)
    except OSError:
        # On Windows without symlink privileges, fallback to copy
        import shutil
        shutil.copytree(target, os.path.join(models_dir, symlink_name), dirs_exist_ok=True)
