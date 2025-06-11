#!/usr/bin/env python3
# scripts/train_fusion_pipeline.py
# Author: Bill Lu
# Description: End-to-end fusion training pipeline

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.logger import get_logger

logger = get_logger(__name__)


def run_command(cmd, check=True):
    """Run shell command and log output"""
    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        logger.info(f"STDOUT: {result.stdout}")
    if result.stderr:
        logger.warning(f"STDERR: {result.stderr}")
    
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {cmd}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="End-to-end fusion training pipeline")
    parser.add_argument('--mode', choices=['sample', 'feedback', 'manual'], default='sample',
                       help='Training data source: sample=generate synthetic, feedback=use logged feedback, manual=use existing CSV')
    parser.add_argument('--input-csv', help='Manual CSV file path (for manual mode)')
    parser.add_argument('--version', default='v1', help='Model version tag')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of synthetic samples (sample mode)')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of folds for cross-validation (passed to train_fusion.py)') # ADDED
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    logger.info(f"Starting fusion training pipeline in {args.mode} mode")
    
    try:
        train_csv_path = "" # Initialize to avoid linter warning if mode is feedback initially
        if args.mode == 'sample':
            # Generate synthetic training data
            logger.info("Step 1: Generating synthetic training data...")
            # Assuming generate_sample_data.py outputs to a known location like data/fusion_train.csv
            run_command(f"python training/generate_sample_data.py --num-samples {args.num_samples} --output data/fusion_train.csv")
            train_csv_path = "data/fusion_train.csv"
            logger.info(f"Generated data at {train_csv_path}. Please ensure it is manually labeled before proceeding if necessary.")
            # Note: The original script for 'sample' mode implies manual labeling might be needed.
            # If generate_sample_data.py creates labels, this note can be adjusted.
            
        elif args.mode == 'feedback':
            # Process feedback logs
            logger.info("Step 1: Processing feedback logs...")
            # Assuming process_feedback.py outputs to data/fusion_train_template.csv or similar
            run_command("python training/process_feedback.py --output data/feedback_processed.csv")
            train_csv_path = "data/feedback_processed.csv"
            logger.info(f"Processed feedback logs to {train_csv_path}.")
            logger.info("IMPORTANT: Please manually open this CSV and fill in the 'label' column with ground truth before proceeding with training.")
            # For a fully automated pipeline, you might pause here or have a separate labeling step.
            # For now, let's assume the user will label it and then re-run with 'manual' mode or point to the labeled file.
            # Or, if the pipeline should proceed, the user must ensure the CSV is labeled.
            # For this example, we'll assume the user will handle labeling and then use manual mode.
            logger.info(f"Please use 'manual' mode with the labeled CSV: --mode manual --input-csv {train_csv_path} (after labeling)")
            return # Exiting here as labeling is a manual step for feedback mode.
            
        elif args.mode == 'manual':
            if not args.input_csv:
                logger.error("--input-csv required for manual mode")
                raise ValueError("--input-csv required for manual mode")
            train_csv_path = args.input_csv
            if not Path(train_csv_path).exists():
                logger.error(f"Input CSV not found: {train_csv_path}")
                raise FileNotFoundError(f"Input CSV not found: {train_csv_path}")
            
        # Train fusion model (only if train_csv_path is set and valid)
        if train_csv_path:
            logger.info(f"Step 2: Training fusion model using {train_csv_path}...")
            run_command(f"python training/train_fusion.py --input {train_csv_path} --version {args.version} --n-splits {args.n_splits}") # MODIFIED: Pass n_splits
            
            # Validate with test pipeline
            logger.info("Step 3: Validating trained model...")
            run_command("python test_pipeline.py")
            
            logger.info("✅ Fusion training pipeline completed successfully!")
            logger.info(f"Model saved as: models/fusion_{args.version}.pkl")
            logger.info("Symlink updated: models/fusion_latest.pkl")
        else:
            logger.warning("No training CSV path determined, skipping training and validation.")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True) # Added exc_info for more details
        # Consider re-raising if this script is part of a larger automated process
        # raise


if __name__ == '__main__':
    main()