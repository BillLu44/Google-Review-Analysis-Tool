#!/usr/bin/env python3
# scripts/train_fusion_pipeline.py
# Author: Bill Lu
# Description: End-to-end fusion training pipeline

import os
import argparse
import subprocess
from pathlib import Path
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
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    logger.info(f"Starting fusion training pipeline in {args.mode} mode")
    
    try:
        if args.mode == 'sample':
            # Generate synthetic training data
            logger.info("Step 1: Generating synthetic training data...")
            run_command(f"python training/generate_sample_data.py --num-samples {args.num_samples}")
            train_csv = "data/fusion_train.csv"
            
        elif args.mode == 'feedback':
            # Process feedback logs
            logger.info("Step 1: Processing feedback logs...")
            run_command("python training/process_feedback.py")
            logger.info("Please manually annotate data/fusion_train_template.csv and save as data/fusion_train.csv")
            return
            
        elif args.mode == 'manual':
            if not args.input_csv:
                raise ValueError("--input-csv required for manual mode")
            train_csv = args.input_csv
            
        # Train fusion model
        logger.info("Step 2: Training fusion model...")
        run_command(f"python training/train_fusion.py --input {train_csv} --version {args.version}")
        
        # Validate with test pipeline
        logger.info("Step 3: Validating trained model...")
        run_command("python test_pipeline.py")
        
        logger.info("✅ Fusion training pipeline completed successfully!")
        logger.info(f"Model saved as: models/fusion_{args.version}.pkl")
        logger.info("Symlink updated: models/fusion_latest.pkl")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()