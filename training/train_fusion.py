# /training/train_fusion.py
# Author: Bill Lu
# Description: Train the fusion meta-classifier using structured feature CSV and save a versioned model.

import os
import sys
import json
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.training_utils import load_config, ensure_model_dir, write_model_symlink
from pipeline.logger import get_logger

logger = get_logger(__name__)


def clean_numeric_columns(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Clean and convert numeric columns, handling Unicode minus signs"""
    df_clean = df.copy()
    
    for col in numeric_cols:
        if col in df_clean.columns:
            # Convert to string first, replace Unicode minus with ASCII minus
            df_clean[col] = df_clean[col].astype(str).str.replace('−', '-', regex=False)
            # Convert to numeric, handling any remaining issues
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Check for any NaN values created by conversion
            nan_count = df_clean[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Column '{col}' has {nan_count} values that couldn't be converted to numeric")
                # Fill NaN with 0
                df_clean[col] = df_clean[col].fillna(0.0)
    
    return df_clean


def train_fusion(
    input_csv: str,
    model_dir: str,
    version: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    # Load data
    logger.info(f"Loading training data from {input_csv}")
    df = pd.read_csv(input_csv)
    
    if 'label' not in df.columns:
        raise ValueError("Input CSV must contain 'label' column as target.")
    
    # Define feature columns expected by fusion model
    feature_cols = [
        'rule_score',
        'sentiment_score', 
        'num_pos_aspects',
        'num_neg_aspects',
        'avg_aspect_score',
        'emotion_score',
        'sarcasm_score'
    ]
    
    # Check required columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean numeric columns (handle Unicode minus signs)
    numeric_cols = ['rule_score', 'sentiment_score', 'avg_aspect_score', 'emotion_score', 'sarcasm_score']
    df = clean_numeric_columns(df, numeric_cols)
    
    X = df[feature_cols]
    y = df['label']

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Label distribution:\n{y.value_counts()}")
    
    # Check for any remaining NaN values
    if X.isna().any().any():
        logger.warning("Found NaN values in features, filling with 0")
        X = X.fillna(0.0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Training fusion model on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # Initialize and train
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        min_samples_split=5,
        random_state=random_state,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)

    # Evaluate
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)
    
    logger.info("Training set performance:")
    logger.info(classification_report(y_train, train_preds))
    
    logger.info("Test set performance:")
    logger.info(classification_report(y_test, test_preds))
    
    # Feature importance
    importances = clf.feature_importances_
    feature_importance = dict(zip(feature_cols, importances))
    logger.info("Feature importances:")
    for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {feat}: {imp:.4f}")

    # Save model
    versioned_path = ensure_model_dir(model_dir, f"fusion_{version}.pkl")
    joblib.dump(clf, versioned_path)
    write_model_symlink(model_dir, versioned_path, 'fusion_latest.pkl')
    logger.info(f"Saved fusion model version '{version}' to {versioned_path}")
    
    return clf


def main():
    parser = argparse.ArgumentParser(description="Train fusion meta-classifier.")
    parser.add_argument('--input', required=True, help='Path to fusion training CSV')
    parser.add_argument('--version', required=True, help='Version tag, e.g. v1')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    config = load_config()
    model_dir = config['models_dir']

    train_fusion(
        input_csv=args.input,
        model_dir=model_dir,
        version=args.version,
        test_size=args.test_size
    )

if __name__ == '__main__':
    main()