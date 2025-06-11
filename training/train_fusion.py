# /training/train_fusion.py
# Author: Bill Lu
# Description: Train the fusion meta-classifier using structured feature CSV and save a versioned model.

import os
import sys
import json
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold # MODIFIED: Import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score # MODIFIED: Added accuracy_score
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
    # test_size: float = 0.2, # MODIFIED: test_size is not directly used for CV evaluation
    random_state: int = 42,
    n_splits: int = 5 # MODIFIED: Added n_splits for cross-validation
):
    # Load data
    logger.info(f"Loading training data from {input_csv}")
    df = pd.read_csv(input_csv)
    
    if 'label' not in df.columns:
        raise ValueError("Input CSV must contain 'label' column as target.")
    
    feature_cols = [
        'rule_score', 'rule_polarity', 'sentiment_score', 'sentiment_confidence',
        'num_pos_aspects', 'num_neg_aspects', 'avg_aspect_score', 'avg_aspect_confidence',
        'emotion_score', 'emotion_confidence', 'emotion_distribution',
        'sarcasm_score', 'sarcasm_confidence'
    ]
    
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = clean_numeric_columns(df, feature_cols)
    
    X = df[feature_cols]
    y = df['label']

    logger.info(f"Full dataset shape for CV: {X.shape}")
    logger.info(f"Label distribution in full dataset:\n{y.value_counts(normalize=True)}")
    
    if X.isna().any().any():
        logger.warning("Found NaN values in features before CV, filling with 0")
        X = X.fillna(0.0)

    # --- Stratified K-Fold Cross-Validation for Performance Estimation ---
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_accuracies = []
    fold_reports = []

    logger.info(f"Starting {n_splits}-fold stratified cross-validation...")
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        logger.info(f"  Fold {fold+1}/{n_splits}")
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index] 
        
        # Initialize your model for this fold
        cv_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            random_state=random_state,
            class_weight='balanced'
        )
        cv_model.fit(X_train_fold, y_train_fold)
        
        y_pred_fold = cv_model.predict(X_val_fold)
        
        fold_accuracy = accuracy_score(y_val_fold, y_pred_fold)
        fold_accuracies.append(fold_accuracy)
        report = classification_report(y_val_fold, y_pred_fold, output_dict=True)
        fold_reports.append(report)
        logger.info(f"  Fold {fold+1} Accuracy: {fold_accuracy:.4f}")
        # logger.info(f"  Fold {fold+1} Classification Report:\n{classification_report(y_val_fold, y_pred_fold)}")

    logger.info("--- Cross-Validation Summary ---")
    logger.info(f"Fold Accuracies: {[round(acc, 4) for acc in fold_accuracies]}")
    logger.info(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    
    # You can also average precision/recall/F1 from fold_reports if needed
    # For example, to get average F1-score for 'positive' class:
    # avg_f1_positive = np.mean([fr['positive']['f1-score'] for fr in fold_reports if 'positive' in fr])
    # logger.info(f"Mean CV F1-score (Positive): {avg_f1_positive:.4f}")


    # --- Train Final Model on ALL Data ---
    logger.info("Training final fusion model on all available data...")
    final_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        min_samples_split=5,
        random_state=random_state,
        class_weight='balanced'
    )
    final_model.fit(X, y) # Train on the entire dataset X, y
    
    # Feature importance from the final model
    importances = final_model.feature_importances_
    feature_importance = dict(zip(feature_cols, importances))
    logger.info("Feature importances (from final model):")
    for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {feat}: {imp:.4f}")

    # Save the final model
    versioned_model_name = f"fusion_{version}.pkl" # Use the original versioned name
    versioned_model_path = ensure_model_dir(model_dir, versioned_model_name) # ensure_model_dir now returns the full path
    
    logger.info(f"Saving final model to {versioned_model_path}")
    joblib.dump(final_model, versioned_model_path)
    
    # Update symlink to 'fusion_latest.pkl'
    # write_model_symlink expects the directory and the target filename, not the full path to target
    write_model_symlink(model_dir, versioned_model_name, 'fusion_latest.pkl') 
    
    logger.info(f"Fusion model training complete. Model version: {version}")
    logger.info(f"Final model saved to: {versioned_model_path}")
    logger.info(f"Mean Cross-Validation Accuracy: {np.mean(fold_accuracies):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train fusion meta-classifier.")
    parser.add_argument('--input', required=True, help='Path to fusion training CSV')
    parser.add_argument('--version', required=True, help='Version tag, e.g. v1')
    # parser.add_argument('--test-size', type=float, default=0.2) # Not used for CV evaluation
    parser.add_argument('--n-splits', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()

    config = load_config()
    model_dir = config['models_dir']

    train_fusion(
        input_csv=args.input,
        model_dir=model_dir,
        version=args.version,
        # test_size=args.test_size, # Not passed
        n_splits=args.n_splits
    )

if __name__ == '__main__':
    main()