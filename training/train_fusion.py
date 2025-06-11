# /training/train_fusion.py
# Author: Bill Lu
# Description: Train the fusion meta-classifier using structured feature CSV and save a versioned model.

import os
import json
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from training.training_utils import load_config, ensure_model_dir, write_model_symlink
from pipeline.logger import get_logger

logger = get_logger(__name__)


def train_fusion(
    input_csv: str,
    model_dir: str,
    version: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    # Load data
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
    
    X = df[feature_cols]
    y = df['label']

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Label distribution:\n{y.value_counts()}")

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


# /training/train_sentiment.py
# Author: Your Name
# Description: Fine-tune a Hugging Face sentiment model on labeled review data.

import os
import json
import argparse
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from training.training_utils import load_config, ensure_model_dir, write_model_symlink
from pipeline.logger import get_logger

logger = get_logger(__name__)


def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding='max_length')


def train_sentiment(
    input_csv: str,
    base_model: str,
    version: str,
    output_dir: str,
    num_epochs: int = 3,
    per_device_batch_size: int = 8
):
    # Load data
    dataset = load_dataset('csv', data_files={'train': input_csv, 'validation': input_csv}, split=['train', 'validation'])
    train_ds, val_ds = dataset

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    config = AutoConfig.from_pretrained(base_model, num_labels=len(set(train_ds['label'])))
    model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config)

    # Tokenize
    train_ds = train_ds.map(lambda ex: preprocess_function(ex, tokenizer), batched=True)
    val_ds = val_ds.map(lambda ex: preprocess_function(ex, tokenizer), batched=True)
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Metrics
    metric = load_metric('accuracy')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return metric.compute(predictions=preds, references=labels)

    # Training args
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"sentiment_{version}"),
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch'
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save + symlink
    versioned_dir = training_args.output_dir
    write_model_symlink(output_dir, versioned_dir, 'sentiment_latest')
    logger.info(f"Saved fine-tuned sentiment model at {versioned_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune sentiment transformer model.")
    parser.add_argument('--input', required=True, help='Path to sentiment training CSV')
    parser.add_argument('--version', required=True, help='Version tag, e.g. v1')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    config = load_config()
    base_model = config['models']['sentiment_base']
    models_dir = config['models_dir']

    train_sentiment(
        input_csv=args.input,
        base_model=base_model,
        version=args.version,
        output_dir=models_dir,
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()