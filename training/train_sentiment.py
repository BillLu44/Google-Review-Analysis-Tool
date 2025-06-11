# /training/train_sentiment.py
# Author: Bill Lu
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
