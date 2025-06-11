#!/usr/bin/env python3
# training/process_feedback.py
# Author: Bill Lu
# Description: Convert feedback_logs.jsonl to CSV format for fusion training

import json
import pandas as pd
import argparse
from pathlib import Path
from pipeline.logger import get_logger

logger = get_logger(__name__)


def process_feedback_logs(
    jsonl_path: str,
    output_csv: str,
    min_records: int = 10
):
    """
    Convert feedback_logs.jsonl to training CSV format.
    
    Args:
        jsonl_path: Path to feedback_logs.jsonl
        output_csv: Output CSV path
        min_records: Minimum records required to proceed
    """
    
    if not Path(jsonl_path).exists():
        logger.error(f"Feedback log file not found: {jsonl_path}")
        return
    
    records = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                signals = record.get('signals', {})
                
                # Extract features for fusion training
                row = {
                    'review_id': record.get('review_id'),
                    'text': record.get('text', ''),
                    'rule_score': signals.get('rule_score', 0.0),
                    'sentiment_score': signals.get('sentiment_score', 0.0),
                    'num_pos_aspects': signals.get('num_pos_aspects', 0),
                    'num_neg_aspects': signals.get('num_neg_aspects', 0),
                    'avg_aspect_score': signals.get('avg_aspect_score', 0.0),
                    'emotion_score': signals.get('emotion_score', 0.0),
                    'sarcasm_score': signals.get('sarcasm_score', 0.0),
                    'fused_label': signals.get('fused_label', 'neutral'),
                    'fused_confidence': signals.get('fused_confidence', 0.0),
                    'label': ''  # To be filled manually
                }
                records.append(row)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON on line {line_num}: {e}")
                continue
    
    if len(records) < min_records:
        logger.warning(f"Only {len(records)} records found, minimum {min_records} required")
        return
    
    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    logger.info(f"Processed {len(records)} feedback records to {output_csv}")
    logger.info("Next steps:")
    logger.info("1. Open the CSV file and fill in the 'label' column with ground truth")
    logger.info("2. Run: python training/train_fusion.py --input data/fusion_train.csv --version v1")


def main():
    parser = argparse.ArgumentParser(description="Process feedback logs for fusion training")
    parser.add_argument('--input', default='data/feedback_logs.jsonl', help='Input JSONL file')
    parser.add_argument('--output', default='data/fusion_train_template.csv', help='Output CSV file')
    parser.add_argument('--min-records', type=int, default=10, help='Minimum records required')
    args = parser.parse_args()
    
    process_feedback_logs(args.input, args.output, args.min_records)


if __name__ == '__main__':
    main()