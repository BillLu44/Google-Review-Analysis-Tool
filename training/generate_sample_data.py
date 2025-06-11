#!/usr/bin/env python3
# training/generate_sample_data.py
# Author: Bill Lu
# Description: Generate sample training data for fusion model testing

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from pipeline.logger import get_logger

logger = get_logger(__name__)

# Sample review data with ground truth labels
SAMPLE_DATA = [
    {
        'review_id': 1,
        'text': 'Amazing food and excellent service! Highly recommend.',
        'rule_score': 0.8,
        'sentiment_score': 0.9,
        'num_pos_aspects': 2,
        'num_neg_aspects': 0,
        'avg_aspect_score': 0.85,
        'emotion_score': 0.7,
        'sarcasm_score': 0.1,
        'label': 'positive'
    },
    {
        'review_id': 2,
        'text': 'Terrible experience, cold food and rude staff.',
        'rule_score': -0.7,
        'sentiment_score': -0.8,
        'num_pos_aspects': 0,
        'num_neg_aspects': 2,
        'avg_aspect_score': -0.75,
        'emotion_score': 0.2,
        'sarcasm_score': 0.1,
        'label': 'negative'
    },
    {
        'review_id': 3,
        'text': 'Food was okay, nothing special but not bad either.',
        'rule_score': 0.1,
        'sentiment_score': 0.0,
        'num_pos_aspects': 1,
        'num_neg_aspects': 1,
        'avg_aspect_score': 0.0,
        'emotion_score': 0.4,
        'sarcasm_score': 0.0,
        'label': 'neutral'
    },
    {
        'review_id': 4,
        'text': 'Great atmosphere but the wait time was too long.',
        'rule_score': 0.3,
        'sentiment_score': 0.2,
        'num_pos_aspects': 1,
        'num_neg_aspects': 1,
        'avg_aspect_score': 0.1,
        'emotion_score': 0.5,
        'sarcasm_score': 0.0,
        'label': 'neutral'
    },
    {
        'review_id': 5,
        'text': 'Outstanding! Best restaurant in town.',
        'rule_score': 0.9,
        'sentiment_score': 0.95,
        'num_pos_aspects': 1,
        'num_neg_aspects': 0,
        'avg_aspect_score': 0.9,
        'emotion_score': 0.8,
        'sarcasm_score': 0.05,
        'label': 'positive'
    }
]


def generate_expanded_data(base_data, num_samples: int = 100):
    """Generate more training samples by adding noise to base data"""
    
    expanded = []
    np.random.seed(42)
    
    for i in range(num_samples):
        # Pick a random base sample
        base = base_data[i % len(base_data)].copy()
        
        # Add small random noise to features
        noise_level = 0.1
        base['rule_score'] += np.random.normal(0, noise_level)
        base['sentiment_score'] += np.random.normal(0, noise_level)
        base['avg_aspect_score'] += np.random.normal(0, noise_level)
        base['emotion_score'] += np.random.normal(0, noise_level * 0.5)
        base['sarcasm_score'] += np.random.normal(0, noise_level * 0.5)
        
        # Clip values to reasonable ranges
        base['rule_score'] = np.clip(base['rule_score'], -1, 1)
        base['sentiment_score'] = np.clip(base['sentiment_score'], -1, 1)
        base['avg_aspect_score'] = np.clip(base['avg_aspect_score'], -1, 1)
        base['emotion_score'] = np.clip(base['emotion_score'], 0, 1)
        base['sarcasm_score'] = np.clip(base['sarcasm_score'], 0, 1)
        
        # Update ID
        base['review_id'] = i + 1
        base['text'] = f"Sample review {i + 1}"
        
        expanded.append(base)
    
    return expanded


def main():
    parser = argparse.ArgumentParser(description="Generate sample fusion training data")
    parser.add_argument('--output', default='data/fusion_train.csv', help='Output CSV file')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to generate')
    args = parser.parse_args()
    
    # Create data directory
    Path(args.output).parent.mkdir(exist_ok=True)
    
    # Generate data
    if args.num_samples <= len(SAMPLE_DATA):
        data = SAMPLE_DATA[:args.num_samples]
    else:
        data = generate_expanded_data(SAMPLE_DATA, args.num_samples)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    
    logger.info(f"Generated {len(data)} training samples in {args.output}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    logger.info("Ready to train fusion model!")


if __name__ == '__main__':
    main()