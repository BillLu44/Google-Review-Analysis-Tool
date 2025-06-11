#!/usr/bin/env python3
# training/generate_sample_data.py
# Author: Bill Lu
# Description: Generate sample training data for fusion model testing

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import uuid # Add this import
from pipeline.logger import get_logger

logger = get_logger(__name__)

# Sample review data with ground truth labels
SAMPLE_DATA = [
    {
        'review_id': str(uuid.uuid4()), # Change to UUID string
        'text': 'Amazing food and excellent service! Highly recommend.',
        'rule_score': 1.0,
        'rule_polarity': 0.8,
        'sentiment_score': 0.9,
        'sentiment_confidence': 0.95,
        'num_pos_aspects': 2,
        'num_neg_aspects': 0,
        'avg_aspect_score': 0.85,
        'avg_aspect_confidence': 0.90,
        'emotion_score': 2.5, # Sum of emotion scores
        'emotion_confidence': 0.41, # Avg of 6 emotion scores (2.5/6)
        'emotion_distribution': 0.3, # Normalized entropy
        'sarcasm_score': 0.0, # 0 for not sarcastic
        'sarcasm_confidence': 0.1, # Low confidence in "not sarcastic"
        'label': 'positive'
    },
    {
        'review_id': str(uuid.uuid4()), # Change to UUID string
        'text': 'Terrible experience, cold food and rude staff.',
        'rule_score': -1.0,
        'rule_polarity': -0.7,
        'sentiment_score': -0.8,
        'sentiment_confidence': 0.92,
        'num_pos_aspects': 0,
        'num_neg_aspects': 2,
        'avg_aspect_score': -0.75,
        'avg_aspect_confidence': 0.88,
        'emotion_score': 1.8,
        'emotion_confidence': 0.3,
        'emotion_distribution': 0.5,
        'sarcasm_score': 0.0,
        'sarcasm_confidence': 0.2,
        'label': 'negative'
    },
    {
        'review_id': str(uuid.uuid4()), # Change to UUID string
        'text': 'Food was okay, nothing special but not bad either.',
        'rule_score': 0.0,
        'rule_polarity': 0.1,
        'sentiment_score': 0.0,
        'sentiment_confidence': 0.6,
        'num_pos_aspects': 1,
        'num_neg_aspects': 1,
        'avg_aspect_score': 0.0,
        'avg_aspect_confidence': 0.7,
        'emotion_score': 0.9,
        'emotion_confidence': 0.15,
        'emotion_distribution': 0.1,
        'sarcasm_score': 0.0,
        'sarcasm_confidence': 0.15,
        'label': 'neutral'
    },
    {
        'review_id': str(uuid.uuid4()), # Change to UUID string
        'text': 'Great atmosphere but the wait time was too long.',
        'rule_score': 0.0, # Example: positive + negative might average to neutral for score
        'rule_polarity': 0.3, # Polarity might still lean one way
        'sentiment_score': 0.2,
        'sentiment_confidence': 0.75,
        'num_pos_aspects': 1,
        'num_neg_aspects': 1,
        'avg_aspect_score': 0.1, # Example: positive aspect stronger
        'avg_aspect_confidence': 0.8,
        'emotion_score': 1.2,
        'emotion_confidence': 0.2,
        'emotion_distribution': 0.6,
        'sarcasm_score': 0.0,
        'sarcasm_confidence': 0.1,
        'label': 'neutral' # Or 'mixed', depending on final labeling scheme
    },
    {
        'review_id': str(uuid.uuid4()), # Change to UUID string
        'text': 'Outstanding! Best restaurant in town.',
        'rule_score': 1.0,
        'rule_polarity': 0.9,
        'sentiment_score': 0.95,
        'sentiment_confidence': 0.98,
        'num_pos_aspects': 1, # Assuming "restaurant" or "town" isn't an aspect here
        'num_neg_aspects': 0,
        'avg_aspect_score': 0.9,
        'avg_aspect_confidence': 0.92,
        'emotion_score': 3.0,
        'emotion_confidence': 0.5,
        'emotion_distribution': 0.2,
        'sarcasm_score': 0.0,
        'sarcasm_confidence': 0.05,
        'label': 'positive'
    }
]


def generate_expanded_data(base_data, num_samples: int = 100):
    """Generate more training samples by adding noise to base data"""
    
    expanded = []
    np.random.seed(42)
    
    feature_keys_to_noise = [
        'rule_polarity', 'sentiment_score', 'sentiment_confidence',
        'avg_aspect_score', 'avg_aspect_confidence',
        'emotion_score', 'emotion_confidence', 'emotion_distribution',
        'sarcasm_confidence'
    ]
    # rule_score, num_pos_aspects, num_neg_aspects, sarcasm_score are often more discrete

    for i in range(num_samples):
        # Pick a random base sample
        base = base_data[i % len(base_data)].copy()
        
        # Add small random noise to continuous features
        noise_level_general = 0.05 # Smaller noise for more stable features
        noise_level_confidence = 0.1 # Confidence can vary a bit more

        for key in feature_keys_to_noise:
            current_noise = noise_level_confidence if 'confidence' in key or 'distribution' in key else noise_level_general
            base[key] += np.random.normal(0, current_noise)

        # Clip values to reasonable ranges
        base['rule_polarity'] = np.clip(base['rule_polarity'], -1, 1)
        base['sentiment_score'] = np.clip(base['sentiment_score'], -1, 1) # Already handled by how it's derived
        base['sentiment_confidence'] = np.clip(base['sentiment_confidence'], 0, 1)
        base['avg_aspect_score'] = np.clip(base['avg_aspect_score'], -1, 1)
        base['avg_aspect_confidence'] = np.clip(base['avg_aspect_confidence'], 0, 1)
        base['emotion_score'] = np.clip(base['emotion_score'], 0, 6) # Max 6 if all 6 emotions are 1.0
        base['emotion_confidence'] = np.clip(base['emotion_confidence'], 0, 1)
        base['emotion_distribution'] = np.clip(base['emotion_distribution'], 0, 1)
        # sarcasm_score is binary (0 or 1), so we don't add noise to it directly here.
        base['sarcasm_confidence'] = np.clip(base['sarcasm_confidence'], 0, 1)
        
        # Ensure discrete scores remain valid if they were part of base
        # (e.g., rule_score, num_aspects, sarcasm_score)
        # For this example, we assume they are reasonably set in SAMPLE_DATA

        # Update ID
        base['review_id'] = str(uuid.uuid4()) # Generate a new UUID string
        base['text'] = f"Generated sample review {base['review_id']}" # Keep text minimal for generated data
        
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