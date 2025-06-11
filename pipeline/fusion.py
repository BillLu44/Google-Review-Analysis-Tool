# pipeline/fusion.py
# Author: Bill Lu
# Description: Fusion intelligence module combining rule-based, transformer, ABSA, emotion, and sarcasm signals into a final sentiment.

import os
import json
import joblib
import numpy as np
import pandas as pd
from pipeline.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
try:
    with open(CONFIG_PATH, 'r') as cf:
        _config = json.load(cf)
except Exception as e:
    logger.error(f"Failed to load config.json: {e}")
    raise

# Fusion model path
FUSION_MODEL_PATH = _config['models'].get('fusion')
if not FUSION_MODEL_PATH:
    logger.error("Fusion model path not set in config.json under 'models.fusion'.")
    raise ValueError("Missing fusion model in config.json")

# Try to load fusion model - don't crash if it doesn't exist yet
fusion_model = None
if os.path.exists(FUSION_MODEL_PATH):
    try:
        fusion_model = joblib.load(FUSION_MODEL_PATH)
        logger.info(f"Loaded fusion model from '{FUSION_MODEL_PATH}'")
    except Exception as e:
        logger.error(f"Failed to load fusion model '{FUSION_MODEL_PATH}': {e}")
        fusion_model = None
else:
    logger.warning(f"Fusion model not found at '{FUSION_MODEL_PATH}'")
    logger.info("To create your first model, run: python scripts/train_fusion_pipeline.py --mode sample --version v1")

def fuse_signals(signals: dict) -> dict:
    """
    Combine signal dictionary into the fusion model to produce final sentiment.

    Args:
        signals: dict containing numeric features:
            - rule_score
            - rule_polarity
            - sentiment_score
            - sentiment_confidence
            - num_pos_aspects
            - num_neg_aspects
            - avg_aspect_score
            - avg_aspect_confidence
            - emotion_score
            - emotion_confidence
            - emotion_distribution
            - sarcasm_score
            - sarcasm_confidence

    Returns:
        dict with:
            - fused_label (str)
            - fused_confidence (float)
    """
    
    # If no trained model exists, return placeholder
    if fusion_model is None:
        logger.warning("No fusion model available - run training first!")
        return {
            'fused_label': 'neutral',
            'fused_confidence': 0.0
        }
    
    # Define feature order expected by the model
    feature_names = [
        'rule_score', # Integer score: -1 (Negative), 0 (Neutral), 1 (Positive)
        'rule_polarity', # Continuous score: Average of VADER compound & TextBlob polarity, range -1 to 1
        'sentiment_score', # Continuous score: Derived from overall sentiment model's confidence, signed, range -1 to 1
        'sentiment_confidence', # Raw confidence from overall sentiment model [0, 1]
        'num_pos_aspects', # Count of aspects with positive sentiment
        'num_neg_aspects', # Count of aspects with negative sentiment
        'avg_aspect_score', # Mean of signed sentiment_scores from individual aspects, range -1 to 1
        'avg_aspect_confidence', # Average of confidence scores across all detected aspects [0, 1]
        'emotion_score',         # Total emotional weight (sum of 6 emotion category scores) [0, 6]
        'emotion_confidence',    # Average confidence across all 6 emotion categories [0, 1]
        'emotion_distribution',  # Normalized entropy of emotion scores, indicating spread [0, 1]
        'sarcasm_score', # Binary score: 0 (not sarcastic), 1 (sarcastic)
        'sarcasm_confidence' # Confidence score from sarcasm model [0, 1]
    ]

    # Build feature vector, default missing to 0
    try:
        feature_vector = [
            float(signals.get(name, 0.0)) for name in feature_names
        ]
    except Exception as e:
        logger.error(f"Error constructing feature vector: {e}")
        feature_vector = [0.0] * len(feature_names)

    # Create DataFrame with proper feature names (fixes sklearn warning)
    X = pd.DataFrame([feature_vector], columns=feature_names)

    try:
        # Predict probabilities and classes
        probs = fusion_model.predict_proba(X)[0]
        classes = fusion_model.classes_
        # Select best
        idx = np.argmax(probs)
        fused_label = classes[idx]
        fused_confidence = float(probs[idx])
        logger.debug(f"Fusion features: {dict(zip(feature_names, feature_vector))}")
        logger.info(f"Fusion result -> {fused_label} ({fused_confidence:.3f})")
        return {
            'fused_label': fused_label,
            'fused_confidence': fused_confidence
        }
    except Exception as e:
        logger.error(f"Fusion prediction failed: {e}")
        return {
            'fused_label': 'neutral',
            'fused_confidence': 0.0
        }

# For testing
if __name__ == "__main__":
    test_signals = {
        'rule_score': 1.0,             # Example: 1 for positive
        'rule_polarity': 0.75,         # Example: continuous polarity score
        'sentiment_score': 0.8,
        'sentiment_confidence': 0.95,
        'num_pos_aspects': 2,
        'num_neg_aspects': 0,
        'avg_aspect_score': 0.7,
        'avg_aspect_confidence': 0.85,
        'emotion_score': 1.5,
        'emotion_confidence': 0.25,
        'emotion_distribution': 0.4,
        'sarcasm_score': 0.1,          # Assuming 0 for not sarcastic, 1 for sarcastic
        'sarcasm_confidence': 0.6
    }
    result = fuse_signals(test_signals)
    print(f"Fusion Result: {result}")
