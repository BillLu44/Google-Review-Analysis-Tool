# pipeline/fusion.py
# Author: Bill Lu
# Description: Fusion intelligence module combining rule-based, transformer, ABSA, emotion, and sarcasm signals into a final sentiment.

import os
import json
import joblib
import numpy as np
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

# Load the fusion model
try:
    fusion_model = joblib.load(FUSION_MODEL_PATH)
    logger.info(f"Loaded fusion model from '{FUSION_MODEL_PATH}'")
except Exception as e:
    logger.error(f"Failed to load fusion model '{FUSION_MODEL_PATH}': {e}")
    raise


def fuse_signals(signals: dict) -> dict:
    """
    Combine signal dictionary into the fusion model to produce final sentiment.

    Args:
        signals: dict containing numeric features:
            - vader_score
            - sentiment_score
            - num_pos_aspects
            - num_neg_aspects
            - avg_aspect_score
            - emotion_score
            - sarcasm_score
            - rule_confidence (optional)

    Returns:
        dict with:
            - fused_label (str)
            - fused_confidence (float)
    """
    # Define feature order expected by the model
    feature_names = [
        'vader_score',
        'sentiment_score',
        'num_pos_aspects',
        'num_neg_aspects',
        'avg_aspect_score',
        'emotion_score',
        'sarcasm_score'
    ]

    # Build feature vector, default missing to 0
    try:
        feature_vector = [
            float(signals.get(name, 0.0)) for name in feature_names
        ]
    except Exception as e:
        logger.error(f"Error constructing feature vector: {e}")
        feature_vector = [0.0] * len(feature_names)

    X = np.array(feature_vector).reshape(1, -1)

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
            'fused_label': signals.get('rule_label', 'neutral'),
            'fused_confidence': 0.0
        }

# Note: No standalone execution. Import and call fuse_signals() in orchestrator.
