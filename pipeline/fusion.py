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

def extract_advanced_features(signals: dict, text: str) -> dict:
    """Extract additional contextual features"""
    
    advanced_signals = signals.copy()
    
    # Add context-aware features
    advanced_signals['hidden_sentiment'] = detect_hidden_sentiment(text)
    advanced_signals['sarcasm_context_score'] = detect_sarcastic_context(text)
    advanced_signals['contradiction_score'] = detect_contradictions(text)
    advanced_signals['aspect_balance'] = calculate_aspect_balance(signals)
    
    return advanced_signals

NEGATIVE_CONTEXT_PHRASES = [
    "you get what you pay for",  # implies low quality
    "matches the low prices",    # implies poor quality
    "reminds me of.*if.*lost",   # backhanded comparison
    "would be.*if they fixed",   # conditional criticism
    "tries really hard",        # implies failure despite effort
    "certainly present",         # damning with faint praise
    "adequate",                  # backhanded compliment
]

BACKHANDED_COMPLIMENTS = [
    "tries really hard",
    "surprisingly good",
    "better than expected",
    "not bad",
    "decent for the price",
    "certainly present",         # damning with faint praise
    "adequate",                  # backhanded compliment
    "good effort",
    "interesting take"
]

def detect_hidden_sentiment(text: str) -> float:
    """Detect hidden negative sentiment in seemingly neutral phrases"""
    text_lower = text.lower()
    
    for phrase in NEGATIVE_CONTEXT_PHRASES:
        if phrase in text_lower:
            return -0.6  # Moderately negative
            
    return 0.0  # No hidden sentiment detected

def detect_sarcastic_context(text: str) -> float:
    """Detect sarcastic context indicators. Returns a score indicating presence of sarcastic cues."""
    sarcasm_indicators = [
        "oh great", "just perfect", "wonderful", "brilliant", "amazing how", "fantastic", "super",
        "exactly what i needed", "so glad", "how exciting", "love it when", "my favorite",
        "couldn't be happier", "as if", "yeah right", "sure", "whatever", "obviously",
        "said no one ever", "i'm sure", "clearly", "of course", "big surprise", "oh joy",
        "can't wait", "thrilled"
    ]
    exclamation_count = text.count('!')
    question_mark_after_positive = any(phrase + "?" in text.lower() for phrase in ["great", "perfect", "wonderful", "fantastic"])
    
    text_lower = text.lower()
    has_sarcasm_indicator = any(phrase in text_lower for phrase in sarcasm_indicators)
    
    # More weight if multiple indicators or strong punctuation cues
    score = 0.0
    if has_sarcasm_indicator:
        score += 0.6
    if exclamation_count > 2: # Excessive exclamations can indicate sarcasm
        score += 0.3
    if question_mark_after_positive: # e.g., "Wonderful service?"
        score += 0.4
    
    return min(1.0, score) # Cap score at 1.0

def calculate_aspect_balance(signals: dict) -> float:
    """Calculate balance between positive and negative aspects"""
    pos_aspects = signals.get('num_pos_aspects', 0)
    neg_aspects = signals.get('num_neg_aspects', 0)
    total_aspects = pos_aspects + neg_aspects
    
    if total_aspects == 0:
        return 0.0
    
    balance = abs(pos_aspects - neg_aspects) / total_aspects
    return 1.0 - balance  # Higher score means more balanced

def detect_contradictions(text: str) -> float:
    """Detect contradictory sentiment within text"""
    positive_words = ["amazing", "wonderful", "great", "excellent", "fantastic"]
    negative_context = ["but", "however", "unfortunately", "disappointed"]
    
    has_positive = any(word in text.lower() for word in positive_words)
    has_negative_context = any(word in text.lower() for word in negative_context)
    
    return 1.0 if (has_positive and has_negative_context) else 0.0

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
