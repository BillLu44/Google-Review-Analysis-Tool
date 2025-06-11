# pipeline/overall_sentiment.py
# Author: Bill Lu
# Description: Transformer-based sentiment analysis using DeBERTa model

import os
import json
from transformers import pipeline, AutoTokenizer
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

# Initialize sentiment model from config
try:
    sentiment_model = _config['models']['sentiment']
    # load slow tokenizer to avoid byte‐fallback warning
    tokenizer = AutoTokenizer.from_pretrained(sentiment_model, use_fast=False)
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=sentiment_model,
        tokenizer=tokenizer,
        top_k=1
    )
    logger.info(f"Loaded sentiment model: {sentiment_model}")
except Exception as e:
    logger.error(f"Failed to load sentiment model: {e}")
    sentiment_pipe = None

def analyze_sentiment(text) -> dict:
    """
    Analyze sentiment using DeBERTa transformer model from config.
    
    Args:
        text: Input text (should be string)
        
    Returns:
        dict with 'sentiment' (int: -1, 0, 1) and 'confidence_score' (float: 0-1)
    """
    if sentiment_pipe is None:
        logger.warning("No sentiment model available")
        return {
            'sentiment': 0,  # neutral
            'confidence_score': 0.0
        }
    
    # Handle edge cases and convert to string
    if text is None:
        logger.warning("None text provided to sentiment analysis")
        return {
            'sentiment': 0,  # neutral
            'confidence_score': 0.0
        }
    
    # Convert to string regardless of input type
    if not isinstance(text, str):
        text = str(text)
        logger.debug(f"Converted non-string input to string: {text}")
    
    # Check if text is meaningful after conversion
    if not text.strip():
        logger.warning("Text is empty after stripping")
        return {
            'sentiment': 0,  # neutral
            'confidence_score': 0.0
        }
    
    try:
        result = sentiment_pipe(text)
        
        # Handle nested list structure from the model
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list) and len(result[0]) > 0:
                result = result[0][0]  # Get first result from nested list
            else:
                result = result[0]  # Get first result from simple list
        
        label_raw = result['label']
        raw_confidence = float(result['score'])
        
        # normalize any “LABEL_n” into a text label first
        text_label = 'neutral' # default
        if isinstance(label_raw, str) and label_raw.upper().startswith('LABEL_'):
            idx = int(label_raw.split('_')[1])
            text_label = {0: 'negative', 1: 'neutral', 2: 'positive'}.get(idx, 'neutral')
        else:
            text_label = str(label_raw).lower()
        
        # Convert text label to integer sentiment
        sentiment_int = 0 # neutral
        if text_label == 'positive':
            sentiment_int = 1
        elif text_label == 'negative':
            sentiment_int = -1
        
        logger.debug(f"Sentiment: {text_label} (int: {sentiment_int}), Confidence: {raw_confidence:.3f}")
        return {
            'sentiment': sentiment_int,
            'confidence_score': raw_confidence
        }
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {
            'sentiment': 0,  # neutral
            'confidence_score': 0.0
        }

# For testing
if __name__ == "__main__":
    test_text = "I love this restaurant!"
    result = analyze_sentiment(test_text)
    print(f"Sentiment Result: {result}")
