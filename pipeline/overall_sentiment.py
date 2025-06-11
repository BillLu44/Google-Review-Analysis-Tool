# pipeline/overall_sentiment.py
# Author: Bill Lu
# Description: Transformer-based overall sentiment analysis using DeBERTa.

import os
import json
from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
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

# Fetch model identifier from config
SENTIMENT_MODEL = _config['models']['sentiment']

# Initialize sentiment pipeline
try:
    sentiment_pipe = pipeline(
        'sentiment-analysis',
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL
    )
    logger.info(f"Loaded sentiment model '{SENTIMENT_MODEL}'")
except Exception as e:
    logger.error(f"Error initializing sentiment pipeline with model '{SENTIMENT_MODEL}': {e}")
    raise


def analyze_sentiment(text: str) -> dict:
    """
    Analyze overall sentiment via transformer.

    Args:
        text: Input review text.

    Returns:
        A dict with:
          - label (str): 'positive', 'negative', or 'neutral'
          - score (float): confidence of the prediction
    """
    logger.debug("Starting transformer-based sentiment analysis")
    try:
        results = sentiment_pipe(text)
        # results is a list of dicts (one per input) if batched, otherwise returns list
        if isinstance(results, list) and len(results) > 0:
            res = results[0]
        else:
            res = results
        label = res.get('label', '').lower()
        score = float(res.get('score', 0.0))
        logger.debug(f"Sentiment result -> label: {label}, score: {score}")
        return {
            'sentiment_label': label,
            'sentiment_score': score
        }
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        # On failure, return neutral fallback
        return {
            'sentiment_label': 'neutral',
            'sentiment_score': 0.0
        }
