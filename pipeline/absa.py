# pipeline/absa.py
# Author: Bill Lu
# Description: Aspect-Based Sentiment Analysis using yangheng/deberta-v3-base-absa-v1.1 model

import os
import json
from typing import List, Dict
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

# Initialize ABSA model from config
try:
    absa_model = _config['models']['absa']  # "yangheng/deberta-v3-base-absa-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(absa_model, use_fast=False)
    absa_classifier = pipeline(
        "text-classification",
        model=absa_model,
        tokenizer=tokenizer,
        top_k=None  # Get all possible labels
    )
    logger.info(f"Loaded ABSA model '{absa_model}' with slow tokenizer")
except Exception as e:
    logger.error(f"Failed to load ABSA model: {e}")
    absa_classifier = None

# Common restaurant/service aspects to look for
COMMON_ASPECTS = [
    "food", "service", "staff", "atmosphere", "ambiance", "price", "value",
    "location", "wait", "portion", "quality", "cleanliness", "menu", "dessert",
    "drink", "wine", "coffee", "parking", "reservation", "noise", "seating",
    # additions so “burger,” “steak,” “fries,” “beer(s),” “sandwich,” etc. are found
    "burger", "steak", "fries", "pizza", "sandwich", "beer", "beers"
]

def extract_aspects(text: str) -> List[str]:
    """Simple keyword-based aspect extraction"""
    text_lower = text.lower()
    found_aspects = [asp for asp in COMMON_ASPECTS if asp in text_lower]
    if not found_aspects:
        return ["overall"]
    # remove singular when plural present
    unique = set(found_aspects)
    for asp in list(unique):
        if asp.endswith("s") and asp[:-1] in unique:
            unique.remove(asp[:-1])
    return list(unique)

def analyze_aspect_sentiment(text: str, aspect: str) -> Dict:
    try:
        # give the ABSA model both text and aspect
        aspect_input = f"{text} [SEP] {aspect}"
        results = absa_classifier(aspect_input)
        logger.info(f"Raw ABSA output for '{aspect}': {results}")
        # unwrap nested lists if necessary
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]
        if not results:
            raise ValueError("empty ABSA output")
        # pick highest‐score label
        best = max(results, key=lambda x: x.get('score', 0))
        label_raw = best.get('label', '')
        score = float(best.get('score', 0))
        # normalize “LABEL_n”
        if label_raw.upper().startswith('LABEL_'):
            idx = int(label_raw.split('_')[1])
            label = {0: 'negative', 1: 'neutral', 2: 'positive'}.get(idx, 'neutral')
        else:
            label = label_raw.lower()
        # convert to signed score
        if label == 'positive':
            sentiment_score = score
        elif label == 'negative':
            sentiment_score = -score
        else:
            sentiment_score = 0.0
        logger.debug(f"ABSA {aspect}: {label_raw}→{label} ({sentiment_score:.3f})")
        return {'aspect': aspect, 'sentiment_label': label, 'sentiment_score': sentiment_score}
    except Exception as e:
        logger.error(f"Error analyzing aspect sentiment for '{aspect}': {e}")
        return {'aspect': aspect, 'sentiment_label': 'neutral', 'sentiment_score': 0.0}

def analyze_absa(text: str) -> List[Dict]:
    """
    Perform Aspect-Based Sentiment Analysis on input text.
    
    Args:
        text (str): Input review text
        
    Returns:
        List[Dict]: List of aspect-sentiment pairs
    """
    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("Empty or invalid text provided to ABSA analysis")
        return []
    
    # Extract aspects mentioned in the text
    aspects = extract_aspects(text)
    
    # Analyze sentiment for each aspect using the dedicated ABSA model
    results = []
    for aspect in aspects:
        aspect_result = analyze_aspect_sentiment(text, aspect)
        results.append(aspect_result)
        logger.debug(f"ABSA: {aspect} -> {aspect_result['sentiment_label']} ({aspect_result['sentiment_score']:.3f})")
    
    logger.info(f"ABSA analysis found {len(results)} aspects")
    return results

# For testing
if __name__ == "__main__":
    test_text = "The food was amazing but the service was terrible and slow."
    result = analyze_absa(test_text)
    print(f"ABSA Result: {result}")
