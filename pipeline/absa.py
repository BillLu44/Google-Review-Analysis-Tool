# pipeline/absa.py
# Author: Bill Lu
# Description: Aspect-Based Sentiment Analysis (ABSA) using a Hugging Face DeBERTa ABSA model and database-driven aspect list.

import os
import json
from typing import List, Dict, Optional
from transformers import pipeline
from pipeline.logger import get_logger
from pipeline.db_utils import fetch_absa_categories

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

# Model identifier for ABSA
ABSA_MODEL = _config['models']['absa']

# Initialize Hugging Face ABSA pipeline
try:
    absa_pipe = pipeline(
        "aspect-based-sentiment-analysis",
        model=ABSA_MODEL,
        tokenizer=ABSA_MODEL
    )
    logger.info(f"Loaded ABSA model '{ABSA_MODEL}'")
except Exception as e:
    logger.error(f"Failed to initialize ABSA pipeline with model '{ABSA_MODEL}': {e}")
    raise


def analyze_absa(
    text: str,
    industry_id: Optional[int] = None
) -> List[Dict]:
    """
    Perform aspect-based sentiment analysis on input text.

    Args:
        text: The review text.
        industry_id: Optional industry filter for which aspects to consider.

    Returns:
        A list of dicts, each containing:
            - category: str, the broader category (e.g., 'Service')
            - aspect: str, the specific aspect term (e.g., 'wait time')
            - sentiment_label: str, 'positive'|'negative'|'neutral'
            - sentiment_score: float, model confidence for this aspect
    """
    # Fetch allowed categories and aspects from DB
    try:
        rows = fetch_absa_categories(industry_id)
    except Exception:
        logger.error("Error fetching ABSA categories; returning empty list.")
        return []

    # Build mapping from aspect term to category
    aspect_to_category = {row['aspect'].lower(): row['category'] for row in rows}
    allowed_aspects = set(aspect_to_category.keys())
    logger.debug(f"Allowed aspects loaded: {allowed_aspects}")

    # Run the ABSA pipeline
    logger.debug("Running ABSA pipeline")
    try:
        raw_results = absa_pipe(text)
    except Exception as e:
        logger.error(f"ABSA pipeline error: {e}")
        return []

    processed: List[Dict] = []
    for entry in raw_results:
        # Extract aspect term
        aspect_term = entry.get('aspect') or entry.get('entity') or entry.get('word')
        if not aspect_term:
            continue
        aspect_norm = aspect_term.strip().lower()

        # Filter to allowed aspects
        if aspect_norm not in allowed_aspects:
            logger.debug(f"Skipping unrecognized aspect '{aspect_term}'")
            continue

        category = aspect_to_category[aspect_norm]
        label = entry.get('label', '').lower()
        score = float(entry.get('score', 0.0))

        processed.append({
            'category': category,
            'aspect': aspect_norm,
            'sentiment_label': label,
            'sentiment_score': score
        })
        logger.debug(f"ABSA detected {category}/{aspect_norm}: {label} ({score})")

    return processed

# Note: No standalone execution. Import and call analyze_absa() in your orchestrator.
