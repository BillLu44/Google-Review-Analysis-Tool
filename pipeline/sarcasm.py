# pipeline/sarcasm.py
# Author: Bill Lu
# Description: Sarcasm detection using dnzblgn/Sarcasm-Detection-Customer-Reviews model

import os
import json
from typing import Dict
from transformers import pipeline
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

# Fetch sarcasm model identifier
SARCASM_MODEL = _config['models'].get('sarcasm')
if not SARCASM_MODEL:
    logger.error("Sarcasm model not set in config.json under 'models.sarcasm'.")
    raise ValueError("Missing sarcasm model in config.json")

# Initialize sarcasm pipeline
# initialize with return_all_scores to get every label
try:
    sarcasm_pipe = pipeline(
        "text-classification",
        model=SARCASM_MODEL,
        tokenizer=SARCASM_MODEL,
        return_all_scores=True
    )
    logger.info(f"Loaded sarcasm model '{SARCASM_MODEL}'")
except Exception as e:
    logger.error(f"Error initializing sarcasm pipeline with model '{SARCASM_MODEL}': {e}")
    raise


def detect_sarcasm(text: str) -> Dict[str, float]:
    """
    Detect sarcasm in the input text.

    Args:
        text: Input review text.

    Returns:
        Dict with keys:
          - sarcasm_score: float (confidence of sarcasm)
          - sarcasm_label: str (label predicted by the model)
    """
    logger.debug("Running sarcasm detection")
    try:
        results = sarcasm_pipe(text)
        # sometimes nested: results[0] is list of dicts
        scores = results[0] if isinstance(results, list) and isinstance(results[0], list) else results
        # pick the highest‐confidence label
        best = max(scores, key=lambda x: x.get('score', 0))
        label_raw = best.get('label', '').lower()
        score = float(best.get('score', 0.0))
        if 'sarcastic' in label_raw and 'not' not in label_raw:
            sarcasm_label = 'sarcastic'
            sarcasm_score = score
        else:
            sarcasm_label = 'not_sarcastic'
            sarcasm_score = 1.0 - score
        logger.debug(f"Sarcasm: {label_raw}→{sarcasm_label} ({sarcasm_score:.3f})")
        return {'sarcasm_label': sarcasm_label, 'sarcasm_score': sarcasm_score}
    except Exception as e:
        logger.error(f"Sarcasm detection failed: {e}")
        return {'sarcasm_label': 'not_sarcastic', 'sarcasm_score': 0.0}
