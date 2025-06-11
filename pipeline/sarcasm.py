# pipeline/sarcasm.py
# Author: Your Name
# Description: Sarcasm detection using a Hugging Face DeBERTa model for classifying customer review sarcasm.

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
try:
    sarcasm_pipe = pipeline(
        "text-classification",  # the model returns labels like 'SARCASM' or 'NOT_SARCASM'
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
        # results: list of list if return_all_scores, take first batch
        scores_list = results[0] if isinstance(results, list) else results

        # Find the SARCASM label
        sarcasm_label = None
        sarcasm_score = 0.0
        for entry in scores_list:
            label = entry.get('label', '').lower()
            score = float(entry.get('score', 0.0))
            if 'sarcasm' in label:
                sarcasm_label = label
                sarcasm_score = score
                break

        if sarcasm_label is None and len(scores_list) > 0:
            # Fallback: use highest scoring label
            best = max(scores_list, key=lambda x: x.get('score', 0))
            sarcasm_label = best['label'].lower()
            sarcasm_score = float(best['score'])

        logger.debug(f"Sarcasm detection -> label: {sarcasm_label}, score: {sarcasm_score}")
        return {
            'sarcasm_label': sarcasm_label,
            'sarcasm_score': sarcasm_score
        }

    except Exception as e:
        logger.error(f"Sarcasm detection failed: {e}")
        return {
            'sarcasm_label': 'none',
            'sarcasm_score': 0.0
        }
