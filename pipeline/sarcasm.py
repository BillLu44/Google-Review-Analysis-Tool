# pipeline/sarcasm.py
# Author: Bill Lu
# Description: Sarcasm detection using dnzblgn/Sarcasm-Detection-Customer-Reviews model

import os
import json
from typing import Dict
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

# Fetch sarcasm model identifier
SARCASM_MODEL = _config['models'].get('sarcasm')
if not SARCASM_MODEL:
    logger.error("Sarcasm model not set in config.json under 'models.sarcasm'.")
    raise ValueError("Missing sarcasm model in config.json")

# Initialize sarcasm pipeline
# load slow (python) tokenizer to avoid auto‐conversion to fast
tokenizer = AutoTokenizer.from_pretrained(SARCASM_MODEL, use_fast=False)
sarcasm_pipe = pipeline(
    "text-classification",
    model=SARCASM_MODEL,
    tokenizer=tokenizer,
    top_k=None
)
logger.info(f"Loaded sarcasm model '{SARCASM_MODEL}' with slow tokenizer")


def detect_sarcasm(text: str) -> Dict[str, float]:
    """
    Detect sarcasm in the input text.

    Args:
        text: Input review text.

    Returns:
        Dict with keys:
          - sarcasm_score: float (positive=P(sarcastic), negative=P(not sarcastic))
          - sarcasm_label: str (label predicted by the model)
    """
    logger.debug("Running sarcasm detection")
    try:
        results = sarcasm_pipe(text)
        scores = results[0] if isinstance(results, list) and isinstance(results[0], list) else results
        best = max(scores, key=lambda x: x.get('score', 0))
        label_raw = best.get('label', '').lower()
        score = float(best.get('score', 0.0))

        if 'sarcastic' in label_raw and 'not' not in label_raw:
            sarcasm_label = 'sarcastic'
        else:
            sarcasm_label = 'not_sarcastic'

        # signed confidence: +score if sarcastic, –score if not
        sarcasm_score = score if sarcasm_label == 'sarcastic' else -score

        logger.debug(f"Sarcasm: {label_raw}→{sarcasm_label} ({sarcasm_score:.3f})")
        return {'sarcasm_label': sarcasm_label, 'sarcasm_score': sarcasm_score}
    except Exception as e:
        logger.error(f"Sarcasm detection failed: {e}")
        return {'sarcasm_label': 'not_sarcastic', 'sarcasm_score': 0.0}
