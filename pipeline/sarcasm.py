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
          - sarcasm_label: str ('sarcastic' or 'not_sarcastic')
          - sarcasm_score: int (1 if sarcastic, 0 if not_sarcastic)
          - sarcasm_confidence: float (model's confidence in the prediction, 0.0 to 1.0)
    """
    logger.debug(f"Running sarcasm detection for text: '{text}'")
    try:
        if sarcasm_pipe is None:
            logger.error("Sarcasm pipeline not initialized.")
            return {'sarcasm_label': 'not_sarcastic', 'sarcasm_score': 0, 'sarcasm_confidence': 0.0}

        results = sarcasm_pipe(text)
        
        # The model might return a list of lists for predictions
        scores_list = results[0] if isinstance(results, list) and results and isinstance(results[0], list) else results
        
        if not scores_list:
            logger.warning(f"No sarcasm scores returned for text: '{text}'")
            return {'sarcasm_label': 'not_sarcastic', 'sarcasm_score': 0, 'sarcasm_confidence': 0.0}

        # Find the prediction with the highest score
        best_prediction = max(scores_list, key=lambda x: x.get('score', 0.0))
        
        label_raw = best_prediction.get('label', '').lower()
        confidence = float(best_prediction.get('score', 0.0))

        sarcasm_label_str = 'not_sarcastic'
        sarcasm_score_val = 0 # Default to not sarcastic

        # The dnzblgn/Sarcasm-Detection-Customer-Reviews model typically has 'Sarcastic' and 'Not Sarcastic' labels
        if 'sarcastic' in label_raw and 'not' not in label_raw: # Check for 'sarcastic' and not 'not sarcastic'
            sarcasm_label_str = 'sarcastic'
            sarcasm_score_val = 1
        
        logger.debug(f"Sarcasm detection: Raw Label='{label_raw}', Confidence={confidence:.3f} -> Final Label='{sarcasm_label_str}', Score={sarcasm_score_val}")
        
        return {
            'sarcasm_label': sarcasm_label_str,
            'sarcasm_score': sarcasm_score_val,
            'sarcasm_confidence': confidence
        }
    except Exception as e:
        logger.error(f"Sarcasm detection failed for text '{text}': {e}", exc_info=True)
        return {'sarcasm_label': 'not_sarcastic', 'sarcasm_score': 0, 'sarcasm_confidence': 0.0}

# For testing
if __name__ == "__main__":
    test_texts = [
        "Oh, another meeting? How exciting.", # Sarcastic
        "I love it when my code breaks right before a deadline.", # Sarcastic
        "This is a genuinely good product, I'm very happy with it.", # Not sarcastic
        "The weather is so lovely today, said no one ever during this blizzard.", # Sarcastic
        "I'm really enjoying this book." # Not sarcastic
    ]

    if sarcasm_pipe is None:
        print("Sarcasm pipeline not loaded. Cannot run tests.")
    else:
        for t_text in test_texts:
            sarcasm_result = detect_sarcasm(t_text)
            print(f"Text: \"{t_text}\"")
            print(f"  Label: {sarcasm_result['sarcasm_label']}, Score: {sarcasm_result['sarcasm_score']}, Confidence: {sarcasm_result['sarcasm_confidence']:.4f}\n")
