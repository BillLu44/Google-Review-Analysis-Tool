# pipeline/emotion.py
# Author: Bill Lu
# Description: Emotion detection using a Hugging Face DeBERTa model for multiclass emotion classification.

import os
import json
from typing import Dict, List
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

# Fetch emotion model identifier
EMOTION_MODEL = _config['models'].get('emotion')
if not EMOTION_MODEL:
    logger.error("Emotion model not set in config.json under 'models.emotion'.")
    raise ValueError("Missing emotion model in config.json")

# Initialize Hugging Face emotion pipeline
try:
    emotion_pipe = pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        tokenizer=EMOTION_MODEL,
        return_all_scores=True
    )
    logger.info(f"Loaded emotion model '{EMOTION_MODEL}'")
except Exception as e:
    logger.error(f"Error initializing emotion pipeline with model '{EMOTION_MODEL}': {e}")
    raise


def detect_emotion(text: str) -> Dict[str, float]:
    """
    Detect emotions in the input text.

    Args:
        text: The review or input text.

    Returns:
        A dict mapping each emotion label to its confidence score.
        Example: {"joy": 0.72, "sadness": 0.10, ...}
    """
    logger.debug("Running emotion detection")
    try:
        results = emotion_pipe(text)
        # Hugging Face returns list of lists for return_all_scores
        scores_list = results[0] if isinstance(results, list) else results
        emotion_scores: Dict[str, float] = {}
        for entry in scores_list:
            label = entry.get('label', '').lower()
            score = float(entry.get('score', 0.0))
            emotion_scores[label] = score
        logger.debug(f"Emotion scores: {emotion_scores}")
        return emotion_scores

    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")
        # Fallback to neutral only
        return {"neutral": 1.0}

# Note: No standalone execution. Import and call detect_emotion() in your orchestrator.
