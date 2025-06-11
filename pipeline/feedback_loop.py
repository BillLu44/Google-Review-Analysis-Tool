# pipeline/feedback_loop.py
# Author: Bill Lu
# Description: Logs model predictions and raw inputs for human review and retraining. Appends structured JSON to feedback_logs.jsonl.

import os
import json
from datetime import datetime
from typing import Any, Dict
from pipeline.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Determine feedback log file path
def _get_feedback_log_path() -> str:
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, 'feedback_logs.jsonl')

FEEDBACK_LOG_PATH = _get_feedback_log_path()


def log_feedback(
    review_id: Any,
    text: str,
    signals: Dict[str, Any]
) -> None:
    """
    Append a feedback record for the given review and model signals.

    Args:
        review_id: Identifier of the review (from database or source)
        text: The raw review text
        signals: Dictionary containing all pipeline outputs, e.g. {
            'rule_label': ...,
            'vader_score': ..., 
            'sentiment_label': ...,
            'sentiment_score': ..., 
            'aspects': [...],  # list of aspect dicts
            'emotion': {...},  # dict of emotion scores
            'sarcasm': {...},  # dict of sarcasm label/score
            'fused_label': ..., 
            'fused_confidence': ...
        }

    Writes:
        A JSON line to feedback_logs.jsonl with timestamp.
    """
    record = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'review_id': review_id,
        'text': text,
        'signals': signals
    }
    try:
        with open(FEEDBACK_LOG_PATH, 'a', encoding='utf-8') as logfile:
            logfile.write(json.dumps(record) + '\n')
        logger.debug(f"Logged feedback for review_id={review_id}")
    except Exception as e:
        logger.error(f"Failed to log feedback for review_id={review_id}: {e}")
