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
    logger.debug(f"Running sarcasm detection for text: '{text[:200]}...'") # Log truncated text
    try:
        if sarcasm_pipe is None:
            logger.error("Sarcasm pipeline not initialized.")
            return {'sarcasm_label': 'not_sarcastic', 'sarcasm_score': 0, 'sarcasm_confidence': 0.0}

        results = sarcasm_pipe(text)
        
        # The model might return a list of lists for predictions
        # Ensure scores_list is the actual list of prediction dictionaries
        scores_list = results[0] if isinstance(results, list) and results and isinstance(results[0], list) and isinstance(results[0][0], dict) else \
                      results if isinstance(results, list) and results and isinstance(results[0], dict) else \
                      [] # Default to empty list if structure is unexpected
        
        if not scores_list:
            logger.warning(f"No sarcasm scores returned or unexpected format for text: '{text[:100]}...'. Raw results: {results}")
            return {'sarcasm_label': 'not_sarcastic', 'sarcasm_score': 0, 'sarcasm_confidence': 0.0}

        # Find the prediction with the highest score
        best_prediction = max(scores_list, key=lambda x: x.get('score', 0.0))
        
        label_raw_original = best_prediction.get('label', '') # Keep original for logging
        label_raw_processed = label_raw_original.strip().upper() # Processed version for comparison
        confidence = float(best_prediction.get('score', 0.0))

        sarcasm_label_str = 'not_sarcastic' # Default to not sarcastic
        sarcasm_score_val = 0

        # dnzblgn/Sarcasm-Detection-Customer-Reviews model outputs LABEL_1 (sarcastic) and LABEL_0 (not_sarcastic)
        if label_raw_processed == 'LABEL_1':
            sarcasm_label_str = 'sarcastic'
            sarcasm_score_val = 1
        elif label_raw_processed == 'LABEL_0':
            sarcasm_label_str = 'not_sarcastic'
            sarcasm_score_val = 0
        # Fallback for models that might output textual labels (e.g. "Sarcastic", "Not Sarcastic")
        elif label_raw_processed == 'SARCASTIC':
            sarcasm_label_str = 'sarcastic'
            sarcasm_score_val = 1
        elif label_raw_processed in ['NOT_SARCASTIC', 'NOT SARCASTIC']: # Handles "NOT_SARCASTIC" or "NOT SARCASTIC"
            sarcasm_label_str = 'not_sarcastic'
            sarcasm_score_val = 0
        else:
            logger.warning(
                f"Unexpected raw label from sarcasm model: '{label_raw_original}'. "
                f"Processed as '{label_raw_processed}'. Defaulting to not_sarcastic. "
                f"Full model output for this text: {results}"
            )
            # Defaulting to not_sarcastic if label is unrecognized after all checks
        
        logger.debug(
            f"Sarcasm detection: Text='{text[:100]}...', Raw Model Output='{results}', "
            f"Best Prediction='{best_prediction}', Processed Label='{label_raw_processed}', "
            f"Confidence={confidence:.4f} -> Final Label='{sarcasm_label_str}', Score={sarcasm_score_val}"
        )
        
        return {
            'sarcasm_label': sarcasm_label_str,
            'sarcasm_score': sarcasm_score_val,
            'sarcasm_confidence': confidence
        }
    except Exception as e:
        logger.error(f"Sarcasm detection failed for text '{text[:100]}...': {e}", exc_info=True)
        return {'sarcasm_label': 'not_sarcastic', 'sarcasm_score': 0, 'sarcasm_confidence': 0.0}

# For testing
if __name__ == "__main__":
    test_texts = [
        # New test cases added below
        "This artisanal coffee is so complex, it's nearly as flavorful as instant coffee from a gas station.",
        "The sommelier's wine recommendation was so exquisite, it was almost on par with a bottle of two-buck chuck.",
        "This handcrafted dessert is so decadent, it's practically as satisfying as a candy bar from the vending machine.",
        "The thread count on these hotel sheets is so luxurious, I almost felt like I was sleeping on sandpaper.",
        "The 'gourmet' burger here is so refined, it's just a short step away from a fast-food patty.",
        "Kudos to the chef for his innovative 'deconstructed water' – a glass of ice. Bold.",
        "The 'mystery meat' special was a true adventure for the palate. And the stomach.",
        "I've never seen pasta stick together with such artistic flair. Truly a sculptural masterpiece.",
        "Ten points for the bartender's unique take on a 'lightly iced' drink – mostly ice, with a whisper of beverage.",
        "The chef's interpretation of 'al dente' as 'still crunchy' was a brave culinary statement.",
        "The ambiance was so communal, we could hear every detail of the neighboring table's argument. Really felt like part of the family.",
        "Such a cozy spot! My chair was practically in the kitchen, so I got a firsthand view of the culinary... process.",
        "The music volume was perfect for those who enjoy a concert with their meal. Lip-reading skills definitely improved.",
        "They foster such a sense of community by having only one working restroom for the entire establishment.",
        "It's wonderfully 'authentic' how the decor includes cobwebs that have clearly been there for generations.",
        "Loved the speed of the main course! It arrived a full 3 minutes after we finished our appetizers – gave us just enough time to ponder life's mysteries.",
        "The reservation system is top-notch. They confirmed my 8 PM booking for 8:30 PM the next day. Very forward-thinking!",
        "Such proactive cleaning! They started mopping under our table while we were still eating. Really keeps you on your toes.",
        "The attention to detail is amazing; they billed us for items we only thought about ordering.",
        "I appreciate how they value my time; the server vanished for 30 minutes, allowing for deep, uninterrupted conversation.",
        "" # Existing empty string test case
    ]

    if sarcasm_pipe is None:
        print("Sarcasm pipeline not loaded. Cannot run tests.")
    else:
        print(f"\n--- Running Sarcasm Detection Tests ({SARCASM_MODEL}) ---")
        for i, t_text in enumerate(test_texts):
            print(f"\nTest Case #{i+1}")
            sarcasm_result = detect_sarcasm(t_text)
            print(f"  Text: \"{t_text}\"")
            print(f"  Label: {sarcasm_result['sarcasm_label']}, Score: {sarcasm_result['sarcasm_score']}, Confidence: {sarcasm_result['sarcasm_confidence']:.4f}\n")
