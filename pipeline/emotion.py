# pipeline/emotion.py
# Author: Bill Lu
# Description: Emotion detection using a Hugging Face DeBERTa model for multiclass emotion classification.

import os
import json
from typing import Dict, List
from transformers import pipeline, AutoTokenizer
from pipeline.logger import get_logger
from pipeline.preprocessing import preprocess_text
import re # Added import

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
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL, use_fast=False)
    emotion_pipe = pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        tokenizer=tokenizer,
        top_k=None
    )
    logger.info(f"Loaded emotion model '{EMOTION_MODEL}' with slow tokenizer")
except Exception as e:
    logger.error(f"Error initializing emotion pipeline with model '{EMOTION_MODEL}': {e}")
    raise


def get_emotion_clauses(original_text: str) -> List[str]:
    """
    Splits text into smaller clauses for emotion analysis.
    Uses spaCy sentences, and if a single sentence is returned
    that contains common conjunctions, it attempts a split.
    """
    processed_text_info = preprocess_text(original_text)
    spacy_sentences = processed_text_info.get('sentences', [])

    # Use spaCy sentences if they provide multiple segments
    if len(spacy_sentences) > 1:
        clauses = [s.strip() for s in spacy_sentences if s.strip()]
        if clauses:
            logger.debug(f"Using {len(clauses)} spaCy sentences for emotion analysis: {clauses}")
            return clauses

    # If spaCy returns 0 or 1 sentence, or only empty sentences,
    # attempt custom splitting on the original text.
    text_to_split = original_text.strip()
    final_clauses = []

    # Try splitting by ". " first to get potential sentences if spaCy missed them or returned one long string.
    # This helps handle cases like "Sentence one. Sentence two but with a comma."
    potential_sentences = [s.strip() for s in text_to_split.split(". ") if s.strip()]
    if not potential_sentences and text_to_split: # If no ". " split, use the whole text as one potential sentence
        potential_sentences = [text_to_split]

    for psent in potential_sentences:
        # For each potential sentence, try splitting by ", but "
        # This is a common pattern that spaCy's default sentence segmenter might not break.
        if ", but " in psent:
            parts = psent.split(", but ", 1)
            final_clauses.extend([p.strip() for p in parts if p.strip()])
        else:
            final_clauses.append(psent) # No ", but " found, add as is

    # Fallback if still no clauses or all are empty
    if not final_clauses or not any(c.strip() for c in final_clauses):
        logger.warning(f"Could not split text into meaningful clauses: '{original_text}'. Using original text as one clause.")
        return [original_text.strip()] if original_text.strip() else []

    # Filter out very short clauses (e.g., less than 2 words) that might result from splitting.
    # This helps avoid feeding tiny, meaningless fragments to the emotion model.
    result_clauses = [c for c in final_clauses if len(c.split()) >= 2]
    
    if not result_clauses: # If all clauses became too short, fallback to original text
        logger.warning(f"All clauses too short after splitting: '{original_text}'. Using original text.")
        return [original_text.strip()] if original_text.strip() else []

    logger.debug(f"Clauses for emotion analysis for '{original_text}': {result_clauses}")
    return result_clauses


def detect_emotion(text: str) -> Dict[str, float]:
    """
    Detect emotions in the input text by running clause-level emotion
    then aggregating (max) across all clauses.

    Args:
        text: The review or input text.

    Returns:
        A dict mapping each emotion label to its confidence score.
        Example: {"joy": 0.72, "sadness": 0.10, ...}
    """
    logger.debug(f"Running emotion detection for text: '{text}'")
    try:
        clauses_to_analyze = get_emotion_clauses(text)

        combined: Dict[str, float] = {}
        if not clauses_to_analyze:
            logger.warning(f"No clauses to analyze for text: '{text}'")
            return {"neutral": 1.0} # Default if no clauses

        for sent_idx, clause_text in enumerate(clauses_to_analyze):
            # Ensure clause_text is not empty or just whitespace
            if not clause_text.strip():
                logger.debug(f"Skipping empty clause at index {sent_idx}")
                continue

            logger.debug(f"Processing clause {sent_idx+1}/{len(clauses_to_analyze)} for emotion: '{clause_text}'")
            results = emotion_pipe(clause_text)
            
            scores_list = results[0] if isinstance(results, list) and results and isinstance(results[0], list) else results
            
            if not scores_list:
                logger.warning(f"No emotion scores returned for clause: '{clause_text}'")
                continue

            current_clause_scores = {}
            for entry in scores_list:
                label = entry['label'].lower()
                score = float(entry['score'])
                current_clause_scores[label] = score
                # keep the highest score seen for each label across ALL clauses
                combined[label] = max(combined.get(label, 0.0), score)
            logger.debug(f"Emotion scores for clause '{clause_text}': {current_clause_scores}")

        if not combined: # If all clauses were empty or yielded no scores
            logger.warning(f"No emotions combined for text: '{text}'. Defaulting to neutral.")
            return {"neutral": 1.0}

        logger.debug(f"Aggregated (max) emotion scores before normalization: {combined}")
        return combined

    except Exception as e:
        logger.error(f"Emotion detection failed for text '{text}': {e}", exc_info=True)
        return {"neutral": 1.0}
