# pipeline/emotion.py
# Author: Bill Lu
# Description: Emotion detection using a Hugging Face DeBERTa model for multiclass emotion classification.

import os
import json
from typing import Dict, List
from transformers import pipeline, AutoTokenizer
from pipeline.logger import get_logger
from pipeline.preprocessing import preprocess_text
import re # Ensure re is imported if get_emotion_clauses uses it extensively
import math # Ensure math is imported

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
emotion_pipe = None
ALL_EMOTION_LABELS = []
N_EMOTION_CATEGORIES = 0

try:
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL, use_fast=False)
    emotion_pipe = pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        tokenizer=tokenizer,
        top_k=None
    )
    logger.info(f"Loaded emotion model '{EMOTION_MODEL}' with slow tokenizer")
    if emotion_pipe and hasattr(emotion_pipe, 'model') and hasattr(emotion_pipe.model, 'config') and hasattr(emotion_pipe.model.config, 'label2id'):
        ALL_EMOTION_LABELS = sorted(list(emotion_pipe.model.config.label2id.keys()))
        N_EMOTION_CATEGORIES = len(ALL_EMOTION_LABELS)
        logger.info(f"Emotion model has {N_EMOTION_CATEGORIES} labels: {ALL_EMOTION_LABELS}")
    else:
        logger.error("Could not retrieve emotion labels from the model configuration.")
        # Fallback if labels can't be dynamically determined, ensure N_EMOTION_CATEGORIES is set if known (e.g. 6)
        # For "AnkitAI/deberta-v3-small-base-emotions-classifier", it's 6.
        ALL_EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] # Default known labels
        N_EMOTION_CATEGORIES = 6
        logger.warning(f"Using default {N_EMOTION_CATEGORIES} emotion labels: {ALL_EMOTION_LABELS}")

except Exception as e:
    logger.error(f"Error initializing emotion pipeline with model '{EMOTION_MODEL}': {e}")
    # Set ALL_EMOTION_LABELS and N_EMOTION_CATEGORIES even if pipe fails, for default return structure
    ALL_EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] # Default known labels
    N_EMOTION_CATEGORIES = 6
    logger.warning(f"Using default {N_EMOTION_CATEGORIES} emotion labels due to pipeline init error: {ALL_EMOTION_LABELS}")
    # We don't raise here, detect_emotion will handle emotion_pipe being None

def get_default_emotion_result() -> Dict:
    """Returns a default structure for emotion analysis results."""
    default_scores = {label: 0.0 for label in ALL_EMOTION_LABELS} if ALL_EMOTION_LABELS else {}
    return {
        'all_emotion_scores': default_scores,
        'emotion_score': 0.0,          # Total emotional weight
        'emotion_confidence': 0.0,     # Average confidence
        'emotion_distribution': 0.0    # Normalized entropy
    }


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
        A dict with:
        - 'all_emotion_scores': Dict[str, float] of each emotion to its max score
        - 'emotion_score': float, sum of all emotion scores (total emotional weight)
        - 'emotion_confidence': float, average of all emotion scores
        - 'emotion_distribution': float, normalized entropy of emotion scores
    """
    logger.debug(f"Running emotion detection for text: '{text}'")

    if emotion_pipe is None or N_EMOTION_CATEGORIES == 0:
        logger.error("Emotion pipeline or labels not initialized. Returning default emotion result.")
        return get_default_emotion_result()

    try:
        clauses_to_analyze = get_emotion_clauses(text)
        combined: Dict[str, float] = {}

        if not clauses_to_analyze:
            logger.warning(f"No clauses to analyze for text: '{text}'")
            # Fall through to calculate scores based on empty 'combined'
        else:
            for sent_idx, clause_text in enumerate(clauses_to_analyze):
                if not clause_text.strip():
                    logger.debug(f"Skipping empty clause at index {sent_idx}")
                    continue # ADDED continue

                logger.debug(f"Processing clause {sent_idx+1}/{len(clauses_to_analyze)} for emotion: '{clause_text}'")
                results = emotion_pipe(clause_text)
                
                scores_list = results[0] if isinstance(results, list) and results and isinstance(results[0], list) else results
                
                if not scores_list:
                    logger.warning(f"No emotion scores returned for clause: '{clause_text}'")
                    continue # ADDED continue

                current_clause_scores = {}
                for entry in scores_list:
                    label = entry['label'].lower()
                    if label in ALL_EMOTION_LABELS: # Ensure we only consider known labels
                        score = float(entry['score'])
                        current_clause_scores[label] = score
                        combined[label] = max(combined.get(label, 0.0), score)
                logger.debug(f"Emotion scores for clause '{clause_text}': {current_clause_scores}")

        # Initialize final scores with all known labels at 0.0
        final_aggregated_scores = {label: 0.0 for label in ALL_EMOTION_LABELS}
        final_aggregated_scores.update(combined) # Update with detected scores

        # Calculate emotion_score (total emotional weight)
        emotion_score_val = sum(final_aggregated_scores.values())

        # Calculate emotion_confidence (average confidence across all N categories)
        emotion_confidence_val = emotion_score_val / N_EMOTION_CATEGORIES if N_EMOTION_CATEGORIES > 0 else 0.0
        
        # Calculate emotion_distribution (normalized entropy)
        emotion_distribution_val = 0.0
        positive_scores = [s for s in final_aggregated_scores.values() if s > 0]

        if len(positive_scores) > 1: # Entropy is 0 if 0 or 1 emotion
            total_positive_score = sum(positive_scores)
            if total_positive_score > 0:
                probabilities = [s / total_positive_score for s in positive_scores]
                # Filter probabilities again in case of floating point issues making some zero
                entropy = -sum(p * math.log(p) for p in probabilities if p > 0) # Ensure math.log is used
                max_entropy = math.log(N_EMOTION_CATEGORIES) if N_EMOTION_CATEGORIES > 0 else 0 # Ensure math.log and check N_EMOTION_CATEGORIES
                if max_entropy > 0:
                    emotion_distribution_val = entropy / max_entropy
        
        logger.debug(f"Final aggregated emotion scores: {final_aggregated_scores}")
        logger.info(f"Emotion analysis: score={emotion_score_val:.3f}, confidence={emotion_confidence_val:.3f}, distribution={emotion_distribution_val:.3f}")

        return {
            'all_emotion_scores': final_aggregated_scores,
            'emotion_score': emotion_score_val,
            'emotion_confidence': emotion_confidence_val,
            'emotion_distribution': emotion_distribution_val
        }

    except Exception as e:
        logger.error(f"Emotion detection failed for text '{text}': {e}", exc_info=True)
        return get_default_emotion_result()
