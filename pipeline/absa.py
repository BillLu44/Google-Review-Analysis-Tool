# pipeline/absa.py
# Author: Bill Lu
# Description: Aspect-Based Sentiment Analysis using yangheng/deberta-v3-base-absa-v1.1 model

import os
import json
from typing import List, Dict
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

# Initialize ABSA model from config
try:
    absa_model = _config['models']['absa']  # "yangheng/deberta-v3-base-absa-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(absa_model, use_fast=False)
    absa_classifier = pipeline(
        "text-classification",
        model=absa_model,
        tokenizer=tokenizer,
        top_k=None  # Get all possible labels
    )
    logger.info(f"Loaded ABSA model '{absa_model}' with slow tokenizer")
except Exception as e:
    logger.error(f"Failed to load ABSA model: {e}")
    absa_classifier = None

# Common restaurant/service aspects to look for
COMMON_ASPECTS = [
    "food", "service", "staff", "atmosphere", "ambiance", "price", "value",
    "location", "wait", "portion", "quality", "cleanliness", "menu", "dessert",
    "drink", "wine", "coffee", "parking", "reservation", "noise", "seating",
    "burger", "steak", "fries", "pizza", "sandwich", "beer", "beers"
]

def extract_aspects(text: str) -> List[str]:
    """Simple keyword-based aspect extraction"""
    text_lower = text.lower()
    found_aspects = [asp for asp in COMMON_ASPECTS if asp in text_lower]
    if not found_aspects:
        return ["overall"]
    # remove singular when plural present
    unique = set(found_aspects)
    for asp in list(unique):
        if asp.endswith("s") and asp[:-1] in unique:
            unique.remove(asp[:-1])
    return list(unique)

def analyze_aspect_sentiment(text: str, aspect: str) -> Dict:
    try:
        if absa_classifier is None:
            logger.warning(f"ABSA classifier not available. Cannot analyze aspect '{aspect}'.")
            return {'aspect': aspect, 'sentiment_label': 'neutral', 'sentiment_score': 0.0, 'confidence': 0.0}

        # give the ABSA model both text and aspect
        aspect_input = f"{text} [SEP] {aspect}"
        results = absa_classifier(aspect_input)
        logger.info(f"Raw ABSA output for '{aspect}': {results}")
        # unwrap nested lists if necessary
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]
        if not results:
            raise ValueError("empty ABSA output")
        # pick highest‐score label
        best = max(results, key=lambda x: x.get('score', 0))
        label_raw = best.get('label', '')
        confidence = float(best.get('score', 0)) # Model's confidence in this label
        # normalize “LABEL_n”
        if label_raw.upper().startswith('LABEL_'):
            idx = int(label_raw.split('_')[1])
            label = {0: 'negative', 1: 'neutral', 2: 'positive'}.get(idx, 'neutral')
        else:
            label = label_raw.lower()
            # Ensure label is one of 'positive', 'negative', 'neutral'
            if 'positive' in label:
                label = 'positive'
            elif 'negative' in label:
                label = 'negative'
            else: # Default to neutral if specific keywords not found or ambiguous
                label = 'neutral'

        # convert to signed score based on label, sentiment_score is 0 for neutral
        if label == 'positive':
            sentiment_score = confidence
        elif label == 'negative':
            sentiment_score = -confidence
        else:
            sentiment_score = 0.0
        
        logger.debug(f"ABSA {aspect}: raw='{label_raw}' -> final_label='{label}', signed_score={sentiment_score:.3f}, confidence={confidence:.3f}")
        return {'aspect': aspect, 'sentiment_label': label, 'sentiment_score': sentiment_score, 'confidence': confidence}
    except Exception as e:
        logger.error(f"Error analyzing aspect sentiment for '{aspect}': {e}")
        return {'aspect': aspect, 'sentiment_label': 'neutral', 'sentiment_score': 0.0, 'confidence': 0.0}

def analyze_aspect_sentiment_with_clauses(text: str, aspect: str) -> List[Dict]:
    """Analyze aspect sentiment by splitting contradictory clauses"""
    try:
        if absa_classifier is None:
            return [{'aspect': aspect, 'sentiment_label': 'neutral', 'sentiment_score': 0.0, 'confidence': 0.0}]

        # Split on common contradiction patterns
        clauses = []
        if ", but " in text.lower():
            clauses = [clause.strip() for clause in text.split(", but ")]
        elif " but " in text.lower():
            clauses = [clause.strip() for clause in text.split(" but ")]
        else:
            clauses = [text]
        
        aspect_mentions = []
        for clause in clauses:
            # Only analyze clauses that mention the aspect
            if aspect.lower() in clause.lower():
                aspect_input = f"{clause} [SEP] {aspect}"
                results = absa_classifier(aspect_input)
                logger.info(f"Clause '{clause}' -> Raw ABSA output for '{aspect}': {results}")
                
                # Process as before
                if isinstance(results, list) and results and isinstance(results[0], list):
                    results = results[0]
                if not results:
                    continue
                    
                best = max(results, key=lambda x: x.get('score', 0))
                label_raw = best.get('label', '')
                confidence = float(best.get('score', 0))
                
                # Normalize label
                if label_raw.upper().startswith('LABEL_'):
                    idx = int(label_raw.split('_')[1])
                    label = {0: 'negative', 1: 'neutral', 2: 'positive'}.get(idx, 'neutral')
                else:
                    label = label_raw.lower()
                    if 'positive' in label:
                        label = 'positive'
                    elif 'negative' in label:
                        label = 'negative'
                    else:
                        label = 'neutral'

                # Convert to signed score
                if label == 'positive':
                    sentiment_score = confidence
                elif label == 'negative':
                    sentiment_score = -confidence
                else:
                    sentiment_score = 0.0
                
                aspect_mentions.append({
                    'aspect': aspect,
                    'sentiment_label': label,
                    'sentiment_score': sentiment_score,
                    'confidence': confidence,
                    'clause': clause
                })
        
        return aspect_mentions if aspect_mentions else [{'aspect': aspect, 'sentiment_label': 'neutral', 'sentiment_score': 0.0, 'confidence': 0.0}]
        
    except Exception as e:
        logger.error(f"Error analyzing aspect sentiment for '{aspect}': {e}")
        return [{'aspect': aspect, 'sentiment_label': 'neutral', 'sentiment_score': 0.0, 'confidence': 0.0}]

def analyze_absa(text: str) -> Dict:
    """
    Perform Aspect-Based Sentiment Analysis on input text.
    
    Args:
        text (str): Input review text
        
    Returns:
        Dict: Contains detailed aspect results and aggregate statistics:
              - 'aspect_details': List[Dict] - List of aspect-sentiment pairs
              - 'num_pos_aspects': int
              - 'num_neg_aspects': int
              - 'avg_aspect_score': float (mean of signed sentiment scores)
              - 'avg_aspect_confidence': float (mean of confidences)
    """
    if absa_classifier is None:
        logger.warning("ABSA classifier not available. Skipping ABSA.")
        return {
            'aspect_details': [],
            'num_pos_aspects': 0,
            'num_neg_aspects': 0,
            'avg_aspect_score': 0.0,
            'avg_aspect_confidence': 0.0
        }

    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("Empty or invalid text provided to ABSA analysis")
        return {
            'aspect_details': [],
            'num_pos_aspects': 0,
            'num_neg_aspects': 0,
            'avg_aspect_score': 0.0,
            'avg_aspect_confidence': 0.0
        }
    
    # Extract aspects mentioned in the text
    aspects = extract_aspects(text)
    
    detailed_results = []
    all_sentiment_scores = []
    all_confidences = []
    num_pos_aspects = 0
    num_neg_aspects = 0

    for aspect in aspects:
        aspect_mentions = analyze_aspect_sentiment_with_clauses(text, aspect)
        
        if len(aspect_mentions) > 1:
            # Count EACH mention, not just the aggregated result
            for mention in aspect_mentions:
                if mention['sentiment_label'] == 'positive':
                    num_pos_aspects += 1
                elif mention['sentiment_label'] == 'negative':
                    num_neg_aspects += 1
            
            # Still create aggregated result for display
            avg_score = sum(m['sentiment_score'] for m in aspect_mentions) / len(aspect_mentions)
            avg_confidence = sum(m['confidence'] for m in aspect_mentions) / len(aspect_mentions)
            
            if avg_score >= 0.1:
                final_label = 'positive'
            elif avg_score <= -0.1:
                final_label = 'negative'
            else:
                final_label = 'neutral'
                
            aspect_result = {
                'aspect': aspect,
                'sentiment_label': final_label,
                'sentiment_score': avg_score,
                'confidence': avg_confidence
            }
        else:
            # Single mention
            aspect_result = aspect_mentions[0]
            if aspect_result['sentiment_label'] == 'positive':
                num_pos_aspects += 1
            elif aspect_result['sentiment_label'] == 'negative':
                num_neg_aspects += 1
        
        detailed_results.append(aspect_result)
        all_sentiment_scores.append(aspect_result['sentiment_score'])
        all_confidences.append(aspect_result['confidence'])
    
    avg_aspect_score = sum(all_sentiment_scores) / len(all_sentiment_scores) if all_sentiment_scores else 0.0
    avg_aspect_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    logger.info(f"ABSA analysis: {len(detailed_results)} aspects. Pos: {num_pos_aspects}, Neg: {num_neg_aspects}, AvgScore: {avg_aspect_score:.3f}, AvgConf: {avg_aspect_confidence:.3f}")
    
    return {
        'aspect_details': detailed_results,
        'num_pos_aspects': num_pos_aspects,
        'num_neg_aspects': num_neg_aspects,
        'avg_aspect_score': avg_aspect_score,
        'avg_aspect_confidence': avg_aspect_confidence
    }

# For testing
if __name__ == "__main__":
    test_text = "The food was amazing but the service was terrible and slow."
    # test_text_2 = "The burger is great and the fries are okay."
    result_dict = analyze_absa(test_text)
    
    print("\n--- ABSA Analysis Results ---")
    print(f"Text: \"{test_text}\"")
    print("Detailed Aspect Sentiments:")
    if result_dict['aspect_details']:
        for detail in result_dict['aspect_details']:
            print(
                f"  - Aspect: {detail['aspect']:<15} "
                f"Label: {detail['sentiment_label']:<10} "
                f"Score: {detail['sentiment_score']:>6.3f} "
                f"Confidence: {detail['confidence']:>6.3f}"
            )
    else:
        print("  No aspects analyzed.")
    
    print("\nAggregate Statistics:")
    print(f"  Number of Positive Aspects: {result_dict['num_pos_aspects']}")
    print(f"  Number of Negative Aspects: {result_dict['num_neg_aspects']}")
    print(f"  Average Aspect Score:      {result_dict['avg_aspect_score']:.3f}")
    print(f"  Average Aspect Confidence: {result_dict['avg_aspect_confidence']:.3f}")
    print("---------------------------\n")

    # Example with "overall" aspect if no specific keywords found
    # test_text_overall = "It was an experience."
    # result_overall = analyze_absa(test_text_overall)
    # print(f"\nABSA Result for '{test_text_overall}': {result_overall}")
