# pipeline/absa.py
# Author: Bill Lu
# Description: Aspect-Based Sentiment Analysis using yangheng/deberta-v3-base-absa-v1.1 model

import os
import json
from typing import List, Dict
from transformers import pipeline, AutoTokenizer
from unittest.mock import MagicMock
from pipeline.logger import get_logger
from pipeline.preprocessing import preprocess_text
import spacy
from spacy.matcher import Matcher

# Initialize logger
logger = get_logger(__name__)

# load spaCy model & build a simple NP matcher
nlp = spacy.load("en_core_web_sm", disable=["ner","textcat"])
matcher = Matcher(nlp.vocab)
matcher.add("NP_PATTERN", [[{"POS": "ADJ", "OP": "*"}, {"POS": "NOUN", "OP": "+"}]])

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
    absa_model_name = _config['models']['absa']
    tokenizer = AutoTokenizer.from_pretrained(absa_model_name, use_fast=False)
    absa_classifier = pipeline(
        "text-classification",
        model=absa_model_name,
        tokenizer=tokenizer,
        top_k=None
    )
    logger.info(f"Loaded ABSA model '{absa_model_name}' with slow tokenizer")
except Exception as e:
    logger.error(f"Failed to load ABSA model: {e}")
    absa_classifier = None

COMMON_ASPECTS = [
    "food", "meal", "dish", "dishes", "cuisine", "cooking", "recipe",
    "appetizer", "appetizers", "starter", "starters", "entree", "entrees", 
    "main", "mains", "dessert", "desserts", "soup", "salad", "bread",
    "burger", "burgers", "steak", "steaks", "fish", "seafood", "chicken", 
    "beef", "pork", "lamb", "pasta", "pizza", "sandwich", "sandwiches",
    "fries", "vegetables", "veggies", "rice", "noodles", "sauce", "sauces",
    "cheese", "meat", "protein", "side", "sides", "plate", "plates",
    "service", "staff", "server", "servers", "waiter", "waiters", 
    "waitress", "waitresses", "bartender", "chef", "chefs", "manager",
    "host", "hostess", "team", "crew", "employee", "employees",
    "atmosphere", "ambiance", "ambience", "environment", "vibe", "mood",
    "decor", "decoration", "interior", "exterior", "design", "lighting",
    "music", "noise", "seating", "seats", "table", "tables", "chair", "chairs",
    "patio", "terrace", "dining", "room", "space", "place", "restaurant",
    "establishment", "venue", "location", "spot",
    "wait", "waiting", "timing", "time", "speed", "pace", "reservation",
    "reservations", "booking", "order", "orders", "delivery", "takeout",
    "portion", "portions", "size", "sizes", "presentation", "plating",
    "price", "prices", "pricing", "cost", "costs", "value", "money", 
    "bill", "check", "charge", "charges", "expensive", "cheap", "affordable",
    "budget", "deal", "deals", "special", "specials",
    "quality", "freshness", "fresh", "flavor", "flavors", "taste", "tastes",
    "texture", "temperature", "hot", "cold", "warm", "cool", "seasoning",
    "salt", "spice", "spices", "preparation", "cooking", "cooked",
    "drink", "drinks", "beverage", "beverages", "wine", "wines", "beer", 
    "beers", "cocktail", "cocktails", "coffee", "tea", "water", "soda",
    "juice", "alcohol", "bar",
    "experience", "visit", "meal", "dinner", "lunch", "breakfast", "brunch",
    "overall", "general", "everything", "nothing"
]

def _clean_trailing_punctuation(s: str) -> str:
    s = s.strip()
    # Repeatedly strip common trailing punctuation
    while s and s[-1] in '.,!?;:':
        s = s[:-1]
    return s.strip()

def extract_aspects(text: str) -> List[str]:
    if not text or not text.strip():
        return ["overall"]

    # extract noun-chunks & matcher spans
    doc = nlp(text)
    noun_chunks = {chunk.text.lower().strip() for chunk in doc.noun_chunks}
    matched_spans = set()
    for _mid, start, end in matcher(doc):
        span = doc[start:end]
        matched_spans.add(span.text.lower().strip())

    # combine and filter by known aspects
    candidates = noun_chunks | matched_spans
    found_aspects = {asp for asp in COMMON_ASPECTS if asp in candidates}
    if not found_aspects:
        return ["overall"]

    aspects_to_return = set(found_aspects)
    
    # Prefer plural forms if both singular and plural are found
    for aspect_in_set in list(aspects_to_return):
        simple_plural_form = aspect_in_set + "s"
        es_plural_form = aspect_in_set + "es" if aspect_in_set.endswith(("sh", "ch", "s", "x", "z")) else None

        if simple_plural_form in aspects_to_return and simple_plural_form != aspect_in_set:
            aspects_to_return.discard(aspect_in_set)  # Remove singular, keep plural
        elif es_plural_form and es_plural_form in aspects_to_return and es_plural_form != aspect_in_set:
            aspects_to_return.discard(aspect_in_set)  # Remove singular, keep es-plural
            
    if not aspects_to_return: 
        return ["overall"]
        
    return sorted(list(aspects_to_return))

def analyze_aspect_sentiment(text: str, aspect: str) -> Dict:
    try:
        if absa_classifier is None:
            logger.warning("ABSA classifier not available")
            return {'aspect': aspect, 'sentiment_label': 'neutral', 'sentiment_score': 0.0, 'confidence': 0.0}

        text_to_classify = text.strip()
        if not text_to_classify:
            return {'aspect': aspect, 'sentiment_label': 'neutral', 'sentiment_score': 0.0, 'confidence': 0.0}

        aspect_input = f"{text_to_classify} [SEP] {aspect}"
        results = absa_classifier(aspect_input)
        
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]

        if not results:            
            return {'aspect': aspect, 'sentiment_label': 'neutral', 'sentiment_score': 0.0, 'confidence': 0.0}
        
        best = max(results, key=lambda x: x.get('score', 0.0))
        label_raw = best.get('label', '')
        
        if 'score' not in best:
            return {'aspect': aspect, 'sentiment_label': 'neutral', 'sentiment_score': 0.0, 'confidence': 0.0}
        
        confidence = float(best['score'])

        if not label_raw:
            return {'aspect': aspect, 'sentiment_label': 'neutral', 'sentiment_score': 0.0, 'confidence': confidence}

        if label_raw.upper().startswith('LABEL_'):
            label_num = int(label_raw.split('_')[1])
            label = {0: 'negative', 1: 'neutral', 2: 'positive'}.get(label_num, 'neutral')
        else:
            label = label_raw.lower()

        sentiment_score = 0.0
        if label == 'positive':
            sentiment_score = confidence
        elif label == 'negative':
            sentiment_score = -confidence
        
        return {'aspect': aspect, 'sentiment_label': label, 'sentiment_score': sentiment_score, 'confidence': confidence}
        
    except Exception as e:
        logger.error(f"Error analyzing aspect sentiment for '{aspect}': {e}")
        return {'aspect': aspect, 'sentiment_label': 'neutral', 'sentiment_score': 0.0, 'confidence': 0.0}

def extract_aspect_context(text: str, aspect: str, window: int = 10) -> str:
    if not text or not aspect:
        return ""
        
    doc = preprocess_text(text)['doc']
    context_strings = [] 

    aspect_token_indices = [i for i, t in enumerate(doc) if aspect.lower() in t.text.lower()]

    for token_doc_index in aspect_token_indices:
        start_window_idx = max(0, token_doc_index - window)
        end_window_idx = min(len(doc), token_doc_index + window + 1)
        
        span = doc[start_window_idx:end_window_idx]
        current_context_tokens = [t.text for t in span if not t.is_punct and not t.is_space]
        if current_context_tokens:
            context_strings.append(" ".join(current_context_tokens))
            
    final_context = " ".join(context_strings) 
    return final_context.strip()

def analyze_absa(text: str) -> Dict:
    """Main ABSA analysis function"""
    try:
        if not text or not text.strip():
            return {
                'aspect_details': [], 'num_pos_aspects': 0, 'num_neg_aspects': 0,
                'avg_aspect_score': 0.0, 'avg_aspect_confidence': 0.0
            }
        
        if absa_classifier is None:
            return {
                'aspect_details': [], 'num_pos_aspects': 0, 'num_neg_aspects': 0,
                'avg_aspect_score': 0.0, 'avg_aspect_confidence': 0.0
            }
        
        aspects = extract_aspects(text)
        aspect_details_aggregated = []
        num_pos_mentions = 0
        num_neg_mentions = 0
        
        all_scores_for_overall_avg = []
        all_confidences_for_overall_avg = []
        
        for aspect_item in aspects:
            sentiment_result = analyze_aspect_sentiment(text, aspect_item)
            context_phrase = extract_aspect_context(text, aspect_item)
            
            aspect_detail = {
                'aspect': aspect_item,
                'sentiment_label': sentiment_result['sentiment_label'],
                'sentiment_score': sentiment_result['sentiment_score'],
                'confidence': sentiment_result['confidence'],
                'context_phrase': context_phrase  # Add context phrase
            }
            
            aspect_details_aggregated.append(aspect_detail)
            
            if sentiment_result['sentiment_label'] == 'positive':
                num_pos_mentions += 1
            elif sentiment_result['sentiment_label'] == 'negative':
                num_neg_mentions += 1
                
            all_scores_for_overall_avg.append(sentiment_result['sentiment_score'])
            all_confidences_for_overall_avg.append(sentiment_result['confidence'])
        
        avg_aspect_score = sum(all_scores_for_overall_avg) / len(all_scores_for_overall_avg) if all_scores_for_overall_avg else 0.0
        avg_aspect_confidence = sum(all_confidences_for_overall_avg) / len(all_confidences_for_overall_avg) if all_confidences_for_overall_avg else 0.0
        
        return {
            'aspect_details': aspect_details_aggregated,
            'num_pos_aspects': num_pos_mentions,
            'num_neg_aspects': num_neg_mentions,
            'avg_aspect_score': avg_aspect_score,
            'avg_aspect_confidence': avg_aspect_confidence
        }
        
    except Exception as e:
        logger.error(f"Error in ABSA analysis for text '{text[:50]}...': {e}", exc_info=True)
        return {
            'aspect_details': [], 'num_pos_aspects': 0, 'num_neg_aspects': 0,
            'avg_aspect_score': 0.0, 'avg_aspect_confidence': 0.0
        }
