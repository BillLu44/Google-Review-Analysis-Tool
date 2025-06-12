# pipeline/overall_sentiment.py
# Author: Bill Lu
# Description: Transformer-based sentiment analysis with support for multiple models

import os
import json
from transformers import pipeline, AutoTokenizer
from pipeline.logger import get_logger
from pipeline.preprocessing import preprocess_text
import torch

# Initialize logger
logger = get_logger(__name__)

# Comprehensive model configurations based on actual model cards
MODEL_CONFIGS = {
    # Binary: 0=negative, 1=positive
    "siebert/sentiment-roberta-large-english": {
        "num_classes": 2,
        "label_mapping": {0: "negative", 1: "positive"},
        "uses_label_prefix": False,
        "description": "Binary sentiment (positive/negative) for English text"
    },
    
    # 3-way: negative/neutral/positive (Amazon reviews)
    "RashidNLP/Amazon-Deberta-Base-Sentiment": {
        "num_classes": 3,
        "label_mapping": {0: "negative", 1: "neutral", 2: "positive"},
        "uses_label_prefix": True,
        "description": "3-way sentiment for English Amazon reviews"
    },
    
    # CRITICAL FIX: This model has REVERSED mapping! 0=Positive, 1=Negative
    "dnzblgn/Sentiment-Analysis-Customer-Reviews": {
        "num_classes": 2,
        "label_mapping": {0: "positive", 1: "negative"},  # REVERSED!
        "uses_label_prefix": False,
        "description": "Binary sentiment for English reviews (0=Positive, 1=Negative)"
    },
    
    # 3-way tweet sentiment
    "Elron/deberta-v3-large-sentiment": {
        "num_classes": 3,
        "label_mapping": {0: "negative", 1: "neutral", 2: "positive"},
        "uses_label_prefix": True,
        "description": "3-way sentiment for English tweet-like text"
    },
    
    # Binary review sentiment
    "Kaludi/Reviews-Sentiment-Analysis": {
        "num_classes": 2,
        "label_mapping": {0: "negative", 1: "positive"},
        "uses_label_prefix": False,
        "description": "Binary sentiment for English reviews"
    },
    
    # 5-way sentiment (Very Negative → Very Positive)
    "tabularisai/robust-sentiment-analysis": {
        "num_classes": 5,
        "label_mapping": {
            0: "very_negative", 
            1: "negative", 
            2: "neutral", 
            3: "positive", 
            4: "very_positive"
        },
        "uses_label_prefix": True,
        "description": "5-way sentiment for English text"
    },
    
    # 5-way multilingual sentiment
    "tabularisai/multilingual-sentiment-analysis": {
        "num_classes": 5,
        "label_mapping": {
            0: "very_negative", 
            1: "negative", 
            2: "neutral", 
            3: "positive", 
            4: "very_positive"
        },
        "uses_label_prefix": True,
        "description": "5-way sentiment for 22 languages"
    },
    
    # === YELP-TRAINED MODELS ===
    
    # RoBERTa fine-tuned on Yelp Polarity (98.08% accuracy)
    "VictorSanh/roberta-base-finetuned-yelp-polarity": {
        "num_classes": 2,
        "label_mapping": {0: "negative", 1: "positive"},
        "uses_label_prefix": False,
        "description": "RoBERTa-base fine-tuned on Yelp Polarity (98.08% accuracy, 125M params)"
    },
    
    # BERT fine-tuned on Yelp Polarity (very popular model)
    "textattack/bert-base-uncased-yelp-polarity": {
        "num_classes": 2,
        "label_mapping": {0: "negative", 1: "positive"},
        "uses_label_prefix": True,  # Likely uses LABEL_0/LABEL_1
        "description": "BERT-base fine-tuned on Yelp Polarity (~97% accuracy, very popular)"
    },
    
    # DistilBERT trained on SST-2 then Yelp (lightweight)
    "AirrStorm/DistilBERT-SST2-Yelp": {
        "num_classes": 2,
        "label_mapping": {0: "negative", 1: "positive"},
        "uses_label_prefix": False,
        "description": "DistilBERT fine-tuned on SST-2 then Yelp (~92% accuracy, 67M params)"
    },
    
    # BERT fine-tuned specifically on Yelp restaurant reviews
    "karimbkh/BERT_fineTuned_Sentiment_Classification_Yelp": {
        "num_classes": 2,
        "label_mapping": {0: "negative", 1: "positive"},
        "uses_label_prefix": True,  # BERT models typically use LABEL_ format
        "description": "BERT-base fine-tuned on Yelp restaurant reviews (97.8% accuracy)"
    },
    
    # RoBERTa fine-tuned on Yelp Review Full (5-star ratings)
    "pig4431/YELP_roBERTa_5E": {
        "num_classes": 5,
        "label_mapping": {
            0: "very_negative",  # 1 star
            1: "negative",       # 2 stars  
            2: "neutral",        # 3 stars
            3: "positive",       # 4 stars
            4: "very_positive"   # 5 stars
        },
        "uses_label_prefix": False,
        "description": "RoBERTa-base fine-tuned on Yelp Review Full 5-star data (98.67% accuracy)"
    }
}

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
try:
    with open(CONFIG_PATH, 'r') as cf:
        _config = json.load(cf)
except Exception as e:
    logger.error(f"Failed to load config.json: {e}")
    raise

# Global variables for model and config
sentiment_pipe = None
model_config = None
current_model_name = None

def discover_model_format(model_name, test_text="I love this!"):
    """
    Dynamically discover a model's output format by testing it.
    """
    try:
        logger.info(f"Discovering format for {model_name}")
        
        # Load model with minimal assumptions
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        pipe = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer, return_all_scores=True)
        
        # Test with a simple positive example
        raw_output = pipe(test_text)
        logger.info(f"Raw output for '{test_text}': {raw_output}")
        
        # Analyze the structure
        predictions = raw_output[0] if isinstance(raw_output[0], list) else raw_output
        
        # Extract all possible labels
        all_labels = [pred['label'] for pred in predictions]
        logger.info(f"All labels: {all_labels}")
        
        # Determine format
        uses_label_prefix = any(label.startswith('LABEL_') for label in all_labels)
        num_classes = len(predictions)
        
        # Try to infer mapping
        if uses_label_prefix:
            # Extract indices and sort
            indices = []
            for label in all_labels:
                try:
                    idx = int(label.split('_')[1])
                    indices.append(idx)
                except:
                    pass
            
            if num_classes == 2:
                if set(indices) == {0, 1}:
                    label_mapping = {0: "negative", 1: "positive"}
                else:
                    label_mapping = {min(indices): "negative", max(indices): "positive"}
            else:  # 3 classes
                label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
        
        else:
            # Direct string labels - need to infer from actual labels
            label_mapping = {}
            for i, label in enumerate(all_labels):
                clean_label = label.lower()
                if 'pos' in clean_label:
                    label_mapping[i] = 'positive'
                elif 'neg' in clean_label:
                    label_mapping[i] = 'negative'
                else:
                    label_mapping[i] = 'neutral'
        
        discovered_config = {
            "num_classes": num_classes,
            "label_mapping": label_mapping,
            "uses_label_prefix": uses_label_prefix,
            "description": f"Auto-discovered config for {model_name}"
        }
        
        logger.info(f"Discovered config: {discovered_config}")
        return discovered_config, pipe
        
    except Exception as e:
        logger.error(f"Failed to discover format for {model_name}: {e}")
        return None, None

def discover_and_test_model(model_name):
    """
    Load a model and test it with known examples to discover correct mapping.
    """
    try:
        logger.info(f"Discovering format for {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        pipe = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer, return_all_scores=True)
        
        # Test with clear positive and negative examples
        test_cases = [
            ("I love this amazing restaurant!", "positive"),
            ("I hate this terrible place!", "negative"),
            ("This is okay", "neutral")
        ]
        
        discovered_mapping = {}
        
        for test_text, expected_sentiment in test_cases:
            raw_output = pipe(test_text)
            predictions = raw_output[0] if isinstance(raw_output[0], list) else raw_output
            
            # Get the highest confidence prediction
            best_pred = max(predictions, key=lambda x: x['score'])
            label = best_pred['label']
            
            logger.info(f"Text: '{test_text}' -> Label: {label} (score: {best_pred['score']:.3f})")
            
            # Extract index from LABEL_X format
            if label.startswith('LABEL_'):
                idx = int(label.split('_')[1])
                discovered_mapping[idx] = expected_sentiment
            else:
                # Handle direct string labels
                clean_label = label.lower()
                if 'pos' in clean_label:
                    discovered_mapping[1] = "positive"  # Assume index 1 for positive
                elif 'neg' in clean_label:
                    discovered_mapping[0] = "negative"  # Assume index 0 for negative
        
        # Validate the mapping by checking if it makes sense
        logger.info(f"Discovered mapping for {model_name}: {discovered_mapping}")
        
        # Create final config
        config = {
            "num_classes": len(predictions),
            "label_mapping": discovered_mapping,
            "uses_label_prefix": any(pred['label'].startswith('LABEL_') for pred in predictions),
            "description": f"Auto-discovered config for {model_name}"
        }
        
        return config, pipe
        
    except Exception as e:
        logger.error(f"Failed to discover format for {model_name}: {e}")
        return None, None

def load_sentiment_model(model_name=None):
    """
    Load a sentiment model with dynamic format discovery.
    """
    global sentiment_pipe, model_config, current_model_name
    
    if model_name is None:
        model_name = _config['models']['sentiment']
    
    if current_model_name == model_name and sentiment_pipe is not None:
        logger.info(f"Model {model_name} already loaded")
        return True
    
    # Check if we have a predefined config
    model_config = MODEL_CONFIGS.get(model_name)
    
    if model_config:
        logger.info(f"Using predefined config for {model_name}")
        try:
            # Standard sentiment analysis pipeline
            logger.info(f"Loading standard sentiment model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            sentiment_pipe = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=tokenizer,
                return_all_scores=True
            )
            current_model_name = model_name
            return True
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return False
    else:
        # Discover the format
        logger.info(f"No predefined config for {model_name}, discovering format...")
        model_config, sentiment_pipe = discover_and_test_model(model_name)
        
        if model_config is None:
            logger.error(f"Could not discover format for {model_name}")
            return False
        
        current_model_name = model_name
        return True

def normalize_label(label_raw, model_config):
    """
    Normalize any model output to a standardized text label.
    """
    confidence_in_mapping = 1.0
    
    # Standard handling for classification models
    if isinstance(label_raw, int):
        idx = label_raw
        text_label = model_config["label_mapping"].get(idx, "neutral")
        logger.debug(f"Direct integer label {idx} -> {text_label}")
        return text_label, confidence_in_mapping
    
    if isinstance(label_raw, str) and label_raw.upper().startswith("LABEL_"):
        try:
            idx = int(label_raw.split("_", 1)[1])
            text_label = model_config["label_mapping"].get(idx, "neutral")
            logger.debug(f"LABEL_ format {label_raw} -> idx {idx} -> {text_label}")
            return text_label, confidence_in_mapping
        except (ValueError, IndexError):
            logger.warning(f"Could not parse LABEL_ format: {label_raw}")
            confidence_in_mapping = 0.5
    
    if isinstance(label_raw, str):
        text_label = str(label_raw).lower().strip()
        
        # Normalize common variations - EXPANDED for 5-way models
        if text_label in ["pos", "positive", "pos_sentiment"]:
            text_label = "positive"
        elif text_label in ["neg", "negative", "neg_sentiment"]:
            text_label = "negative"
        elif text_label in ["neu", "neutral", "neu_sentiment"]:
            text_label = "neutral"
        elif text_label in ["very positive", "very_positive"]:
            text_label = "very_positive"
        elif text_label in ["very negative", "very_negative"]:
            text_label = "very_negative"
        
        logger.debug(f"Direct string label {label_raw} -> {text_label}")
        return text_label, confidence_in_mapping
    
    logger.warning(f"Unknown label format: {label_raw} (type: {type(label_raw)})")
    return "neutral", 0.3

def analyze_sentiment(text, model_name=None) -> dict:
    """
    Analyze sentiment using transformer model.
    
    Args:
        text: Input text (should be string)
        model_name: Optional specific model to use. If None, uses current loaded model.
        
    Returns:
        dict with 'sentiment' (int: -1, 0, 1), 'confidence_score' (float: 0-1), and 'model_used'
    """
    # Load model if specified or if no model is currently loaded
    if model_name is not None or sentiment_pipe is None:
        if not load_sentiment_model(model_name):
            return {
                'sentiment': 0,
                'confidence_score': 0.0,
                'model_used': model_name or 'none'
            }
    
    if sentiment_pipe is None or model_config is None:
        logger.warning("No sentiment model available")
        return {
            'sentiment': 0,
            'confidence_score': 0.0,
            'model_used': current_model_name or 'none'
        }
    
    # Handle edge cases
    if text is None:
        logger.warning("None text provided to sentiment analysis")
        return {
            'sentiment': 0,
            'confidence_score': 0.0,
            'model_used': current_model_name
        }
    
    if not isinstance(text, str):
        text = str(text)
        logger.debug(f"Converted non-string input to string: {text}")
    
    if not text.strip():
        logger.warning("Text is empty after stripping")
        return {
            'sentiment': 0,
            'confidence_score': 0.0,
            'model_used': current_model_name
        }
    
    logger.debug(f"[Sentiment] raw input: {repr(text)}")
    
    # Optional preprocessing
    try:
        clean = preprocess_text(text)
        text = " ".join(clean['sentences'])
        logger.debug(f"[Sentiment] post-preprocess: {repr(text)}")
    except Exception as e:
        logger.warning(f"[Sentiment] preprocessing failed, using raw text: {e}")

    try:
        # Standard sentiment analysis
        raw_output = sentiment_pipe(text)
        
        # Handle nested list structure
        if isinstance(raw_output, list) and len(raw_output) > 0:
            if isinstance(raw_output[0], list):
                predictions = raw_output[0]
            else:
                predictions = raw_output
        else:
            predictions = [raw_output]
        
        logger.debug(f"[Sentiment] ALL model predictions: {predictions}")
        
        # FOR DEBUGGING: Log all predictions for 5-way models
        if model_config.get("num_classes") == 5:
            logger.info(f"[DEBUG] 5-way model {current_model_name} predictions:")
            for pred in predictions:
                label_idx = None
                if pred['label'].startswith('LABEL_'):
                    label_idx = int(pred['label'].split('_')[1])
                mapped_label = model_config["label_mapping"].get(label_idx, pred['label'])
                logger.info(f"  {pred['label']} (idx {label_idx}) -> {mapped_label}: {pred['score']:.3f}")
        
        # Find best prediction
        best_prediction = max(predictions, key=lambda x: x['score'])
        label_raw = best_prediction['label']
        raw_confidence = float(best_prediction['score'])
        
        logger.debug(f"[Sentiment] Best prediction: {best_prediction}")
        
        # Normalize label
        text_label, mapping_confidence = normalize_label(label_raw, model_config)
        
        # Calculate final confidence
        adjusted_confidence = raw_confidence * mapping_confidence
        
        # Convert text label to integer sentiment
        sentiment_int = 0
        if text_label in ["positive", "very_positive"]:
            sentiment_int = 1
        elif text_label in ["negative", "very_negative"]:
            sentiment_int = -1
        # neutral stays 0
        
        logger.debug(f"[Sentiment] Final result: {text_label} (int: {sentiment_int}), "
                    f"Model confidence: {raw_confidence:.3f}, "
                    f"Mapping confidence: {mapping_confidence:.3f}, "
                    f"Adjusted confidence: {adjusted_confidence:.3f}")
        
        return {
            'sentiment': sentiment_int,
            'confidence_score': adjusted_confidence,
            'model_used': current_model_name
        }
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {
            'sentiment': 0,
            'confidence_score': 0.0,
            'model_used': current_model_name
        }

def get_available_models():
    """Return list of all available sentiment models."""
    return list(MODEL_CONFIGS.keys())

def get_model_info(model_name):
    """Get configuration info for a specific model."""
    return MODEL_CONFIGS.get(model_name, {})

def compare_models(text, models=None):
    """
    Compare sentiment analysis results across multiple models.
    
    Args:
        text: Input text to analyze
        models: List of model names to compare. If None, uses all available models.
        
    Returns:
        dict: Results from each model
    """
    if models is None:
        models = get_available_models()
    
    results = {}
    
    for model_name in models:
        logger.info(f"Testing model: {model_name}")
        try:
            result = analyze_sentiment(text, model_name)
            results[model_name] = result
        except Exception as e:
            logger.error(f"Failed to test model {model_name}: {e}")
            results[model_name] = {
                'sentiment': 0,
                'confidence_score': 0.0,
                'model_used': model_name,
                'error': str(e)
            }
    
    return results

# Initialize with default model from config
load_sentiment_model()

# For testing
if __name__ == "__main__":
    test_texts = [
        "I love this restaurant!",
        "I hate this place!",
        "This is okay, nothing special.",
        "The food was amazing and the service was excellent!",
        "Terrible experience, would never go back."
    ]
    
    print("=== TESTING ALL SENTIMENT MODELS ===\n")
    
    for test_text in test_texts:
        print(f"Testing text: '{test_text}'")
        print("-" * 50)
        
        results = compare_models(test_text)
        
        for model_name, result in results.items():
            sentiment_labels = {-1: "NEGATIVE", 0: "NEUTRAL", 1: "POSITIVE"}
            sentiment_label = sentiment_labels.get(result['sentiment'], 'UNKNOWN')
            
            if 'error' in result:
                print(f"{model_name:<50} ERROR: {result['error']}")
            else:
                print(f"{model_name:<50} {sentiment_label:<8} (conf: {result['confidence_score']:.3f})")
        
        print("\n")
