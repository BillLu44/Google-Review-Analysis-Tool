#!/usr/bin/env python3
# test_pipeline.py
# Author: Bill Lu
# Description: Run a suite of sample reviews through the full NLP pipeline for validation and timing.

import time
import json
import uuid # Add this import
from pipeline.logger import get_logger
from pipeline.preprocessing import preprocess_text
from pipeline.rule_based import rule_based_sentiment
from pipeline.overall_sentiment import analyze_sentiment
from pipeline.absa import analyze_absa
from pipeline.emotion import detect_emotion
from pipeline.sarcasm import detect_sarcasm
from pipeline.fusion import fuse_signals
from utils.output_formatter import save_results_to_file

# Initialize logger
logger = get_logger(__name__)

# Sample reviews for testing
SAMPLE_REVIEWS = [
    # Sarcasm & Irony Tests
    {
        'review_id': str(uuid.uuid4()),
        'text': "Oh wonderful, another 45-minute wait for cold pasta. Exactly what I needed after a long day."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "The service was so fast I almost forgot I was dining and not running a marathon."
    },
    
    # Contradictory Aspects
    {
        'review_id': str(uuid.uuid4()),
        'text': "The steak was absolutely divine, but the waiter was incredibly rude and the restaurant was filthy."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Terrible food, awful atmosphere, but the staff bent over backwards to help us - best service ever!"
    },
    
    # Backhanded Compliments
    {
        'review_id': str(uuid.uuid4()),
        'text': "The pizza was decent for a place that usually serves cardboard with cheese on top."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Finally found a restaurant where the food matches the low prices - you get what you pay for."
    },
    
    # Emotional Manipulation
    {
        'review_id': str(uuid.uuid4()),
        'text': "This place reminds me of my grandmother's cooking - if she had lost her sense of taste and forgot how to season food."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "I'm heartbroken to say this because I really wanted to love this place, but everything was disappointing."
    },
    
    # Positive Words, Negative Context
    {
        'review_id': str(uuid.uuid4()),
        'text': "Amazing how they managed to overcook the fish, undercook the vegetables, and serve it all lukewarm."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Incredible service - they only forgot half our order and brought the wrong drinks twice!"
    },
    
    # Negative Words, Positive Context
    {
        'review_id': str(uuid.uuid4()),
        'text': "I was worried this place would be terrible based on reviews, but I was completely wrong - everything was fantastic!"
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Don't let the awful exterior fool you - this hidden gem serves the best food in town."
    },
    
    # Mixed Temporal Sentiment
    {
        'review_id': str(uuid.uuid4()),
        'text': "Started terribly with a 30-minute wait and cold appetizers, but the main course was perfection and the dessert was heavenly."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "We had an amazing dinner and fantastic service, but then they charged us for items we never ordered."
    },
    
    # Subtle Negativity
    {
        'review_id': str(uuid.uuid4()),
        'text': "The restaurant tries really hard and the staff is very enthusiastic about the food they serve."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "For what we paid, the portion sizes were... adequate, and the flavors were certainly present."
    },
    
    # False Expectations
    {
        'review_id': str(uuid.uuid4()),
        'text': "If you enjoy waiting 2 hours for mediocre food while listening to screaming children, this is your paradise!"
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Perfect restaurant for people who love spending money on tiny portions and pretentious service."
    },
    
    # Conditional Sentiment
    {
        'review_id': str(uuid.uuid4()),
        'text': "This would be a 5-star restaurant if they fixed the service, improved the food, and cleaned the place."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Great potential, but until they replace the chef and retrain the staff, avoid at all costs."
    }
]


def run_tests():
    """
    Processes each sample review through all pipeline stages and logs results and timings.
    """
    total_start = time.time()
    all_results = []

    for sample in SAMPLE_REVIEWS:
        rid = sample['review_id']
        text = sample['text']
        logger.info(f"\n=== Processing review_id={rid} ===")

        # 1. Preprocessing
        t0 = time.time()
        pre = preprocess_text(text)
        t1 = time.time()

        # 2. Rule-based
        rule_output = rule_based_sentiment(text)
        t2 = time.time()

        # 3. Transformer sentiment
        sentiment_output = analyze_sentiment(text) # analyze_sentiment now returns {'sentiment': int, 'confidence_score': float}
        t3 = time.time()

        # 4. Aspect-based sentiment
        absa_output = analyze_absa(text) # analyze_absa now returns a dict
        t4 = time.time()

        # 5. Emotion detection
        emotion_output = detect_emotion(text) # detect_emotion now returns a dict with multiple scores
        t5 = time.time()

        # 6. Sarcasm detection
        sarcasm_output = detect_sarcasm(text)
        t6 = time.time()

        # Adapt sentiment_output for fusion model's expected sentiment_score (-1 to 1 float)
        transformer_sentiment_score_for_fusion = 0.0
        if sentiment_output['sentiment'] == 1: # positive
            transformer_sentiment_score_for_fusion = sentiment_output['confidence_score']
        elif sentiment_output['sentiment'] == -1: # negative
            transformer_sentiment_score_for_fusion = -sentiment_output['confidence_score']
        # if sentiment_output['sentiment'] == 0 (neutral), it remains 0.0

        # 7. Fusion
        signals = {
            'rule_score': rule_output['rule_score'],
            'rule_polarity': rule_output['rule_polarity'],
            'sentiment_score': transformer_sentiment_score_for_fusion,
            'sentiment_confidence': sentiment_output['confidence_score'],
            'num_pos_aspects': absa_output['num_pos_aspects'],
            'num_neg_aspects': absa_output['num_neg_aspects'],
            'avg_aspect_score': absa_output['avg_aspect_score'],
            'avg_aspect_confidence': absa_output['avg_aspect_confidence'],
            'emotion_score': emotion_output['emotion_score'],
            'emotion_confidence': emotion_output['emotion_confidence'],
            'emotion_distribution': emotion_output['emotion_distribution'],
            'sarcasm_score': sarcasm_output['sarcasm_score'],
            'sarcasm_confidence': sarcasm_output['sarcasm_confidence']
        }
        fused = fuse_signals(signals)
        t7 = time.time()

        # Compile results
        result = {
            'review_id': rid,
            'text': text,  # Include original text for output formatting
            'preprocessing_time': round(t1 - t0, 3),
            'rule_based_time': round(t2 - t1, 3),
            'transformer_sentiment_time': round(t3 - t2, 3),
            'absa_time': round(t4 - t3, 3),
            'emotion_time': round(t5 - t4, 3),
            'sarcasm_time': round(t6 - t5, 3),
            'fusion_time': round(t7 - t6, 3),
            'signals': signals, # This now contains all 13 features for fusion
            'fused': fused,
            'aspects': absa_output['aspect_details'], # Use 'aspect_details' from absa_output
            'emotion_scores': emotion_output, # Pass the whole dict
            'sarcasm': sarcasm_output,
            'rule': rule_output,
            'sentiment': sentiment_output
        }

        all_results.append(result)

        # Log structured result (for debugging)
        logger.info(json.dumps(result, indent=2))

    total_time = time.time() - total_start
    logger.info(f"\nProcessed {len(SAMPLE_REVIEWS)} reviews in {total_time:.2f}s")
    
    # Save human-readable results to output file
    output_file = save_results_to_file(all_results)
    logger.info(f"📄 Human-readable results saved to: {output_file}")
    print(f"\n✅ Pipeline completed! Results saved to: {output_file}")


if __name__ == '__main__':
    run_tests()
