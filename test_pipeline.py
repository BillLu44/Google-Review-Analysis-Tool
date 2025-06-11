#!/usr/bin/env python3
# test_pipeline.py
# Author: Bill Lu
# Description: Run a suite of sample reviews through the full NLP pipeline for validation and timing.

import time
import json
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
    # {
    #     'review_id': 1,
    #     'text': 'The food was delicious, but the service was painfully slow.'
    # },
    # {
    #     'review_id': 2,
    #     'text': 'I absolutely loved how quick and friendly the staff were!'
    # },
    # {
    #     'review_id': 3,
    #     'text': 'The ambiance was nice, but the burger arrived cold and soggy.'
    # },
    # {
    #     'review_id': 4,
    #     'text': 'Terrible experience: rude staff and a burnt steak ruined our night.'
    # },
    # {
    #     'review_id': 5,
    #     'text': 'Great selection of beers—will definitely come back for more!'
    # }
    {
        'review_id': 6,
        'text': "Amazing food, but food is awful"
    },
    {
        'review_id': 6,
        'text': "Amazing food, but burger is awful"
    },
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
