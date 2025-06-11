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
    {
        'review_id': 1,
        'text': 'The food was delicious, but the service was painfully slow.'
    },
    {
        'review_id': 2,
        'text': 'I absolutely loved how quick and friendly the staff were!'
    },
    {
        'review_id': 3,
        'text': 'The ambiance was nice, but the burger arrived cold and soggy.'
    },
    {
        'review_id': 4,
        'text': 'Terrible experience: rude staff and a burnt steak ruined our night.'
    },
    {
        'review_id': 5,
        'text': 'Great selection of beers—will definitely come back for more!'
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
        rule = rule_based_sentiment(text)
        t2 = time.time()

        # 3. Transformer sentiment
        sentiment = analyze_sentiment(text)
        t3 = time.time()

        # 4. Aspect-based sentiment
        aspects = analyze_absa(text)
        t4 = time.time()

        # 5. Emotion detection
        emotion_scores = detect_emotion(text)
        t5 = time.time()

        # 6. Sarcasm detection
        sarcasm = detect_sarcasm(text)
        t6 = time.time()

        # Compute aspect summary for fusion input
        num_pos_aspects = sum(1 for a in aspects if a['sentiment_label'] == 'positive')
        num_neg_aspects = sum(1 for a in aspects if a['sentiment_label'] == 'negative')
        avg_aspect_score = (sum(a['sentiment_score'] for a in aspects) / len(aspects)) if aspects else 0.0

        # 7. Fusion
        signals = {
            'rule_score': rule['rule_score'],
            'sentiment_score': sentiment['sentiment_score'],
            'num_pos_aspects': num_pos_aspects,
            'num_neg_aspects': num_neg_aspects,
            'avg_aspect_score': avg_aspect_score,
            'emotion_score': max(emotion_scores.values()) if emotion_scores else 0.0,
            'sarcasm_score': sarcasm.get('sarcasm_score', 0.0)
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
            'signals': signals,
            'fused': fused,
            'aspects': aspects,
            'emotion_scores': emotion_scores,
            'sarcasm': sarcasm,
            'rule': rule,
            'sentiment': sentiment
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
