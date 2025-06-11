#!/usr/bin/env python3
# run_nlp_pipeline.py
# Author: Bill Lu
# Description: Orchestrates the full NLP pipeline: fetch reviews, process stages, and store results.

import os
import json
import argparse
from typing import List, Dict
import psycopg2
from psycopg2.extras import RealDictCursor
from pipeline.logger import get_logger
from pipeline.db_utils import get_db_connection, fetch_absa_categories
from pipeline.preprocessing import preprocess_text
from pipeline.rule_based import rule_based_sentiment
from pipeline.overall_sentiment import analyze_sentiment
from pipeline.absa import analyze_absa
from pipeline.emotion import detect_emotion
from pipeline.sarcasm import detect_sarcasm
from pipeline.fusion import fuse_signals
from pipeline.feedback_loop import log_feedback

# Initialize logger
global_logger = get_logger(__name__)

def fetch_reviews(conn, table: str, id_col: str, text_col: str, last_id: int = 0, limit: int = None) -> List[Dict]:
    """
    Fetch reviews from the database for processing.
    """
    sql = f"SELECT {id_col} AS review_id, {text_col} AS text FROM {table}"
    params = []
    if last_id > 0:
        sql += f" WHERE {id_col} > %s"
        params.append(last_id)
    sql += f" ORDER BY {id_col} ASC"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, tuple(params))
        return cur.fetchall()


def upsert_results(conn, review: Dict, result: Dict, absa_details: List[Dict]) -> None:
    """
    Upsert overall NLP results into nlp_review_results and aspects into nlp_review_aspects.
    """
    review_id = review['review_id']
    # text = review['text'] # text is not directly used here, but good to have for context if needed

    # Determine top emotion for DB logging (if still desired in this format)
    # This assumes 'all_emotion_scores' is part of the result['emotion_output']
    top_emotion_label = 'neutral'
    top_emotion_score = 0.0
    if result.get('emotion_output') and result['emotion_output'].get('all_emotion_scores'):
        all_emotions = result['emotion_output']['all_emotion_scores']
        if all_emotions:
            top_emotion_label = max(all_emotions, key=all_emotions.get)
            top_emotion_score = all_emotions[top_emotion_label]

    # Upsert main results
    upsert_sql = """
    INSERT INTO nlp_review_results
      (review_id, overall_sentiment, overall_score,
       emotion_label, emotion_score, -- These might represent the 'top' emotion
       sarcasm_label, sarcasm_score,
       fusion_label, fusion_confidence,
       created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    ON CONFLICT (review_id) DO UPDATE SET
      overall_sentiment = EXCLUDED.overall_sentiment,
      overall_score = EXCLUDED.overall_score,
      emotion_label = EXCLUDED.emotion_label,
      emotion_score = EXCLUDED.emotion_score,
      sarcasm_label = EXCLUDED.sarcasm_label,
      sarcasm_score = EXCLUDED.sarcasm_score,
      fusion_label = EXCLUDED.fusion_label,
      fusion_confidence = EXCLUDED.fusion_confidence,
      updated_at = NOW();
    """
    params = (
        review_id,
        result['sentiment_output']['sentiment_label'] if result.get('sentiment_output') else 'neutral', # Assuming sentiment_output has sentiment_label
        result['sentiment_output']['sentiment'] if result.get('sentiment_output') else 0,             # This is the int score
        top_emotion_label,
        top_emotion_score,
        result['sarcasm_output']['sarcasm_label'] if result.get('sarcasm_output') else 'not_sarcastic',
        result['sarcasm_output']['sarcasm_score'] if result.get('sarcasm_output') else 0, # This is the binary score
        result['fused']['fused_label'],
        result['fused']['fused_confidence']
    )
    with conn.cursor() as cur:
        cur.execute(upsert_sql, params)

    # Upsert aspects: delete existing then insert
    del_sql = "DELETE FROM nlp_review_aspects WHERE review_id = %s"
    with conn.cursor() as cur:
        cur.execute(del_sql, (review_id,))

    insert_sql = """
    INSERT INTO nlp_review_aspects
      (review_id, category_id, span_text, aspect_sentiment, aspect_confidence, created_at)
    VALUES (%s, %s, %s, %s, %s, NOW())
    """
    # Fetch aspect category mapping
    absa_rows = fetch_absa_categories(None)
    aspect_to_id = {row['aspect'].lower(): row['category_id'] for row in absa_rows}

    with conn.cursor() as cur:
        for aspect in absa_details: # Use absa_details
            aspect_norm = aspect['aspect']
            category_id = aspect_to_id.get(aspect_norm.lower()) # Ensure lowercase for matching
            if category_id is None:
                global_logger.warning(f"No category_id for aspect '{aspect_norm}', skipping DB insert.")
                continue
            span = aspect_norm
            sentiment = aspect['sentiment_label']
            score = aspect['sentiment_score']
            cur.execute(insert_sql, (review_id, category_id, span, sentiment, score))

    conn.commit()
    global_logger.info(f"Upserted results for review_id={review_id}")


def main(args):
    # DB setup
    conn = get_db_connection()
    reviews_table = args.reviews_table
    id_col = args.id_col
    text_col = args.text_col

    # Determine last processed ID
    last_id = 0
    if not args.full:
        with conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(review_id), 0) FROM nlp_review_results")
            last_id = cur.fetchone()[0]
        global_logger.info(f"Delta mode: processing reviews with ID > {last_id}")
    else:
        global_logger.info("Full mode: processing all reviews.")

    # Fetch reviews
    reviews = fetch_reviews(conn, reviews_table, id_col, text_col, last_id, args.limit)
    if not reviews:
        global_logger.info("No reviews to process. Exiting.")
        return

    for review in reviews:
        rid = review['review_id']
        text = review['text']
        global_logger.info(f"Processing review_id={rid}")

        # Pipeline stages
        # preprocess_result = preprocess_text(text) # 'preprocess_result' not used later, direct call is fine
        rule_output = rule_based_sentiment(text)
        sentiment_output = analyze_sentiment(text)
        absa_output = analyze_absa(text) # Returns a dict
        emotion_output = detect_emotion(text) # Returns a dict
        sarcasm_output = detect_sarcasm(text)
        
        # Adapt sentiment_output for fusion model's expected sentiment_score (-1 to 1 float)
        transformer_sentiment_score_for_fusion = 0.0
        if sentiment_output['sentiment'] == 1: # positive
            transformer_sentiment_score_for_fusion = sentiment_output['confidence_score']
        elif sentiment_output['sentiment'] == -1: # negative
            transformer_sentiment_score_for_fusion = -sentiment_output['confidence_score']

        # Collect signals for fusion
        signals_for_fusion = {
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
        fused_output = fuse_signals(signals_for_fusion)

        # Compile result record for DB and feedback
        # This 'result_for_db' can be a subset or a differently structured dict for DB needs
        result_for_db_and_feedback = {
            'sentiment_output': sentiment_output, # Contains original int score and label
            'emotion_output': emotion_output, # Contains all emotion scores for flexibility
            'sarcasm_output': sarcasm_output,
            # 'aspects': absa_output['aspect_details'], # Passed separately to upsert_results
            'fused': fused_output,
            # Include other raw outputs if needed for DB or feedback
            'rule_output': rule_output,
            'absa_output': absa_output
        }

        # Upsert into DB
        try:
            # Pass absa_output['aspect_details'] explicitly
            upsert_results(conn, review, result_for_db_and_feedback, absa_output['aspect_details'])
        except Exception as e:
            global_logger.error(f"Failed to upsert results for review_id={rid}: {e}")

        # Log feedback if low confidence
        if args.log_feedback and fused_output['fused_confidence'] < args.conf_threshold:
            # For feedback, log the signals that went into fusion, and the fusion output itself.
            # You might also want to log the more detailed raw outputs from each module.
            feedback_payload = {
                'signals_fed_to_fusion': signals_for_fusion,
                'fused_output': fused_output,
                # Optionally add more raw component outputs if useful for retraining/analysis
                'rule_details': rule_output,
                'sentiment_details': sentiment_output,
                'absa_details': absa_output,
                'emotion_details': emotion_output,
                'sarcasm_details': sarcasm_output
            }
            log_feedback(rid, text, feedback_payload)

    global_logger.info("Pipeline run complete.")
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the full NLP sentiment analysis pipeline.")
    parser.add_argument('--full', action='store_true', help='Process all reviews (default: delta only)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of reviews to process')
    parser.add_argument('--log-feedback', action='store_true', help='Log low-confidence results for feedback')
    parser.add_argument('--conf-threshold', type=float, default=0.6, help='Confidence threshold for logging feedback')
    parser.add_argument('--reviews-table', type=str, default='reviews', help='Name of the reviews table')
    parser.add_argument('--id-col', type=str, default='review_id', help='Primary key column in reviews table')
    parser.add_argument('--text-col', type=str, default='text', help='Text column in reviews table')
    args = parser.parse_args()
    main(args)
