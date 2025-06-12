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
import time # Add time import for timing
import datetime

from utils.output_formatter import save_results_to_file

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


def upsert_results(conn, review: Dict, result_components: Dict, absa_details: List[Dict]) -> None:
    """
    Upsert overall NLP results into nlp_review_results and aspects into nlp_review_aspects.
    """
    review_id = review['review_id']
    # text = review['text'] # text is not directly used here, but good to have for context if needed

    # Determine top emotion for DB logging
    top_emotion_label = 'neutral'
    top_emotion_score = 0.0
    emotion_output_db = result_components.get('emotion_output', {})
    if emotion_output_db and emotion_output_db.get('all_emotion_scores'):
        all_emotions = emotion_output_db['all_emotion_scores']
        if all_emotions:
            top_emotion_label = max(all_emotions, key=all_emotions.get)
            top_emotion_score = all_emotions[top_emotion_label]

    # Upsert main results
    upsert_sql = """
    INSERT INTO nlp_review_results
      (review_id, overall_sentiment, overall_score,
       emotion_label, emotion_score, 
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
    sentiment_output_db = result_components.get('sentiment_output', {})
    sarcasm_output_db = result_components.get('sarcasm_output', {})
    fused_output_db = result_components.get('fused', {})

    params = (
        review_id,
        sentiment_output_db.get('sentiment_label', 'neutral'), # Assuming sentiment_output has sentiment_label for DB
        sentiment_output_db.get('sentiment', 0),             # This is the int score
        top_emotion_label,
        top_emotion_score,
        sarcasm_output_db.get('sarcasm_label', 'not_sarcastic'),
        sarcasm_output_db.get('sarcasm_score', 0), # This is the binary score
        fused_output_db.get('fused_label', 'neutral'),
        fused_output_db.get('fused_confidence', 0.0)
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
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(review_id) FROM nlp_review_results")
                max_id_result = cur.fetchone()
                if max_id_result and max_id_result[0] is not None:
                    last_id = max_id_result[0]
        except Exception as e:
            global_logger.error(f"Could not fetch max review_id from nlp_review_results: {e}. Defaulting to last_id = 0.")
            last_id = 0 # Fallback if table doesn't exist or other error
        global_logger.info(f"Delta mode: processing reviews with ID > {last_id}")
    else:
        global_logger.info("Full mode: processing all reviews.")

    # Fetch reviews
    reviews = fetch_reviews(conn, reviews_table, id_col, text_col, last_id, args.limit)
    if not reviews:
        global_logger.info("No new reviews to process. Exiting.")
        conn.close()
        return

    all_results_for_file = [] # Store results formatted for file output

    for review in reviews:
        start_time = time.time()
        rid = review['review_id']
        text = review['text']
        global_logger.info(f"Processing review_id={rid}")

        # Pipeline stages
        preprocess_result = preprocess_text(text)
        rule_output = rule_based_sentiment(text)
        sentiment_output = analyze_sentiment(text) # Returns {'sentiment': int, 'confidence_score': float}
        absa_output = analyze_absa(text)
        emotion_output = detect_emotion(text)
        sarcasm_output = detect_sarcasm(text)
        
        transformer_sentiment_score_for_fusion = 0.0
        if sentiment_output['sentiment'] == 1:
            transformer_sentiment_score_for_fusion = sentiment_output['confidence_score']
        elif sentiment_output['sentiment'] == -1:
            transformer_sentiment_score_for_fusion = -sentiment_output['confidence_score']

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

        # Prepare components for DB and feedback logging
        # Add 'sentiment_label' to sentiment_output for DB consistency if needed
        db_sentiment_label = "neutral"
        if sentiment_output['sentiment'] == 1: db_sentiment_label = "positive"
        elif sentiment_output['sentiment'] == -1: db_sentiment_label = "negative"
        
        sentiment_output_for_db = sentiment_output.copy()
        sentiment_output_for_db['sentiment_label'] = db_sentiment_label

        result_components_for_db_and_log = {
            'review_id': rid,
            'text': text,
            'sentiment_output': sentiment_output_for_db, # Includes sentiment_label
            'emotion_output': emotion_output,
            'sarcasm_output': sarcasm_output,
            'fused': fused_output,
            'rule_output': rule_output,
            'absa_output': absa_output,
            'signals_fed_to_fusion': signals_for_fusion
        }

        # Prepare the 'signals' dictionary for format_signals_summary
        overall_sentiment_display_label = "neutral"
        if sentiment_output['sentiment'] == 1:
            overall_sentiment_display_label = "positive"
        elif sentiment_output['sentiment'] == -1:
            overall_sentiment_display_label = "negative"

        signals_for_file_display = {
            'OverallSentiment (Transformer)': {
                'label': overall_sentiment_display_label,
                'confidence': sentiment_output['confidence_score']
            },
            'RuleBasedSentiment (VADER/TextBlob)': {
                'label': rule_output['rule_label'],
                'confidence': abs(rule_output['rule_polarity']) # Using absolute polarity as confidence proxy
            }
            # Sarcasm, Emotion, and Aspects are handled by their own dedicated formatters
            # in format_single_result, so they are not strictly needed here.
        }
        
        end_time = time.time()
        total_time_taken = end_time - start_time

        # This is the dict that goes into the list for save_results_to_file
        result_for_file = {
            'review_id': rid,
            'text': text,
            'fused': fused_output,
            'aspects': absa_output.get('aspect_details', []),
            'emotion_scores': emotion_output,
            'sarcasm': sarcasm_output,
            'signals': signals_for_file_display, # Correctly structured for format_signals_summary
            'timing': {
                'total_time': total_time_taken,
                # Individual component times could be added here if tracked
            }
        }
        all_results_for_file.append(result_for_file)

        # Upsert into DB
        try:
            upsert_results(conn, review, result_components_for_db_and_log, absa_output.get('aspect_details', []))
        except Exception as e:
            global_logger.error(f"DB upsert failed for review_id={rid}: {e}", exc_info=True)
            conn.rollback()

        # Log for feedback
        if args.log_feedback and fused_output['fused_confidence'] < args.conf_threshold:
            log_feedback(rid, text, result_components_for_db_and_log)

    if all_results_for_file:
        output_filename = args.output_file if args.output_file else None
        saved_filepath = save_results_to_file(all_results_for_file, filename=output_filename)
        global_logger.info(f"Formatted results saved to: {saved_filepath}")
    else:
        global_logger.info("No results to save to file.")

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
    parser.add_argument('--output-file', type=str, default=None, help='Optional: Specify output file name for formatted results')
    args = parser.parse_args()
    main(args)
