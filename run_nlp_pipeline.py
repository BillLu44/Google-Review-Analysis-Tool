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
    sql += " ORDER BY {id_col} ASC"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, tuple(params))
        return cur.fetchall()


def upsert_results(conn, review: Dict, result: Dict) -> None:
    """
    Upsert overall NLP results into nlp_review_results and aspects into nlp_review_aspects.
    """
    review_id = review['review_id']
    text = review['text']

    # Upsert main results\    
    upsert_sql = """
    INSERT INTO nlp_review_results
      (review_id, overall_sentiment, overall_score,
       emotion_label, emotion_score,
       sarcasm_label, sarcasm_score,
       fusion_label, fusion_confidence,
       created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
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
        result['sentiment_label'],
        result['sentiment_score'],
        result['emotion_label'],
        result['emotion_score'],
        result['sarcasm_label'],
        result['sarcasm_score'],
        result['fused_label'],
        result['fused_confidence']
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
        for aspect in result['aspects']:
            aspect_norm = aspect['aspect']
            category_id = aspect_to_id.get(aspect_norm)
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
        preprocess = preprocess_text(text)
        rule = rule_based_sentiment(text)
        sentiment = analyze_sentiment(text)
        aspects = analyze_absa(text)
        emotion_scores = detect_emotion(text)
        # pick top emotion
        emotion_label = max(emotion_scores, key=emotion_scores.get)
        emotion_score = emotion_scores[emotion_label]
        sarcasm = detect_sarcasm(text)
        
        # ABSA summary
        num_pos = sum(1 for a in aspects if a['sentiment_label'] == 'positive')
        num_neg = sum(1 for a in aspects if a['sentiment_label'] == 'negative')
        avg_aspect = (sum(a['sentiment_score'] for a in aspects) / len(aspects)) if aspects else 0.0

        # Collect signals for fusion
        signals = {
            'vader_score': rule['vader_score'],
            'sentiment_score': sentiment['sentiment_score'],
            'num_pos_aspects': num_pos,
            'num_neg_aspects': num_neg,
            'avg_aspect_score': avg_aspect,
            'emotion_score': emotion_score,
            'sarcasm_score': sarcasm['sarcasm_score']
        }
        fused = fuse_signals(signals)

        # Compile result record
        result = {
            'sentiment_label': sentiment['sentiment_label'],
            'sentiment_score': sentiment['sentiment_score'],
            'emotion_label': emotion_label,
            'emotion_score': emotion_score,
            'sarcasm_label': sarcasm['sarcasm_label'],
            'sarcasm_score': sarcasm['sarcasm_score'],
            'aspects': aspects,
            'fused_label': fused['fused_label'],
            'fused_confidence': fused['fused_confidence']
        }

        # Upsert into DB
        try:
            upsert_results(conn, review, result)
        except Exception as e:
            global_logger.error(f"Failed to upsert results for review_id={rid}: {e}")

        # Log feedback if low confidence
        if args.log_feedback and fused['fused_confidence'] < args.conf_threshold:
            log_feedback(rid, text, result)

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
