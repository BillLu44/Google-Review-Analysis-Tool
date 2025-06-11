# pipeline/rule_based.py
# Author: Your Name
# Description: Enhanced rule-based sentiment module merging VADER and TextBlob outputs into a single rule_score feature.

import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from pipeline.logger import get_logger
from pipeline.preprocessing import preprocess_text # Added import

logger = get_logger(__name__)

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()


def rule_based_sentiment(text: str) -> dict:
    """
    Run rule-based sentiment using VADER and TextBlob on a phrase-by-phrase basis,
    then aggregate into overall scores.

    Returns a dict with:
      - vader_score: float [-1,1] (average VADER compound score across sentences)
      - textblob_polarity: float [-1,1] (average TextBlob polarity across sentences)
      - rule_polarity: float [-1,1] (primary aggregated polarity score)
      - rule_score: int (-1, 0, 1) (integer representation of rule_label)
      - rule_label: str ('positive', 'negative', 'neutral')
    """
    logger.debug(f"Applying rule-based analysis to: '{text[:100]}...'")

    processed_info = preprocess_text(text)
    sentences = processed_info.get('sentences', [])

    # Filter out very short or empty sentences
    meaningful_sentences = [s for s in sentences if s.strip() and len(s.split()) >= 2]

    if not meaningful_sentences:
        logger.debug("No meaningful sentences found after preprocessing, processing text as a whole.")
        # Fallback to processing the entire text if no meaningful sentences
        vader_res = vader_analyzer.polarity_scores(text)
        vader_score_overall = vader_res['compound']
        
        blob = TextBlob(text)
        tb_polarity_overall = blob.sentiment.polarity
        
        avg_vader_score = vader_score_overall
        avg_tb_polarity = tb_polarity_overall
        rule_polarity = (vader_score_overall + tb_polarity_overall) / 2
    else:
        sentence_vader_scores = []
        sentence_tb_polarities = []
        sentence_combined_polarities = []

        for i, sentence_text in enumerate(meaningful_sentences):
            logger.debug(f"Processing sentence {i+1}/{len(meaningful_sentences)}: '{sentence_text}'")
            vader_res = vader_analyzer.polarity_scores(sentence_text)
            v_score = vader_res['compound']
            
            blob = TextBlob(sentence_text)
            tb_pol = blob.sentiment.polarity
            
            sentence_vader_scores.append(v_score)
            sentence_tb_polarities.append(tb_pol)
            sentence_combined_polarities.append((v_score + tb_pol) / 2)

        if sentence_combined_polarities:
            avg_vader_score = sum(sentence_vader_scores) / len(sentence_vader_scores)
            avg_tb_polarity = sum(sentence_tb_polarities) / len(sentence_tb_polarities)
            rule_polarity = sum(sentence_combined_polarities) / len(sentence_combined_polarities)
        else:
            # This case should ideally not be hit if meaningful_sentences was not empty,
            # but as a robust fallback:
            logger.debug("No sentence polarities calculated, processing text as a whole as fallback.")
            vader_res = vader_analyzer.polarity_scores(text)
            vader_score_overall = vader_res['compound']
            blob = TextBlob(text)
            tb_polarity_overall = blob.sentiment.polarity
            avg_vader_score = vader_score_overall
            avg_tb_polarity = tb_polarity_overall
            rule_polarity = (vader_score_overall + tb_polarity_overall) / 2

    # Derive label and integer score from the aggregated rule_polarity
    if rule_polarity >= 0.05:
        rule_label = 'positive'
        rule_score_int = 1
    elif rule_polarity <= -0.05:
        rule_label = 'negative'
        rule_score_int = -1
    else:
        rule_label = 'neutral'
        rule_score_int = 0

    logger.debug(
        f"Rule-based aggregated -> Avg VADER: {avg_vader_score:.3f}, Avg TB Polarity: {avg_tb_polarity:.3f}, "
        f"Rule Polarity: {rule_polarity:.3f}, Label: {rule_label}, Int Score: {rule_score_int}"
    )

    return {
        'vader_score': avg_vader_score,
        'textblob_polarity': avg_tb_polarity,
        'rule_polarity': rule_polarity,
        'rule_score': rule_score_int, # Integer score
        'rule_label': rule_label
    }
