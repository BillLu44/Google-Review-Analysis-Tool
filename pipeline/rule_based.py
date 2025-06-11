# pipeline/rule_based.py
# Author: Your Name
# Description: Enhanced rule-based sentiment module merging VADER and TextBlob outputs into a single rule_score feature.

import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from pipeline.logger import get_logger

logger = get_logger(__name__)

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()


def rule_based_sentiment(text: str) -> dict:
    """
    Run rule-based sentiment using VADER and TextBlob, then merge into a single rule_score.

    Returns a dict with:
      - vader_score: float [-1,1]
      - textblob_polarity: float [-1,1]
      - textblob_subjectivity: float [0,1]
      - rule_score: float [-1,1]  # averaged score used by fusion
      - rule_label: str           # derived label from rule_score
    """
    logger.debug("Applying VADER analysis")
    vader_res = vader_analyzer.polarity_scores(text)
    vader_score = vader_res['compound']

    logger.debug("Applying TextBlob analysis")
    blob = TextBlob(text)
    tb_polarity = blob.sentiment.polarity
    tb_subjectivity = blob.sentiment.subjectivity

    # Redundancy check: if both agree in sign, merge directly; otherwise, weight by confidence
    same_sign = (vader_score >= 0 and tb_polarity >= 0) or (vader_score < 0 and tb_polarity < 0)
    if same_sign:
        # average magnitude
        rule_score = (vader_score + tb_polarity) / 2
    else:
        # pick the stronger signal
        rule_score = vader_score if abs(vader_score) > abs(tb_polarity) else tb_polarity

    # Derive label
    if rule_score >= 0.05:
        rule_label = 'positive'
    elif rule_score <= -0.05:
        rule_label = 'negative'
    else:
        rule_label = 'neutral'

    logger.debug(
        f"Rule-based merged -> vader: {vader_score:.3f}, tb: {tb_polarity:.3f}, "
        f"rule_score: {rule_score:.3f}, label: {rule_label}"
    )

    return {
        'vader_score': vader_score,
        'textblob_polarity': tb_polarity,
        'textblob_subjectivity': tb_subjectivity,
        'rule_score': rule_score,
        'rule_label': rule_label
    }
