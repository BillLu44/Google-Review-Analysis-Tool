# pipeline/preprocessing.py
# Author: Bill Lu
# Description: Production‑ready spaCy preprocessing for tokenization, sentence splitting, POS tagging, lemmatization, and noun‑chunk extraction.

import os
import json
import spacy
from pipeline.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
try:
    with open(CONFIG_PATH, 'r') as cf:
        _config = json.load(cf)
except Exception as e:
    logger.error(f"Failed to load config.json: {e}")
    raise

# Determine spaCy model to use
SPACY_MODEL = _config.get('spacy_model', 'en_core_web_sm')

# Load or download the spaCy model
try:
    nlp = spacy.load(SPACY_MODEL)
    logger.info(f"Loaded spaCy model '{SPACY_MODEL}'")
except OSError:
    from spacy.cli import download
    logger.warning(f"spaCy model '{SPACY_MODEL}' not found, attempting download...")
    download(SPACY_MODEL)
    nlp = spacy.load(SPACY_MODEL)
    logger.info(f"Successfully downloaded and loaded spaCy model '{SPACY_MODEL}'")

# Customize pipeline: ensure required components are present
required_components = ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer']
for comp in required_components:
    if comp not in nlp.pipe_names:
        logger.warning(f"spaCy model missing component '{comp}', this may affect preprocessing quality.")


def preprocess_text(text: str) -> dict:
    """
    Run full spaCy preprocessing on input text.

    Returns a dict containing:
      - doc: spaCy Doc object
      - sentences: list of sentence texts
      - tokens: list of token dicts (text, lemma, POS, tag)
      - noun_chunks: list of noun-chunk texts
    """
    logger.debug("Starting preprocessing")
    doc = nlp(text)

    # Sentence splitting
    sentences = [sent.text.strip() for sent in doc.sents]
    logger.debug(f"Identified {len(sentences)} sentences")

    # Token details
    tokens = []
    for token in doc:
        tokens.append({
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'tag': token.tag_,
            'is_alpha': token.is_alpha,
            'is_stop': token.is_stop
        })
    logger.debug(f"Extracted {len(tokens)} tokens with POS and lemma")

    # Noun chunks
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    logger.debug(f"Extracted {len(noun_chunks)} noun chunks")

    return {
        'doc': doc,
        'sentences': sentences,
        'tokens': tokens,
        'noun_chunks': noun_chunks
    }