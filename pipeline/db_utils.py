# pipeline/db_utils.py
# Author: Bill Lu
# Description: Database utility functions for PostgreSQL (Neon), including connection management and ABSA category retrieval.

import os
import json
from typing import Optional, List, Dict
import psycopg2
from psycopg2.extras import RealDictCursor
from pipeline.logger import get_logger

logger = get_logger(__name__)

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
try:
    with open(CONFIG_PATH, 'r') as cf:
        _config = json.load(cf)
except Exception as e:
    logger.error(f"Failed to load config.json: {e}")
    raise

DB_URL_ENV = _config.get('database', {}).get('url_env_var', 'DATABASE_URL')
ABSA_TABLE = _config.get('database', {}).get('absa_category_table', 'absa_categories')


def get_db_connection():
    """
    Establish and return a new database connection using DATABASE_URL from environment.

    Returns:
        psycopg2.connection: open database connection
    Raises:
        KeyError: if DATABASE_URL not set
        psycopg2.Error: on connection failure
    """
    db_url = os.getenv(DB_URL_ENV)
    if not db_url:
        logger.error(f"Environment variable '{DB_URL_ENV}' not set.")
        raise KeyError(f"Environment variable '{DB_URL_ENV}' not set.")

    try:
        conn = psycopg2.connect(db_url)
        logger.debug("Database connection established.")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise


def fetch_absa_categories(industry_id: Optional[int] = None) -> List[Dict]:
    """
    Fetch ABSA categories and aspects from the database table.

    Args:
        industry_id: Optional industry filter; if provided, only categories for that industry are returned.

    Returns:
        List of dicts with keys: category_id, category, aspect, industry_id (if present in table)
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if industry_id is not None:
                query = f"SELECT * FROM {ABSA_TABLE} WHERE industry_id = %s"
                cur.execute(query, (industry_id,))
                logger.debug(f"Fetched ABSA categories for industry_id={industry_id}.")
            else:
                query = f"SELECT * FROM {ABSA_TABLE}"
                cur.execute(query)
                logger.debug("Fetched all ABSA categories.")

            rows = cur.fetchall()
            return rows
    except Exception as e:
        logger.error(f"Error fetching ABSA categories: {e}")
        raise
    finally:
        conn.close()
        logger.debug("Database connection closed.")
