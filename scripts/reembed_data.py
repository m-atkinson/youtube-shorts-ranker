"""
Script to re-embed all data in the database using Gemini embeddings.
Run this ONCE to migrate the dataset.
"""
import sys
import os
from pathlib import Path
import sqlite3
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.embedder import TranscriptEmbedder
from src.utils import setup_logger
from dotenv import load_dotenv

load_dotenv()

def main():
    logger = setup_logger("reembed_data")
    
    # Configuration
    DB_PATH = "data/database/shorts_data.db"
    MODEL_NAME = "models/text-embedding-004"
    
    if not os.environ.get("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found. Please set it.")
        sys.exit(1)
        
    logger.info(f"Initializing Gemini Embedder ({MODEL_NAME})...")
    embedder = TranscriptEmbedder(model_name=MODEL_NAME)
    
    # Connect to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Fetch all videos
    cursor.execute("SELECT video_id, transcript_text FROM transcripts WHERE transcript_text IS NOT NULL")
    rows = cursor.fetchall()
    
    if not rows:
        logger.warning("No videos found with transcripts.")
        return
        
    logger.info(f"Found {len(rows)} videos to re-embed.")
    
    video_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    
    # Generate embeddings (this will populate the cache)
    # We don't strictly need to update the DB if the training script loads from cache,
    # but the training script (ShortsDataset) uses the embedder which uses the cache.
    # So simply running this will populate the cache for the new model.
    
    # Note: ShortsDataset usually loads from DB, then calls embedder.embed_batch(texts, video_ids=video_ids)
    # which checks the cache. So we just need to warm the cache.
    
    logger.info("Generating embeddings (warming cache)...")
    try:
        # Process in batches to avoid hitting API limits too hard if we did it all at once,
        # though embedder handles batching.
        embedder.embed_batch(texts, video_ids=video_ids, batch_size=50, use_cache=True)
        logger.info("Successfully re-embedded all data!")
    except Exception as e:
        logger.error(f"Failed to re-embed data: {e}")
        sys.exit(1)
        
    conn.close()
    logger.info("Done. You can now train the model using this embedding model.")

if __name__ == "__main__":
    main()
