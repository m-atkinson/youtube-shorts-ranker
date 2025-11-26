"""Database setup script for YouTube Shorts data storage."""

import sqlite3
from pathlib import Path
from datetime import datetime


def create_database(db_path: Path) -> None:
    """
    Create SQLite database with required tables.
    
    Args:
        db_path: Path to SQLite database file
    """
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Shorts metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shorts_metadata (
            video_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            publish_date TEXT NOT NULL,
            duration_seconds INTEGER,
            view_count INTEGER,
            like_count INTEGER,
            comment_count INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    
    # Transcripts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            video_id TEXT PRIMARY KEY,
            transcript_text TEXT NOT NULL,
            language TEXT,
            word_count INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (video_id) REFERENCES shorts_metadata(video_id)
        )
    """)
    
    # Download status tracking table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS download_status (
            video_id TEXT PRIMARY KEY,
            video_downloaded BOOLEAN DEFAULT 0,
            video_path TEXT,
            transcript_generated BOOLEAN DEFAULT 0,
            transcript_path TEXT,
            download_attempts INTEGER DEFAULT 0,
            last_attempt_at TEXT,
            status TEXT,
            error_message TEXT,
            FOREIGN KEY (video_id) REFERENCES shorts_metadata(video_id)
        )
    """)
    
    # Data refresh log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS refresh_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            refresh_date TEXT NOT NULL,
            videos_fetched INTEGER,
            videos_downloaded INTEGER,
            videos_transcribed INTEGER,
            duration_seconds REAL,
            status TEXT,
            notes TEXT
        )
    """)
    
    # Create indexes for better query performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_publish_date 
        ON shorts_metadata(publish_date)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_view_count 
        ON shorts_metadata(view_count)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_download_status 
        ON download_status(status)
    """)
    
    conn.commit()
    conn.close()
    
    print(f"Database created successfully at: {db_path}")


def add_sample_queries(db_path: Path) -> None:
    """
    Print sample SQL queries for reference.
    
    Args:
        db_path: Path to SQLite database file
    """
    queries = """
    Sample Queries:
    
    1. Get all shorts with transcripts:
       SELECT sm.*, t.transcript_text 
       FROM shorts_metadata sm
       JOIN transcripts t ON sm.video_id = t.video_id;
    
    2. Get shorts by view count (top performers):
       SELECT * FROM shorts_metadata 
       ORDER BY view_count DESC 
       LIMIT 10;
    
    3. Check download progress:
       SELECT 
         COUNT(*) as total,
         SUM(video_downloaded) as downloaded,
         SUM(transcript_generated) as transcribed
       FROM download_status;
    
    4. Get failed downloads:
       SELECT * FROM download_status 
       WHERE status = 'failed';
    
    5. View refresh history:
       SELECT * FROM refresh_log 
       ORDER BY refresh_date DESC;
    """
    print(queries)


def main():
    """Main entry point for database setup."""
    # Default database path
    db_path = Path("data/database/shorts_data.db")
    
    print("Setting up YouTube Shorts Predictor database...")
    create_database(db_path)
    add_sample_queries(db_path)
    print("\nDatabase setup complete!")


if __name__ == "__main__":
    main()
