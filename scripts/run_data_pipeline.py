"""Main script to run the complete data pipeline for YouTube Shorts."""

import sys
import time
from datetime import datetime
from pathlib import Path

import click

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline import YouTubeScraper, VideoDownloader, Transcriber
from src.utils import DataPipelineConfig, load_config, setup_logger


def log_refresh(db_path: Path, stats: dict, start_time: float) -> None:
    """
    Log pipeline execution to database.
    
    Args:
        db_path: Path to database
        stats: Statistics dictionary
        start_time: Pipeline start time
    """
    import sqlite3
    
    duration = time.time() - start_time
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO refresh_log
        (refresh_date, videos_fetched, videos_downloaded, videos_transcribed,
         duration_seconds, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        stats.get('fetched', 0),
        stats.get('downloaded', 0),
        stats.get('transcribed', 0),
        duration,
        stats.get('status', 'completed'),
        stats.get('notes', ''),
    ))
    
    conn.commit()
    conn.close()


@click.command()
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    default=Path('config/data_pipeline.yaml'),
    help='Path to configuration file'
)
@click.option(
    '--skip-scrape',
    is_flag=True,
    help='Skip metadata scraping step'
)
@click.option(
    '--skip-download',
    is_flag=True,
    help='Skip video download step'
)
@click.option(
    '--skip-transcribe',
    is_flag=True,
    help='Skip transcription step'
)
@click.option(
    '--max-videos',
    type=int,
    default=None,
    help='Maximum number of videos to process (for testing)'
)
@click.option(
    '--setup-db',
    is_flag=True,
    help='Initialize database before running pipeline'
)
def main(
    config: Path,
    skip_scrape: bool,
    skip_download: bool,
    skip_transcribe: bool,
    max_videos: int,
    setup_db: bool,
):
    """
    Run the complete YouTube Shorts data pipeline.
    
    This script orchestrates:
    1. Scraping metadata from YouTube
    2. Downloading Shorts videos
    3. Generating transcripts with Whisper
    """
    logger = setup_logger(__name__)
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("YouTube Shorts Data Pipeline")
    logger.info("=" * 60)
    
    # Load configuration
    try:
        logger.info(f"Loading configuration from: {config}")
        pipeline_config = load_config(config, DataPipelineConfig)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Setup database if requested
    if setup_db:
        logger.info("Setting up database...")
        from scripts.setup_database import create_database
        create_database(pipeline_config.database_path)
    
    stats = {
        'fetched': 0,
        'downloaded': 0,
        'transcribed': 0,
        'status': 'completed',
        'notes': '',
    }
    
    try:
        # Step 1: Scrape metadata
        if not skip_scrape:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 1: Scraping YouTube Shorts Metadata")
            logger.info("=" * 60)
            
            scraper = YouTubeScraper(
                api_key=pipeline_config.youtube.api_key,
                channel_id=pipeline_config.youtube.channel_id,
                db_path=pipeline_config.database_path,
                max_duration_seconds=pipeline_config.youtube.max_duration_seconds,
                max_results=pipeline_config.youtube.max_results,
                retry_attempts=pipeline_config.retry_attempts,
                retry_delay=pipeline_config.retry_delay,
            )
            
            stats['fetched'] = scraper.scrape_and_save()
            logger.info(f"Scraped {stats['fetched']} Shorts")
        else:
            logger.info("Skipping metadata scraping (--skip-scrape)")
        
        # Step 2: Download videos
        if not skip_download:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 2: Downloading YouTube Shorts Videos")
            logger.info("=" * 60)
            
            downloader = VideoDownloader(
                db_path=pipeline_config.database_path,
                output_dir=pipeline_config.raw_dir / "shorts_videos",
                video_format=pipeline_config.video_format,
                retry_attempts=pipeline_config.retry_attempts,
            )
            
            download_stats = downloader.download_all_pending(max_videos=max_videos)
            stats['downloaded'] = download_stats['successful']
            
            logger.info(
                f"Downloaded {download_stats['successful']} videos "
                f"({download_stats['failed']} failed)"
            )
        else:
            logger.info("Skipping video downloads (--skip-download)")
        
        # Step 3: Generate transcripts
        if not skip_transcribe:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 3: Generating Transcripts with Whisper")
            logger.info("=" * 60)
            
            transcriber = Transcriber(
                db_path=pipeline_config.database_path,
                output_dir=pipeline_config.raw_dir / "shorts_transcripts",
                model_size=pipeline_config.whisper_model,
                language=pipeline_config.transcription_language,
            )
            
            transcribe_stats = transcriber.transcribe_all_pending(max_videos=max_videos)
            stats['transcribed'] = transcribe_stats['successful']
            
            logger.info(
                f"Transcribed {transcribe_stats['successful']} videos "
                f"({transcribe_stats['failed']} failed)"
            )
        else:
            logger.info("Skipping transcription (--skip-transcribe)")
        
        # Log execution
        log_refresh(pipeline_config.database_path, stats, start_time)
        
        # Summary
        duration = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Execution Complete")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Videos fetched: {stats['fetched']}")
        logger.info(f"Videos downloaded: {stats['downloaded']}")
        logger.info(f"Videos transcribed: {stats['transcribed']}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        stats['status'] = 'interrupted'
        stats['notes'] = 'User interrupted'
        log_refresh(pipeline_config.database_path, stats, start_time)
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}", exc_info=True)
        stats['status'] = 'failed'
        stats['notes'] = str(e)
        log_refresh(pipeline_config.database_path, stats, start_time)
        sys.exit(1)


if __name__ == "__main__":
    main()
