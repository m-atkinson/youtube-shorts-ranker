"""Video downloader using yt-dlp for YouTube Shorts."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yt_dlp

from ..utils.logger import setup_logger


class VideoDownloader:
    """Download YouTube Shorts videos using yt-dlp."""
    
    def __init__(
        self,
        db_path: Path,
        output_dir: Path,
        video_format: str = "mp4",
        retry_attempts: int = 3,
    ):
        """
        Initialize video downloader.
        
        Args:
            db_path: Path to SQLite database
            output_dir: Directory to save downloaded videos
            video_format: Desired video format
            retry_attempts: Number of download retry attempts
        """
        self.db_path = db_path
        self.output_dir = output_dir
        self.video_format = video_format
        self.retry_attempts = retry_attempts
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(__name__)
    
    def get_pending_downloads(self) -> List[str]:
        """
        Get list of video IDs that need to be downloaded.
        
        Returns:
            List of video IDs pending download
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT video_id FROM download_status
            WHERE video_downloaded = 0 OR status = 'failed'
            ORDER BY last_attempt_at IS NULL DESC, last_attempt_at ASC
        """)
        
        video_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return video_ids
    
    def download_video(self, video_id: str) -> bool:
        """
        Download a single video.
        
        Args:
            video_id: YouTube video ID
        
        Returns:
            True if successful, False otherwise
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        output_path = self.output_dir / f"{video_id}.{self.video_format}"
        metadata_path = self.output_dir / f"{video_id}.info.json"
        
        # Skip if already downloaded
        if output_path.exists():
            self.logger.info(f"Video {video_id} already exists, skipping download")
            self._update_download_status(video_id, True, str(output_path), None)
            return True
        
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': str(self.output_dir / f'{video_id}.%(ext)s'),
            'writeinfojson': True,
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': False,
        }
        
        for attempt in range(self.retry_attempts):
            try:
                self.logger.info(
                    f"Downloading video {video_id} "
                    f"(attempt {attempt + 1}/{self.retry_attempts})"
                )
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                if output_path.exists():
                    self.logger.info(f"Successfully downloaded {video_id}")
                    self._update_download_status(video_id, True, str(output_path), None)
                    return True
                else:
                    error_msg = "Video file not found after download"
                    self.logger.error(f"{error_msg}: {video_id}")
                    
            except Exception as e:
                error_msg = str(e)
                self.logger.error(
                    f"Error downloading {video_id} "
                    f"(attempt {attempt + 1}/{self.retry_attempts}): {error_msg}"
                )
                
                if attempt == self.retry_attempts - 1:
                    self._update_download_status(video_id, False, None, error_msg)
                    return False
        
        return False
    
    def _update_download_status(
        self,
        video_id: str,
        success: bool,
        video_path: Optional[str],
        error_message: Optional[str]
    ) -> None:
        """
        Update download status in database.
        
        Args:
            video_id: YouTube video ID
            success: Whether download was successful
            video_path: Path to downloaded video (if successful)
            error_message: Error message (if failed)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = datetime.now().isoformat()
        status = 'completed' if success else 'failed'
        
        cursor.execute("""
            UPDATE download_status
            SET video_downloaded = ?,
                video_path = ?,
                status = ?,
                error_message = ?,
                download_attempts = download_attempts + 1,
                last_attempt_at = ?
            WHERE video_id = ?
        """, (
            1 if success else 0,
            video_path,
            status,
            error_message,
            current_time,
            video_id,
        ))
        
        conn.commit()
        conn.close()
    
    def download_all_pending(self, max_videos: Optional[int] = None) -> Dict[str, int]:
        """
        Download all pending videos.
        
        Args:
            max_videos: Maximum number of videos to download (None for all)
        
        Returns:
            Dictionary with download statistics
        """
        pending_ids = self.get_pending_downloads()
        
        if max_videos:
            pending_ids = pending_ids[:max_videos]
        
        self.logger.info(f"Starting download of {len(pending_ids)} videos")
        
        stats = {
            'total': len(pending_ids),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
        }
        
        for i, video_id in enumerate(pending_ids, 1):
            self.logger.info(f"Processing video {i}/{len(pending_ids)}: {video_id}")
            
            success = self.download_video(video_id)
            
            if success:
                stats['successful'] += 1
            else:
                stats['failed'] += 1
        
        self.logger.info(
            f"Download complete: {stats['successful']} successful, "
            f"{stats['failed']} failed out of {stats['total']} total"
        )
        
        return stats
    
    def get_downloaded_videos(self) -> List[Dict[str, str]]:
        """
        Get list of successfully downloaded videos.
        
        Returns:
            List of dictionaries with video_id and video_path
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT video_id, video_path
            FROM download_status
            WHERE video_downloaded = 1 AND video_path IS NOT NULL
        """)
        
        videos = [
            {'video_id': row[0], 'video_path': row[1]}
            for row in cursor.fetchall()
        ]
        
        conn.close()
        return videos
    
    def cleanup_failed_downloads(self) -> int:
        """
        Remove partial/corrupted files from failed downloads.
        
        Returns:
            Number of files cleaned up
        """
        self.logger.info("Cleaning up failed downloads...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT video_id FROM download_status
            WHERE status = 'failed' OR (video_downloaded = 0 AND download_attempts > 0)
        """)
        
        failed_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        cleaned_count = 0
        for video_id in failed_ids:
            for ext in ['mp4', 'webm', 'part', 'ytdl']:
                file_path = self.output_dir / f"{video_id}.{ext}"
                if file_path.exists():
                    file_path.unlink()
                    cleaned_count += 1
                    self.logger.debug(f"Removed: {file_path}")
        
        self.logger.info(f"Cleaned up {cleaned_count} files")
        return cleaned_count
