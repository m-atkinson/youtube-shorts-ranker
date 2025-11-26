"""YouTube Data API scraper for fetching Shorts metadata."""

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ..utils.logger import setup_logger


class YouTubeScraper:
    """Scraper for fetching YouTube Shorts metadata."""
    
    def __init__(
        self,
        api_key: str,
        channel_id: str,
        db_path: Path,
        max_duration_seconds: int = 60,
        max_results: int = 50,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize YouTube scraper.
        
        Args:
            api_key: YouTube Data API v3 key
            channel_id: Target channel ID
            db_path: Path to SQLite database
            max_duration_seconds: Maximum video duration to consider as Short
            max_results: Maximum results per API call
            retry_attempts: Number of retry attempts for API calls
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.api_key = api_key
        self.channel_id = channel_id
        self.db_path = db_path
        self.max_duration_seconds = max_duration_seconds
        self.max_results = max_results
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.logger = setup_logger(__name__)
    
    def _retry_api_call(self, func, *args, **kwargs):
        """
        Execute API call with exponential backoff retry logic.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            API response
        
        Raises:
            HttpError: If all retry attempts fail
        """
        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except HttpError as e:
                if attempt == self.retry_attempts - 1:
                    raise
                
                delay = self.retry_delay * (2 ** attempt)
                self.logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.retry_attempts}): {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
    
    def fetch_channel_videos(self) -> List[Dict]:
        """
        Fetch all videos from the channel.
        
        Returns:
            List of video metadata dictionaries
        """
        self.logger.info(f"Fetching videos from channel: {self.channel_id}")
        
        videos = []
        next_page_token = None
        
        while True:
            # Get uploads playlist ID
            try:
                channel_response = self._retry_api_call(
                    self.youtube.channels().list,
                    part='contentDetails',
                    id=self.channel_id
                ).execute()
                
                if not channel_response.get('items'):
                    self.logger.error(f"Channel not found: {self.channel_id}")
                    break
                
                uploads_playlist_id = (
                    channel_response['items'][0]['contentDetails']
                    ['relatedPlaylists']['uploads']
                )
                
            except HttpError as e:
                self.logger.error(f"Error fetching channel details: {e}")
                break
            
            # Fetch videos from uploads playlist
            try:
                playlist_response = self._retry_api_call(
                    self.youtube.playlistItems().list,
                    part='snippet',
                    playlistId=uploads_playlist_id,
                    maxResults=self.max_results,
                    pageToken=next_page_token
                ).execute()
                
                video_ids = [
                    item['snippet']['resourceId']['videoId']
                    for item in playlist_response.get('items', [])
                ]
                
                if video_ids:
                    video_details = self._fetch_video_details(video_ids)
                    videos.extend(video_details)
                
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break
                
            except HttpError as e:
                self.logger.error(f"Error fetching playlist items: {e}")
                break
        
        self.logger.info(f"Fetched {len(videos)} total videos")
        return videos
    
    def _fetch_video_details(self, video_ids: List[str]) -> List[Dict]:
        """
        Fetch detailed information for videos.
        
        Args:
            video_ids: List of video IDs
        
        Returns:
            List of video detail dictionaries
        """
        try:
            response = self._retry_api_call(
                self.youtube.videos().list,
                part='snippet,contentDetails,statistics',
                id=','.join(video_ids)
            ).execute()
            
            videos = []
            for item in response.get('items', []):
                video_data = self._parse_video_item(item)
                if video_data:
                    videos.append(video_data)
            
            return videos
            
        except HttpError as e:
            self.logger.error(f"Error fetching video details: {e}")
            return []
    
    def _parse_video_item(self, item: Dict) -> Optional[Dict]:
        """
        Parse video item from API response.
        
        Args:
            item: Video item from API
        
        Returns:
            Parsed video dictionary or None if not a Short
        """
        duration = item['contentDetails']['duration']
        duration_seconds = self._parse_duration(duration)
        
        # Filter by duration (Shorts are typically <= 60 seconds)
        if duration_seconds > self.max_duration_seconds:
            return None
        
        video_data = {
            'video_id': item['id'],
            'title': item['snippet']['title'],
            'description': item['snippet'].get('description', ''),
            'publish_date': item['snippet']['publishedAt'],
            'duration_seconds': duration_seconds,
            'view_count': int(item['statistics'].get('viewCount', 0)),
            'like_count': int(item['statistics'].get('likeCount', 0)),
            'comment_count': int(item['statistics'].get('commentCount', 0)),
        }
        
        return video_data
    
    @staticmethod
    def _parse_duration(duration: str) -> int:
        """
        Parse ISO 8601 duration string to seconds.
        
        Args:
            duration: Duration string (e.g., 'PT1M30S')
        
        Returns:
            Duration in seconds
        """
        import re
        
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration)
        
        if not match:
            return 0
        
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds
    
    def save_to_database(self, videos: List[Dict]) -> int:
        """
        Save video metadata to database.
        
        Args:
            videos: List of video metadata dictionaries
        
        Returns:
            Number of videos saved
        """
        self.logger.info(f"Saving {len(videos)} videos to database")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        current_time = datetime.now().isoformat()
        
        for video in videos:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO shorts_metadata
                    (video_id, title, description, publish_date, duration_seconds,
                     view_count, like_count, comment_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 
                            COALESCE((SELECT created_at FROM shorts_metadata WHERE video_id = ?), ?),
                            ?)
                """, (
                    video['video_id'],
                    video['title'],
                    video['description'],
                    video['publish_date'],
                    video['duration_seconds'],
                    video['view_count'],
                    video['like_count'],
                    video['comment_count'],
                    video['video_id'],
                    current_time,
                    current_time,
                ))
                
                # Initialize download status if not exists
                cursor.execute("""
                    INSERT OR IGNORE INTO download_status (video_id, status)
                    VALUES (?, 'pending')
                """, (video['video_id'],))
                
                saved_count += 1
                
            except sqlite3.Error as e:
                self.logger.error(f"Error saving video {video['video_id']}: {e}")
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Successfully saved {saved_count} videos")
        return saved_count
    
    def scrape_and_save(self) -> int:
        """
        Execute full scraping pipeline: fetch videos and save to database.
        
        Returns:
            Number of videos saved
        """
        self.logger.info("Starting YouTube Shorts scraping...")
        
        videos = self.fetch_channel_videos()
        
        # Filter for Shorts (already done in _parse_video_item, but double check)
        shorts = [v for v in videos if v['duration_seconds'] <= self.max_duration_seconds]
        
        self.logger.info(f"Found {len(shorts)} Shorts out of {len(videos)} videos")
        
        if shorts:
            saved_count = self.save_to_database(shorts)
            self.logger.info("Scraping completed successfully")
            return saved_count
        else:
            self.logger.warning("No Shorts found")
            return 0
