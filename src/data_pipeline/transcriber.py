"""Transcription pipeline using OpenAI Whisper."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import whisper

from ..utils.logger import setup_logger


class Transcriber:
    """Generate transcripts from video files using Whisper."""
    
    def __init__(
        self,
        db_path: Path,
        output_dir: Path,
        model_size: str = "base",
        language: str = "en",
        device: Optional[str] = None,
    ):
        """
        Initialize transcriber.
        
        Args:
            db_path: Path to SQLite database
            output_dir: Directory to save transcripts
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Transcription language code
            device: Device to run model on (None for auto-detect)
        """
        self.db_path = db_path
        self.output_dir = output_dir
        self.model_size = model_size
        self.language = language
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(__name__)
        
        # Load Whisper model
        self.logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size, device=device)
        self.logger.info("Whisper model loaded successfully")
    
    def get_pending_transcriptions(self) -> List[Dict[str, str]]:
        """
        Get list of videos that need transcription.
        
        Returns:
            List of dictionaries with video_id and video_path
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ds.video_id, ds.video_path
            FROM download_status ds
            WHERE ds.video_downloaded = 1 
              AND ds.video_path IS NOT NULL
              AND ds.transcript_generated = 0
            ORDER BY ds.last_attempt_at IS NULL DESC
        """)
        
        videos = [
            {'video_id': row[0], 'video_path': row[1]}
            for row in cursor.fetchall()
        ]
        
        conn.close()
        return videos
    
    def transcribe_video(self, video_id: str, video_path: str) -> bool:
        """
        Transcribe a single video.
        
        Args:
            video_id: YouTube video ID
            video_path: Path to video file
        
        Returns:
            True if successful, False otherwise
        """
        output_path = self.output_dir / f"{video_id}.json"
        
        # Skip if transcript already exists
        if output_path.exists():
            self.logger.info(f"Transcript for {video_id} already exists")
            self._load_existing_transcript(video_id, output_path)
            return True
        
        video_file = Path(video_path)
        if not video_file.exists():
            self.logger.error(f"Video file not found: {video_path}")
            self._update_transcription_status(video_id, False, None, "Video file not found")
            return False
        
        try:
            self.logger.info(f"Transcribing video: {video_id}")
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                str(video_file),
                language=self.language,
                verbose=False,
            )
            
            # Extract transcript text and metadata
            transcript_data = {
                'video_id': video_id,
                'text': result['text'].strip(),
                'language': result.get('language', self.language),
                'segments': result.get('segments', []),
                'word_count': len(result['text'].split()),
            }
            
            # Save transcript to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            
            # Save to database
            self._save_transcript_to_db(transcript_data)
            
            # Update status
            self._update_transcription_status(
                video_id,
                True,
                str(output_path),
                None
            )
            
            self.logger.info(
                f"Successfully transcribed {video_id}: "
                f"{transcript_data['word_count']} words"
            )
            return True
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error transcribing {video_id}: {error_msg}")
            self._update_transcription_status(video_id, False, None, error_msg)
            return False
    
    def _load_existing_transcript(self, video_id: str, transcript_path: Path) -> None:
        """
        Load existing transcript and ensure it's in database.
        
        Args:
            video_id: YouTube video ID
            transcript_path: Path to transcript JSON file
        """
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            self._save_transcript_to_db(transcript_data)
            self._update_transcription_status(
                video_id,
                True,
                str(transcript_path),
                None
            )
        except Exception as e:
            self.logger.error(f"Error loading existing transcript {video_id}: {e}")
    
    def _save_transcript_to_db(self, transcript_data: Dict) -> None:
        """
        Save transcript data to database.
        
        Args:
            transcript_data: Dictionary with transcript information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = datetime.now().isoformat()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO transcripts
                (video_id, transcript_text, language, word_count, created_at)
                VALUES (?, ?, ?, ?, 
                        COALESCE((SELECT created_at FROM transcripts WHERE video_id = ?), ?))
            """, (
                transcript_data['video_id'],
                transcript_data['text'],
                transcript_data['language'],
                transcript_data['word_count'],
                transcript_data['video_id'],
                current_time,
            ))
            
            conn.commit()
            
        except sqlite3.Error as e:
            self.logger.error(f"Error saving transcript to database: {e}")
        finally:
            conn.close()
    
    def _update_transcription_status(
        self,
        video_id: str,
        success: bool,
        transcript_path: Optional[str],
        error_message: Optional[str]
    ) -> None:
        """
        Update transcription status in database.
        
        Args:
            video_id: YouTube video ID
            success: Whether transcription was successful
            transcript_path: Path to transcript file (if successful)
            error_message: Error message (if failed)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = datetime.now().isoformat()
        
        cursor.execute("""
            UPDATE download_status
            SET transcript_generated = ?,
                transcript_path = ?,
                error_message = CASE 
                    WHEN ? IS NOT NULL THEN error_message || '; ' || ?
                    ELSE error_message
                END,
                last_attempt_at = ?
            WHERE video_id = ?
        """, (
            1 if success else 0,
            transcript_path,
            error_message,
            error_message,
            current_time,
            video_id,
        ))
        
        conn.commit()
        conn.close()
    
    def transcribe_all_pending(self, max_videos: Optional[int] = None) -> Dict[str, int]:
        """
        Transcribe all pending videos.
        
        Args:
            max_videos: Maximum number of videos to transcribe (None for all)
        
        Returns:
            Dictionary with transcription statistics
        """
        pending_videos = self.get_pending_transcriptions()
        
        if max_videos:
            pending_videos = pending_videos[:max_videos]
        
        self.logger.info(f"Starting transcription of {len(pending_videos)} videos")
        
        stats = {
            'total': len(pending_videos),
            'successful': 0,
            'failed': 0,
        }
        
        for i, video_info in enumerate(pending_videos, 1):
            self.logger.info(
                f"Processing video {i}/{len(pending_videos)}: "
                f"{video_info['video_id']}"
            )
            
            success = self.transcribe_video(
                video_info['video_id'],
                video_info['video_path']
            )
            
            if success:
                stats['successful'] += 1
            else:
                stats['failed'] += 1
        
        self.logger.info(
            f"Transcription complete: {stats['successful']} successful, "
            f"{stats['failed']} failed out of {stats['total']} total"
        )
        
        return stats
    
    def get_all_transcripts(self) -> List[Dict]:
        """
        Get all transcripts from database.
        
        Returns:
            List of transcript dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT video_id, transcript_text, language, word_count
            FROM transcripts
            ORDER BY video_id
        """)
        
        transcripts = [
            {
                'video_id': row[0],
                'transcript_text': row[1],
                'language': row[2],
                'word_count': row[3],
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        return transcripts
