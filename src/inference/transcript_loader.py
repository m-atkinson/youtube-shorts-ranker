"""Transcript loading utilities for inference (Phase 3 - 3.1).

Load long-form podcast transcripts from .docx and .txt files
for segment extraction and prediction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


from ..utils import setup_logger


class TranscriptLoader:
    """Load transcripts from various file formats for inference.
    
    Supports:
    - .txt (plain text)
    - .docx (Microsoft Word documents)
    """

    def __init__(self) -> None:
        self.logger = setup_logger(__name__)

    def load(self, file_path: Union[Path, str]) -> str:
        """Load transcript from file.
        
        Args:
            file_path: Path to transcript file
            
        Returns:
            Transcript text as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {file_path}")
        
        self.logger.info(f"Loading transcript from {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == ".txt":
            text = self._load_txt(file_path)
        elif suffix == ".docx":
            text = self._load_docx(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Supported formats: .txt, .docx"
            )
        
        self.logger.info(f"Loaded transcript: {len(text)} characters, ~{len(text.split())} words")
        
        return text

    def _load_txt(self, path: Path) -> str:
        """Load plain text file.
        
        Args:
            path: Path to .txt file
            
        Returns:
            File contents as string
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            self.logger.warning(f"UTF-8 decoding failed for {path}, trying latin-1")
            with open(path, "r", encoding="latin-1") as f:
                text = f.read()
        
        return text.strip()

    def _load_docx(self, path: Path) -> str:
        """Load Microsoft Word document.
        
        Args:
            path: Path to .docx file
            
        Returns:
            Document text as string (paragraphs joined by newlines)
        """
        if not HAS_DOCX:
            raise ImportError(
                "python-docx is required to load .docx files. "
                "Install it with: pip install python-docx"
            )
        
        doc = Document(path)

        
        # Extract text from all paragraphs
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        
        # Join with newlines to preserve structure
        text = "\n".join(paragraphs)
        
        return text.strip()

    def validate_transcript(self, text: str, *, min_words: int = 100) -> bool:
        """Validate that transcript meets minimum requirements.
        
        Args:
            text: Transcript text
            min_words: Minimum word count required
            
        Returns:
            True if valid, False otherwise
        """
        if not text or not text.strip():
            self.logger.warning("Transcript is empty")
            return False
        
        word_count = len(text.split())
        if word_count < min_words:
            self.logger.warning(
                f"Transcript too short: {word_count} words (minimum: {min_words})"
            )
            return False
        
        return True
