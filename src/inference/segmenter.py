"""Transcript segmentation logic (Phase 3 - 3.2).

Implements a dense sliding window strategy to generate candidate segments
of varying lengths.
"""
from __future__ import annotations

from typing import List, Dict, Any
import re

from ..utils import setup_logger


class SlidingWindowSegmenter:
    """Segments transcript into overlapping candidates of varying durations."""

    def __init__(
        self,
        min_duration: int = 30,
        max_duration: int = 90,
        step_size: int = 5,
        words_per_second: float = 2.5,
    ) -> None:
        """Initialize segmenter.

        Args:
            min_duration: Minimum segment duration in seconds.
            max_duration: Maximum segment duration in seconds.
            step_size: Number of words to step forward for next set of chunks.
            words_per_second: Estimated speaking rate to convert time to word counts.
        """
        self.logger = setup_logger(__name__)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.step_size = step_size
        self.words_per_second = words_per_second

        # Calculate word limits
        self.min_words = int(min_duration * words_per_second)
        self.max_words = int(max_duration * words_per_second)

        self.logger.info(
            f"Initialized SlidingWindowSegmenter: min_words={self.min_words}, "
            f"max_words={self.max_words}, step={self.step_size}"
        )

    def segment(self, transcript_text: str) -> List[Dict[str, Any]]:
        """Generate candidate segments from transcript text.

        Args:
            transcript_text: Full text of the transcript.

        Returns:
            List of segment dictionaries containing:
            - text: The segment text
            - start_word_idx: Index of the first word
            - end_word_idx: Index of the last word
            - estimated_duration: Duration in seconds
        """
        # Simple whitespace tokenization
        words = re.findall(r'\S+', transcript_text)
        total_words = len(words)
        segments = []

        self.logger.info(f"Segmenting transcript with {total_words} words...")

        # Sliding window
        # Outer loop: Step through the transcript
        for start_idx in range(0, total_words, self.step_size):
            # Inner loop: Generate varying lengths from min to max
            # We iterate by 1 second increments (approx 2-3 words)
            # To be precise to the user's request of "30s, 31s, 32s...", we iterate duration
            
            for duration in range(self.min_duration, self.max_duration + 1):
                word_count = int(duration * self.words_per_second)
                end_idx = start_idx + word_count
                
                if end_idx > total_words:
                    break
                
                segment_text = " ".join(words[start_idx:end_idx])
                
                segments.append({
                    "text": segment_text,
                    "start_word_idx": start_idx,
                    "end_word_idx": end_idx,
                    "estimated_duration": duration
                })

        self.logger.info(f"Generated {len(segments)} candidate segments.")
        return segments


class SentenceWindowSegmenter:
    """Segments transcript by sliding a window sentence-by-sentence."""

    def __init__(
        self,
        target_duration: int = 60,
        words_per_second: float = 2.5,
    ) -> None:
        """Initialize segmenter.

        Args:
            target_duration: Target segment duration in seconds.
            words_per_second: Estimated speaking rate.
        """
        self.logger = setup_logger(__name__)
        self.target_duration = target_duration
        self.words_per_second = words_per_second
        self.target_words = int(target_duration * words_per_second)

        self.logger.info(
            f"Initialized SentenceWindowSegmenter: target_duration={target_duration}s "
            f"(~{self.target_words} words)"
        )

    def segment(self, transcript_text: str) -> List[Dict[str, Any]]:
        """Generate candidate segments starting at each sentence.

        Args:
            transcript_text: Full text of the transcript.

        Returns:
            List of segment dictionaries.
        """
        # Split into sentences (keeping punctuation)
        # This regex splits after .!? followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', transcript_text)
        # Filter empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        
        total_sentences = len(sentences)
        segments = []

        self.logger.info(f"Segmenting transcript with {total_sentences} sentences...")

        for i in range(total_sentences):
            current_words = 0
            current_segment = []
            
            # Look ahead to build the segment
            for j in range(i, total_sentences):
                sentence = sentences[j]
                word_count = len(sentence.split())
                
                # Add sentence
                current_segment.append(sentence)
                current_words += word_count
                
                # Check if we've reached or exceeded the target
                if current_words >= self.target_words:
                    # We've reached the target. 
                    # Option: check if the previous sentence was closer to target?
                    # For now, just take this chunk.
                    break
            
            # Construct segment
            segment_text = " ".join(current_segment)
            duration = int(current_words / self.words_per_second)
            
            # Only add if it's reasonably close to target (e.g. at least 50% of target)
            # to avoid tiny tail segments
            if duration >= self.target_duration * 0.5:
                segments.append({
                    "text": segment_text,
                    "start_word_idx": -1, # Not easily tracked with sentence splitting, but less critical for this method
                    "end_word_idx": -1,
                    "estimated_duration": duration,
                    "sentence_start_idx": i,
                    "sentence_end_idx": i + len(current_segment)
                })

        self.logger.info(f"Generated {len(segments)} candidate segments.")
        return segments
