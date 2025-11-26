"""Segment prediction and ranking logic (Phase 3 - 3.3).

Implements the predictor that scores segments and selects the top non-overlapping
candidates.
"""
from __future__ import annotations

# Disable OpenMP to prevent XGBoost segfaults on macOS
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle

import numpy as np

from ..utils import setup_logger
from ..training.embedder import TranscriptEmbedder


class SegmentPredictor:
    """Scores and ranks transcript segments."""

    def __init__(
        self,
        model_path: str | Path,
        embedder: TranscriptEmbedder,
    ) -> None:
        """Initialize predictor.

        Args:
            model_path: Path to the trained model pickle file.
            embedder: Initialized TranscriptEmbedder instance.
        """
        self.logger = setup_logger(__name__)
        self.embedder = embedder
        self.model = self._load_model(model_path)

    def _load_model(self, path: str | Path) -> Any:
        """Load trained model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")
        
        self.logger.info(f"Loading model from {path}")
        
        # XGBoost models can segfault when loaded with pickle on macOS
        # Try multiple loading strategies
        
        # Strategy 1: Try joblib (safest for sklearn models)
        try:
            import joblib
            model = joblib.load(path)
            self.logger.info("Successfully loaded model with joblib")
            return model
        except Exception as e:
            self.logger.warning(f"joblib load failed: {e}")
        
        # Strategy 2: Try pickle with protocol 4
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            self.logger.info("Successfully loaded model with pickle")
            return model
        except Exception as e:
            self.logger.warning(f"pickle load failed: {e}")
            
        # Strategy 3: If it's an XGBoost model, the pickle might be corrupted
        # Let's try to extract the booster and wrap it
        self.logger.error("All loading strategies failed. Model may be corrupted or incompatible.")
        raise RuntimeError(f"Failed to load model from {path}")

    def predict_top_segments(
        self,
        segments: List[Dict[str, Any]],
        top_k: int = 5,
        overlap_threshold: float = 0.2,
        batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """Predict scores and return top non-overlapping segments.

        Args:
            segments: List of segment dicts (must contain 'text', 'start_word_idx', 'end_word_idx').
            top_k: Number of top segments to return.
            overlap_threshold: Max allowed intersection-over-union (or simple overlap) ratio.
            batch_size: Number of segments to process at once to avoid memory issues.

        Returns:
            List of top K segment dicts with 'score' and 'rank'.
        """
        if not segments:
            return []

        self.logger.info(f"Processing {len(segments)} segments in batches of {batch_size}...")
        
        # Process in batches to avoid memory issues
        all_scores = []
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            texts = [s["text"] for s in batch]
            
            self.logger.info(f"Embedding batch {i//batch_size + 1}/{(len(segments)-1)//batch_size + 1}...")
            embeddings = self.embedder.embed_batch(texts, show_progress_bar=False, use_cache=False)

            self.logger.info(f"Predicting scores for batch...")
            # Handle different model types (sklearn vs xgboost native)
            if hasattr(self.model, "predict"):
                scores = self.model.predict(embeddings)
            else:
                # Fallback if it's a raw booster or something else, though we expect sklearn-compatible
                scores = self.model.predict(embeddings)
            
            all_scores.extend(scores)

        
        # Add scores to segments
        for i, segment in enumerate(segments):
            segment["score"] = float(all_scores[i])
            segment["original_index"] = i

        # Sort by score descending
        sorted_segments = sorted(segments, key=lambda x: x["score"], reverse=True)

        # Non-Maximum Suppression (NMS)
        selected_segments = []
        
        self.logger.info(f"Performing NMS with threshold {overlap_threshold}...")
        
        for candidate in sorted_segments:
            if len(selected_segments) >= top_k:
                break
            
            is_overlapping = False
            for selected in selected_segments:
                if self._calculate_overlap(candidate, selected) > overlap_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                candidate["rank"] = len(selected_segments) + 1
                selected_segments.append(candidate)

        return selected_segments

    def _calculate_overlap(self, seg1: Dict[str, Any], seg2: Dict[str, Any]) -> float:
        """Calculate overlap ratio between two segments.
        
        Uses Intersection over Union (IoU).
        Handles both word-based indices and sentence-based indices.
        """
        # Determine which indices to use
        if seg1.get("start_word_idx", -1) != -1 and seg2.get("start_word_idx", -1) != -1:
            start1, end1 = seg1["start_word_idx"], seg1["end_word_idx"]
            start2, end2 = seg2["start_word_idx"], seg2["end_word_idx"]
        elif seg1.get("sentence_start_idx") is not None and seg2.get("sentence_start_idx") is not None:
            start1, end1 = seg1["sentence_start_idx"], seg1["sentence_end_idx"]
            start2, end2 = seg2["sentence_start_idx"], seg2["sentence_end_idx"]
        else:
            # Incompatible types or missing indices, assume no overlap to be safe
            # (or could assume full overlap to be conservative)
            return 0.0

        # Calculate intersection
        inter_start = max(start1, start2)
        inter_end = min(end1, end2)
        
        if inter_end <= inter_start:
            return 0.0

        intersection = inter_end - inter_start
        
        # Calculate union
        len1 = end1 - start1
        len2 = end2 - start2
        union = len1 + len2 - intersection
        
        if union == 0:
            return 0.0
            
        return intersection / union
