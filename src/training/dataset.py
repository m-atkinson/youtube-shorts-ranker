"""Dataset preparation for training (Phase 2 - 2.2).

Build a training dataset from the Phase 1 SQLite database by joining
shorts metadata with transcripts, generating embeddings, and creating
train/val/test splits stratified by view count bins.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split

from .embedder import TranscriptEmbedder
from ..utils import setup_logger


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_video_ids: list
    val_video_ids: list
    test_video_ids: list


class ShortsDataset:
    """Prepare dataset for training YouTube Shorts view prediction.

    Steps:
    - Load processed data from SQLite (shorts_metadata JOIN transcripts)
    - Generate/load embeddings via TranscriptEmbedder (with caching)
    - Create train/val/test splits (70/15/15 by default) stratified by view bins
    """

    def __init__(
        self,
        *,
        db_path: Path | str = Path("data/database/shorts_data.db"),
        embedding_model: str = "all-MiniLM-L6-v2",
        cache_dir: Path | str = Path("data/processed/embeddings"),
        train_split: float = 0.70,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42,
        n_view_bins: int = 5,
        target_transform: Optional[str] = "log1p",  # None | 'log1p'
    ) -> None:
        self.logger = setup_logger(__name__)

        # Validate split ratios
        total = train_split + val_split + test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train/val/test splits must sum to 1.0")

        self.db_path = Path(db_path)
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.n_view_bins = n_view_bins
        self.target_transform = target_transform

        # Load tabular data
        df = self._load_dataframe()
        if df.empty:
            raise RuntimeError("No training data found (ensure Phase 1 pipeline has run)")

        # Compute target
        y_raw = df["view_count"].to_numpy(dtype=np.float64)
        if target_transform == "log1p":
            y = np.log1p(y_raw).astype(np.float32)
        else:
            y = y_raw.astype(np.float32)

        # Create view bins for stratification (fall back gracefully for tiny datasets)
        bins = self._make_view_bins(df["view_count"].to_numpy())

        # Generate/load embeddings for all rows once
        embedder = TranscriptEmbedder(
            model_name=self.embedding_model, cache_dir=self.cache_dir
        )
        texts = df["transcript_text"].astype(str).tolist()
        video_ids = df["video_id"].astype(str).tolist()
        X_all = embedder.embed_batch(
            texts, video_ids=video_ids, batch_size=32, show_progress_bar=True, use_cache=True
        )

        # Two-stage split to preserve exact requested ratios
        # 1) Train vs Temp (Val+Test)
        test_val_size = self.val_split + self.test_split
        
        if test_val_size == 0:
            # Use all data for training
            X_train, y_train, ids_train = X_all, y, video_ids
            X_val = X_test = np.array([])
            y_val = y_test = np.array([])
            ids_val = ids_test = []
        else:
            X_train, X_temp, y_train, y_temp, bins_train, bins_temp, ids_train, ids_temp = train_test_split(
                X_all,
                y,
                bins,
                video_ids,
                train_size=self.train_split,
                random_state=self.random_seed,
                stratify=bins,
            )

            # 2) Split Temp into Val and Test (if needed)
            if self.val_split == 0:
                # No validation set, everything in temp goes to test
                X_val, y_val, ids_val = np.array([]), np.array([]), []
                X_test, y_test, ids_test = X_temp, y_temp, ids_temp
            elif self.test_split == 0:
                # No test set, everything in temp goes to val
                X_test, y_test, ids_test = np.array([]), np.array([]), []
                X_val, y_val, ids_val = X_temp, y_temp, ids_temp
            else:
                # Split temp into val and test
                val_ratio_rel = self.val_split / test_val_size
                X_val, X_test, y_val, y_test, _, _, ids_val, ids_test = train_test_split(
                    X_temp,
                    y_temp,
                    bins_temp,
                    ids_temp,
                    train_size=val_ratio_rel,
                    random_state=self.random_seed,
                    stratify=bins_temp,
                )

        self.splits = DatasetSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            train_video_ids=ids_train,
            val_video_ids=ids_val,
            test_video_ids=ids_test,
        )

    def _load_dataframe(self) -> pd.DataFrame:
        """Load joined metadata + transcripts from SQLite into a DataFrame."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")

        query = """
            SELECT
                sm.video_id,
                sm.title,
                sm.description,
                sm.publish_date,
                sm.duration_seconds,
                sm.view_count,
                t.transcript_text
            FROM shorts_metadata sm
            JOIN transcripts t ON sm.video_id = t.video_id
            WHERE t.transcript_text IS NOT NULL
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)

        # Basic cleaning: drop missing values we cannot use
        before = len(df)
        df = df.dropna(subset=["transcript_text", "view_count"]).reset_index(drop=True)
        after = len(df)
        if after < before:
            self.logger.info(f"Dropped {before - after} rows with missing fields")

        # Remove obvious empties
        df = df[df["transcript_text"].str.strip().str.len() > 0]
        df = df[df["view_count"] >= 0]
        df = df.reset_index(drop=True)
        self.logger.info(f"Loaded {len(df)} examples from database")
        return df

    def _make_view_bins(self, views: np.ndarray) -> np.ndarray:
        """Create stratification bins from view counts.

        Prefer quantile bins; if not enough unique values, fall back to uniform bins.
        """
        s = pd.Series(views)
        try:
            bins = pd.qcut(s, q=min(self.n_view_bins, len(s.unique())), labels=False, duplicates="drop")
        except Exception:
            # Fallback: equal-width bins
            bins = pd.cut(s, bins=min(self.n_view_bins, max(1, len(s.unique()))), labels=False)
        # If still NaNs (e.g., single unique value), put all in one bin
        bins = bins.fillna(0).astype(int)
        return bins.to_numpy()

    def get_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return train/val/test splits as numpy arrays."""
        sp = self.splits
        return sp.X_train, sp.y_train, sp.X_val, sp.y_val, sp.X_test, sp.y_test
    
    def get_split_video_ids(self) -> Tuple[list, list, list]:
        """Return video IDs for each split."""
        return self.splits.train_video_ids, self.splits.val_video_ids, self.splits.test_video_ids

    def augment(self) -> None:
        """Placeholder for optional text augmentation strategies."""
        return
