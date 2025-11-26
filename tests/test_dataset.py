import sqlite3
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pytest

# Ensure project root (so 'src' is a package) is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training import dataset as dataset_module  # noqa: E402

class FakeEmbedder:
    """Minimal stub to replace TranscriptEmbedder in dataset tests."""

    def __init__(self, model_name: str = "dummy", cache_dir: Union[Path, str] = Path(".")) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)

    def embed_batch(
        self,
        texts,
        *,
        video_ids=None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        use_cache: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        # Deterministic small embeddings based on text length and index
        embs = []
        for i, t in enumerate(texts):
            L = len(str(t))
            embs.append([L, i, float(L % 5)])
        return np.asarray(embs, dtype=np.float32)


def create_minimal_db(db_path: Path, *, n_valid: int = 20) -> list[int]:
    """Create a minimal SQLite DB with required tables and data.

    Returns the list of valid view_counts kept after cleaning.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        # Minimal schemas containing only columns used by the query
        cur.execute(
            """
            CREATE TABLE shorts_metadata (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                publish_date TEXT,
                duration_seconds INTEGER,
                view_count INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE transcripts (
                video_id TEXT PRIMARY KEY,
                transcript_text TEXT
            )
            """
        )
        # Insert valid rows
        valid_views = []
        for i in range(n_valid):
            vid = f"vid_{i}"
            title = f"Title {i}"
            desc = f"Desc {i}"
            pub = "2024-01-01"
            dur = 30 + i
            views = i  # increasing views
            tt = f"Transcript text {i}"
            cur.execute(
                "INSERT INTO shorts_metadata (video_id, title, description, publish_date, duration_seconds, view_count) VALUES (?, ?, ?, ?, ?, ?)",
                (vid, title, desc, pub, dur, views),
            )
            cur.execute(
                "INSERT INTO transcripts (video_id, transcript_text) VALUES (?, ?)",
                (vid, tt),
            )
            valid_views.append(views)
        # Insert rows that should be dropped by cleaning
        # 1) Missing transcript_text (filtered by WHERE and dropna)
        cur.execute(
            "INSERT INTO shorts_metadata (video_id, title, description, publish_date, duration_seconds, view_count) VALUES (?, ?, ?, ?, ?, ?)",
            ("drop_null_t", "Title X", "Desc X", "2024-01-01", 42, 10),
        )
        cur.execute(
            "INSERT INTO transcripts (video_id, transcript_text) VALUES (?, ?)",
            ("drop_null_t", None),
        )
        # 2) Empty transcript_text (filtered by strip length > 0)
        cur.execute(
            "INSERT INTO shorts_metadata (video_id, title, description, publish_date, duration_seconds, view_count) VALUES (?, ?, ?, ?, ?, ?)",
            ("drop_empty_t", "Title Y", "Desc Y", "2024-01-01", 43, 11),
        )
        cur.execute(
            "INSERT INTO transcripts (video_id, transcript_text) VALUES (?, ?)",
            ("drop_empty_t", "   "),
        )
        # 3) Null view_count (dropna)
        cur.execute(
            "INSERT INTO shorts_metadata (video_id, title, description, publish_date, duration_seconds, view_count) VALUES (?, ?, ?, ?, ?, ?)",
            ("drop_null_v", "Title Z", "Desc Z", "2024-01-01", 44, None),
        )
        cur.execute(
            "INSERT INTO transcripts (video_id, transcript_text) VALUES (?, ?)",
            ("drop_null_v", "OK"),
        )
        # 4) Negative view_count (filter >= 0)
        cur.execute(
            "INSERT INTO shorts_metadata (video_id, title, description, publish_date, duration_seconds, view_count) VALUES (?, ?, ?, ?, ?, ?)",
            ("drop_neg_v", "Title N", "Desc N", "2024-01-01", 45, -5),
        )
        cur.execute(
            "INSERT INTO transcripts (video_id, transcript_text) VALUES (?, ?)",
            ("drop_neg_v", "OK"),
        )
        conn.commit()
    return valid_views


def test_split_ratio_validation_raises_value_error(tmp_path: Path):
    db_path = tmp_path / "db.sqlite"
    # Create empty db file; validation occurs before reading DB
    sqlite3.connect(db_path).close()
    with pytest.raises(ValueError):
        dataset_module.ShortsDataset(
            db_path=db_path,
            train_split=0.6,
            val_split=0.3,
            test_split=0.2,
        )


def test_dataset_loads_sqlite_cleans_and_prepares_splits(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "db.sqlite"
    valid_views = create_minimal_db(db_path, n_valid=20)

    # Patch embedder to avoid model load
    monkeypatch.setattr(dataset_module, "TranscriptEmbedder", FakeEmbedder)

    ds = dataset_module.ShortsDataset(
        db_path=db_path,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
        random_seed=123,
        target_transform=None,
        n_view_bins=3,
    )

    X_train, y_train, X_val, y_val, X_test, y_test = ds.get_splits()

    # Expect exact counts for 20 rows -> 14/3/3
    assert X_train.shape[0] == 14
    assert X_val.shape[0] == 3
    assert X_test.shape[0] == 3

    # Embedding dimensions from FakeEmbedder
    assert X_train.shape[1] == 3
    assert X_val.shape[1] == 3
    assert X_test.shape[1] == 3

    # Targets are raw (no transform here)
    all_targets = np.concatenate([y_train, y_val, y_test])
    assert set(all_targets.astype(int)) == set(valid_views)


def test_log1p_transformation_applied(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "db.sqlite"
    valid_views = create_minimal_db(db_path, n_valid=12)

    monkeypatch.setattr(dataset_module, "TranscriptEmbedder", FakeEmbedder)

    ds = dataset_module.ShortsDataset(
        db_path=db_path,
        train_split=0.5,
        val_split=0.25,
        test_split=0.25,
        random_seed=0,
        target_transform="log1p",
        n_view_bins=3,
    )

    X_train, y_train, X_val, y_val, X_test, y_test = ds.get_splits()

    # Combine and compare ignoring order
    y_all = np.concatenate([y_train, y_val, y_test])
    expected = np.log1p(np.asarray(valid_views, dtype=np.float64)).astype(np.float32)

    assert np.allclose(np.sort(y_all), np.sort(expected), rtol=1e-6, atol=1e-6)
