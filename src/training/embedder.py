"""Embedding generation utilities (Phase 2 - 2.1).

Implements a configurable transcript embedder with on-disk caching and
batch processing using sentence-transformers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import hashlib
import os

import numpy as np
from sentence_transformers import SentenceTransformer
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from ..utils import setup_logger


class TranscriptEmbedder:
    """Generate and cache text embeddings for transcripts.

    Features:
    - Configurable model (e.g. 'all-MiniLM-L6-v2', 'all-mpnet-base-v2')
    - Batch encoding with optional progress bar
    - On-disk caching keyed by (video_id, text_hash, model)
    - Helper to expose embedding dimension
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        device: Optional[str] = None,
        cache_dir: Path | str = Path("data/processed/embeddings"),
    ) -> None:
        self.logger = setup_logger(__name__)
        self.model_name = model_name
        self.is_gemini = model_name.startswith("models/")
        
        if self.is_gemini:
            if not HAS_GEMINI:
                raise ImportError("google-generativeai is required for Gemini models. Install it with pip install google-generativeai")
            
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini embeddings")
            
            genai.configure(api_key=api_key)
            self.model = None # No local model
            self.logger.info(f"Initialized Gemini Embedder with model='{model_name}'")
        else:
            self.model = SentenceTransformer(model_name)
            if device:
                # Move model to requested device ('cpu' or 'cuda')
                try:
                    self.model = self.model.to(device)
                except Exception:
                    self.logger.warning(
                        f"Failed to move model to device '{device}', continuing on default device"
                    )
        
        # Prepare cache directory per model
        self.cache_dir = Path(cache_dir) / self._sanitize_model_name(model_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            f"TranscriptEmbedder initialized with model='{model_name}', cache='{self.cache_dir}'"
        )

    def _sanitize_model_name(self, name: str) -> str:
        return name.replace("/", "_")

    def _hash_text(self, text: str) -> str:
        # Short stable hash to invalidate cache when transcript changes
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

    def _cache_path(self, video_id: str, text_hash: str) -> Path:
        return self.cache_dir / f"{video_id}_{text_hash}.npy"

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        video_ids: Optional[Sequence[str]] = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        use_cache: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        """Embed a batch of texts with optional per-item caching.

        Args:
            texts: Iterable of transcript texts to embed
            video_ids: Optional list of IDs matching texts for cache keys
            batch_size: Encode batch size
            show_progress_bar: Whether to show a progress bar during encoding
            use_cache: If True and video_ids provided, load/save per-item cache
            normalize_embeddings: If True, L2-normalize embeddings

        Returns:
            Numpy array of shape (N, D)
        """
        if len(texts) == 0:
            return np.zeros((0, self.get_embedding_dim()), dtype=np.float32)

        if not use_cache or video_ids is None:
            # Single call, no per-item cache
            self.logger.info(
                f"Encoding {len(texts)} texts (no per-item cache); batch_size={batch_size}"
            )
            
            if self.is_gemini:
                return self._embed_gemini(texts, batch_size)
            
            embeddings = self.model.encode(
                list(texts),
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
            )
            return embeddings.astype(np.float32)

        if len(video_ids) != len(texts):
            raise ValueError("video_ids length must match texts length when provided")

        # Attempt per-item cache hits, collect misses
        cached: List[Optional[np.ndarray]] = [None] * len(texts)
        miss_indices: List[int] = []
        miss_texts: List[str] = []
        miss_cache_paths: List[Path] = []

        for i, (vid, text) in enumerate(zip(video_ids, texts)):
            th = self._hash_text(text)
            cpath = self._cache_path(str(vid), th)
            if cpath.exists():
                try:
                    cached[i] = np.load(cpath)
                    continue
                except Exception:
                    # Treat as cache miss if load fails
                    pass
            miss_indices.append(i)
            miss_texts.append(text)
            miss_cache_paths.append(cpath)

        # Encode misses in a single batched call
        if miss_indices:
            self.logger.info(
                f"Cache hits: {len(texts) - len(miss_indices)}, misses: {len(miss_indices)}; "
                f"encoding with batch_size={batch_size}"
            )
            
            if self.is_gemini:
                miss_embeddings = self._embed_gemini(miss_texts, batch_size)
            else:
                miss_embeddings = self.model.encode(
                    miss_texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize_embeddings,
                ).astype(np.float32)
            # Save to cache
            for emb, cpath in zip(miss_embeddings, miss_cache_paths):
                try:
                    np.save(cpath, emb)
                except Exception:
                    # Best-effort caching; ignore failures
                    pass
            # Merge back
            for idx, emb in zip(miss_indices, miss_embeddings):
                cached[idx] = emb
        else:
            self.logger.info(f"All {len(texts)} items loaded from cache")

        # Stack in original order
        return np.stack([emb for emb in cached if emb is not None], axis=0)

    def _embed_gemini(self, texts: Sequence[str], batch_size: int = 100) -> np.ndarray:
        """Embed texts using Gemini API."""
        embeddings = []
        # Gemini batch size limit is typically 100 or less depending on payload
        # We'll use the provided batch_size but cap it if needed
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                # genai.embed_content supports batching
                result = genai.embed_content(
                    model=self.model_name,
                    content=batch,
                    task_type="retrieval_document", # Good default for storage
                )
                # result['embedding'] is a list of lists
                batch_embs = result['embedding']
                embeddings.extend(batch_embs)
            except Exception as e:
                self.logger.error(f"Gemini API error: {e}")
                raise
        
        return np.array(embeddings, dtype=np.float32)

    def get_embedding_dim(self) -> int:
        if self.is_gemini:
            # text-embedding-004 is 768 dimensions
            return 768
        return int(self.model.get_sentence_embedding_dimension())
