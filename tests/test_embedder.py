import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root (so 'src' is a package) is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training import embedder as embedder_module  # noqa: E402


class FakeModel:
    def __init__(self, *args, **kwargs):
        self.encode_call_count = 0

    def to(self, device):
        # mimic sentence-transformers .to()
        return self

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False):
        self.encode_call_count += 1
        embs = []
        for i, t in enumerate(texts):
            s = str(t)
            L = len(s)
            embs.append([L, i, float(L % 7)])
        arr = np.asarray(embs, dtype=np.float32)
        return arr


def test_caching_generates_and_retrieves_from_disk(monkeypatch, tmp_path: Path):
    # Patch SentenceTransformer used inside TranscriptEmbedder
    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeModel)

    cache_dir = tmp_path / "cache"
    emb = embedder_module.TranscriptEmbedder(model_name="fake-model", cache_dir=cache_dir)

    texts = ["hello", "world", "hello"]
    video_ids = ["a", "b", "a"]

    # First call: should encode once and create cache files
    out1 = emb.embed_batch(texts, video_ids=video_ids, use_cache=True, show_progress_bar=False)
    assert out1.shape == (3, 3)
    # Ensure cache files exist
    # Cache path format: cache_dir/<model>/<video_id>_<hash>.npy
    model_cache = cache_dir / emb._sanitize_model_name(emb.model_name)
    assert model_cache.exists()
    # There are 3 items, but 2 unique (a+hello, b+world) -> at least those two should exist
    # We check both explicitly
    from hashlib import sha1
    h_hello = sha1("hello".encode("utf-8")).hexdigest()[:10]
    h_world = sha1("world".encode("utf-8")).hexdigest()[:10]
    assert (model_cache / f"a_{h_hello}.npy").exists()
    assert (model_cache / f"b_{h_world}.npy").exists()

    # Track encode calls
    model_obj = emb.model  # FakeModel instance
    assert model_obj.encode_call_count == 1

    # Second call: should load entirely from cache, not call encode again
    out2 = emb.embed_batch(texts, video_ids=video_ids, use_cache=True, show_progress_bar=False)
    assert out2.shape == (3, 3)
    # Because two items share the same cache key (a+hello), the cached vector reflects the
    # last write for that key, so both rows 0 and 2 should be identical.
    assert np.allclose(out2[0], out2[2])
    assert model_obj.encode_call_count == 1  # no additional encode


def test_mismatch_video_ids_raises_value_error(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeModel)
    emb = embedder_module.TranscriptEmbedder(model_name="fake-model", cache_dir=tmp_path / "cache2")

    with pytest.raises(ValueError):
        emb.embed_batch(["a", "b"], video_ids=["only_one"], use_cache=True)
