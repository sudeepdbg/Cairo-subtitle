"""
core/search_engine.py
Hybrid semantic search combining:
  - TF-IDF vector similarity (dense retrieval)
  - BM25 keyword matching (sparse retrieval)
  - Score fusion with configurable weights
  - MMR diversification on final results
  - Query expansion via synonym/related term injection
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional
import numpy as np

from core.embeddings import (
    TFIDFVectorizer, BM25, cosine_matrix, mmr_diversify, tokenize
)
from core.scene_detector import Scene


# ── Query expansion vocabulary ────────────────────────────────────────────────
_SYNONYMS: dict[str, list[str]] = {
    "car": ["vehicle", "automobile", "drive", "auto", "motor"],
    "movie": ["film", "cinema", "show", "series", "episode"],
    "food": ["meal", "eat", "cook", "recipe", "cuisine", "dish"],
    "happy": ["joy", "laugh", "smile", "excited", "cheerful"],
    "sad": ["cry", "unhappy", "grief", "sorrow", "depressed"],
    "fight": ["battle", "conflict", "struggle", "combat", "clash"],
    "love": ["romance", "affection", "relationship", "heart"],
    "money": ["finance", "wealth", "income", "profit", "revenue"],
    "technology": ["tech", "digital", "software", "computer", "ai"],
    "travel": ["journey", "trip", "adventure", "destination", "explore"],
    "health": ["wellness", "fitness", "medical", "wellbeing", "exercise"],
    "music": ["song", "melody", "beat", "artist", "concert"],
    "work": ["job", "career", "professional", "business", "office"],
    "nature": ["wildlife", "environment", "outdoor", "forest", "ocean"],
    "family": ["parent", "child", "home", "relationship", "together"],
}


def expand_query(query: str) -> str:
    """Add related terms to improve recall."""
    tokens = tokenize(query)
    expansions = []
    for t in tokens:
        if t in _SYNONYMS:
            expansions.extend(_SYNONYMS[t][:2])
    if expansions:
        return query + " " + " ".join(expansions)
    return query


@dataclass
class SearchResult:
    scene: Scene
    score: float
    vector_score: float
    bm25_score: float
    rank: int

    def to_dict(self) -> dict:
        d = self.scene.to_dict()
        d["search_score"] = round(self.score, 4)
        d["vector_score"] = round(self.vector_score, 4)
        d["bm25_score"] = round(self.bm25_score, 4)
        d["rank"] = self.rank
        return d


class HybridSearchEngine:
    """
    Two-stage retrieval:
      Stage 1 — Recall: BM25 + TF-IDF vector similarity, fused by RRF
      Stage 2 — Diversify: MMR on top-N results
    """

    def __init__(
        self,
        vector_weight: float = 0.55,
        bm25_weight: float = 0.45,
        mmr_lambda: float = 0.70,
    ):
        self.vector_weight = vector_weight
        self.bm25_weight   = bm25_weight
        self.mmr_lambda    = mmr_lambda

        self.scenes: list[Scene] = []
        self._vectorizer: Optional[TFIDFVectorizer] = None
        self._bm25: Optional[BM25] = None
        self._corpus_vecs: Optional[np.ndarray] = None
        self._texts: list[str] = []

    def index(self, scenes: list[Scene]) -> None:
        """Build vector + BM25 indexes from a list of scenes."""
        self.scenes = scenes
        self._texts = [s.text for s in scenes]

        if not self._texts:
            return

        self._vectorizer = TFIDFVectorizer(max_features=2000)
        self._corpus_vecs = self._vectorizer.fit_transform(self._texts)

        self._bm25 = BM25()
        self._bm25.fit(self._texts)

        # Always overwrite scene embeddings so they share the same vocabulary
        # space as the ad engine after sync_vectorizer() is called.
        for i, scene in enumerate(scenes):
            scene.embedding = self._corpus_vecs[i]

    def add_scenes(self, new_scenes: list[Scene]) -> None:
        """Incrementally add new scenes and rebuild index."""
        all_scenes = self.scenes + new_scenes
        self.index(all_scenes)

    def search(
        self,
        query: str,
        top_k: int = 10,
        diversify: bool = True,
        video_id: Optional[str] = None,
        min_safety: float = 0.0,
        iab_filter: Optional[list[str]] = None,
        expand: bool = True,
    ) -> list[SearchResult]:
        """
        Full hybrid search with optional filters.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            diversify: Apply MMR diversification
            video_id: Restrict to a specific video
            min_safety: Minimum brand safety score
            iab_filter: Restrict to specific IAB category IDs
            expand: Apply query expansion
        """
        if not self.scenes or self._vectorizer is None:
            return []

        # Filter scenes
        candidates = self.scenes
        candidate_indices = list(range(len(self.scenes)))

        if video_id:
            filtered = [(i, s) for i, s in enumerate(self.scenes)
                        if s.video_id == video_id]
            candidate_indices, candidates = zip(*filtered) if filtered else ([], [])
            candidates = list(candidates)
            candidate_indices = list(candidate_indices)

        if not candidates:
            return []

        # Query expansion
        effective_query = expand_query(query) if expand else query

        # Vector scores
        q_vec = self._vectorizer.embed(effective_query)
        corpus_vecs = self._corpus_vecs[candidate_indices]
        vec_scores_raw = cosine_matrix(q_vec.reshape(1, -1), corpus_vecs)[0]
        vec_scores = (vec_scores_raw - vec_scores_raw.min()) / (
            vec_scores_raw.max() - vec_scores_raw.min() + 1e-10
        )

        # BM25 scores
        bm25_raw = self._bm25.score_all(effective_query)[candidate_indices]
        bm25_max = bm25_raw.max()
        bm25_scores = bm25_raw / (bm25_max + 1e-10)

        # Fused scores
        fused = (
            self.vector_weight * vec_scores
            + self.bm25_weight * bm25_scores
        )

        # Build ranked results
        ranked_indices = np.argsort(fused)[::-1]
        results: list[SearchResult] = []

        for local_idx in ranked_indices:
            scene = candidates[local_idx]

            # Safety filter
            if scene.brand_safety.get("safety_score", 1.0) < min_safety:
                continue

            # IAB filter
            if iab_filter:
                scene_cats = {c["id"] for c in scene.iab_categories}
                if not scene_cats & set(iab_filter):
                    continue

            results.append(SearchResult(
                scene=scene,
                score=float(fused[local_idx]),
                vector_score=float(vec_scores[local_idx]),
                bm25_score=float(bm25_scores[local_idx]),
                rank=len(results) + 1,
            ))

        # Diversify with MMR
        if diversify and len(results) > top_k:
            candidate_vecs = np.array([r.scene.embedding for r in results
                                       if r.scene.embedding is not None])
            ids = list(range(len(results)))
            selected_ids = mmr_diversify(q_vec, candidate_vecs, ids,
                                          top_k=top_k, lambda_param=self.mmr_lambda)
            results = [results[i] for i in selected_ids]
        else:
            results = results[:top_k]

        # Re-rank after MMR
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    def find_similar_scenes(
        self,
        scene: Scene,
        top_k: int = 5,
        exclude_same_video: bool = False,
    ) -> list[SearchResult]:
        """Find scenes semantically similar to a given scene."""
        if scene.embedding is None or self._corpus_vecs is None:
            return []

        sims = cosine_matrix(scene.embedding.reshape(1, -1), self._corpus_vecs)[0]

        results = []
        for i, (s, sim) in enumerate(zip(self.scenes, sims)):
            if s.scene_id == scene.scene_id:
                continue
            if exclude_same_video and s.video_id == scene.video_id:
                continue
            results.append(SearchResult(
                scene=s, score=float(sim),
                vector_score=float(sim), bm25_score=0.0, rank=0,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_k]
        for i, r in enumerate(results):
            r.rank = i + 1
        return results

    @property
    def stats(self) -> dict:
        videos = set(s.video_id for s in self.scenes)
        return {
            "total_scenes": len(self.scenes),
            "total_videos": len(videos),
            "vocab_size": len(self._vectorizer.vocab) if self._vectorizer else 0,
        }

    @property
    def vectorizer(self) -> Optional[TFIDFVectorizer]:
        """Expose fitted vectorizer so AdMatchingEngine can share vocab space."""
        return self._vectorizer
