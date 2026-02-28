"""
Vector Semantic Layer for Travel Recommendation
===============================================
Semantic module based on sentence-BERT to enrich hybrid ranking.

Theoretical references:
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks. EMNLP-IJCNLP.
- Burke, R. (2002). Hybrid Recommender Systems: Survey and Experiments.

Idea: represent each activity as a dense embedding (384 dim) and use
cosine similarity as an additional semantic signal in the final score.
"""

import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None

log = logging.getLogger(__name__)


class VectorSemanticRecommender:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. Run: pip install -r requirements.txt"
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.activities_df = None
        self.activity_ids: List[str] = []
        self.embeddings = None

    def fit(self, activities_df: pd.DataFrame) -> "VectorSemanticRecommender":
        df = activities_df.reset_index(drop=True).copy()
        self.activities_df = df
        self.activity_ids = df["activity_id"].astype(str).tolist()
        texts = self._build_activity_texts(df)
        self.embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        log.info("Vector fit completed: %s activities, dim=%s", len(df), self.embeddings.shape[1])
        return self

    def build_query_embedding(self, user_preferences: Dict[str, float]) -> np.ndarray:
        query_text = self._build_user_query_text(user_preferences)
        vector = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return vector.astype(float)

    def score_for_user(self, user_preferences: Dict[str, float]) -> np.ndarray:
        if self.embeddings is None:
            raise RuntimeError("Call fit() before score_for_user()")
        q = self.build_query_embedding(user_preferences)
        return self.score_for_query_embedding(q)

    def score_for_query_embedding(self, query_embedding: np.ndarray) -> np.ndarray:
        if self.embeddings is None:
            raise RuntimeError("Call fit() before score_for_query_embedding()")
        scores = np.dot(self.embeddings, query_embedding)
        scores = np.clip(scores, 0.0, 1.0)
        return scores.astype(float)

    def get_embeddings_df(self) -> pd.DataFrame:
        if self.embeddings is None:
            raise RuntimeError("Call fit() before get_embeddings_df()")
        return pd.DataFrame(
            {
                "activity_id": self.activity_ids,
                "embedding": [emb.astype(float).tolist() for emb in self.embeddings],
            }
        )

    def _build_activity_texts(self, df: pd.DataFrame) -> List[str]:
        tags_series = df.get("tags", pd.Series([""] * len(df), index=df.index)).fillna("")
        tags_text = tags_series.apply(self._parse_tags)
        parts = (
            df.get("name", "").fillna("").astype(str)
            + " "
            + df.get("category", "").fillna("").astype(str)
            + " "
            + df.get("experience_type", "").fillna("").astype(str)
            + " "
            + tags_text.astype(str)
            + " "
            + df.get("description", "").fillna("").astype(str)
        )
        return parts.str.strip().tolist()

    @staticmethod
    def _build_user_query_text(user_preferences: Dict[str, float]) -> str:
        tokens = []
        for experience_type, value in user_preferences.items():
            weight = max(1, int(round(float(value) * 3)))
            tokens.extend([experience_type] * weight)
        if not tokens:
            return "cultura natura cibo svago"
        return " ".join(tokens)

    @staticmethod
    def _parse_tags(tags_str: str) -> str:
        try:
            parsed = json.loads(tags_str)
            if isinstance(parsed, list):
                return " ".join(str(x) for x in parsed)
        except Exception:
            return ""
        return ""
