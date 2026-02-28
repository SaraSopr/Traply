import json
import unittest

import numpy as np
import pandas as pd

import src.recommender.hybrid_recommender as hybrid_mod


class _FakeVectorSemanticRecommender:
    def __init__(self, model_name=None):
        self.model_name = model_name

    def fit(self, activities_df):
        self.n = len(activities_df)
        return self

    def get_embeddings_df(self):
        return pd.DataFrame({"activity_id": [], "embedding": []})

    def score_for_user(self, user_preferences):
        return np.full(self.n, 0.5, dtype=float)

    def build_query_embedding(self, user_preferences):
        return np.zeros(384, dtype=float)


class TestHybridRecommender(unittest.TestCase):
    def setUp(self):
        self.original_vector_cls = hybrid_mod.VectorSemanticRecommender
        hybrid_mod.VectorSemanticRecommender = _FakeVectorSemanticRecommender

    def tearDown(self):
        hybrid_mod.VectorSemanticRecommender = self.original_vector_cls

    def test_fit_and_recommend(self):
        activities_df = pd.DataFrame(
            {
                "activity_id": ["A1", "A2", "A3", "A4"],
                "name": ["Museo", "Risto", "Parco", "Club"],
                "category": ["museum", "restaurant", "park", "bar"],
                "experience_type": ["culture", "food", "nature", "nightlife"],
                "rating": [4.5, 4.2, 4.4, 4.1],
                "review_count": [100, 120, 80, 90],
                "price_tier": [2, 2, 1, 3],
                "lat": [41.9, 41.91, 41.92, 41.93],
                "lng": [12.5, 12.51, 12.52, 12.53],
                "description": ["art", "food", "nature", "night"],
                "tags": ['["arte"]', '["food"]', '["green"]', '["night"]'],
            }
        )

        users_df = pd.DataFrame(
            {
                "user_id": ["U1", "U2"],
                "preferences_json": [
                    json.dumps({"culture": 0.9, "food": 0.7, "nature": 0.5, "shopping": 0.2, "nightlife": 0.1, "leisure": 0.4, "other": 0.2}),
                    json.dumps({"culture": 0.3, "food": 0.8, "nature": 0.6, "shopping": 0.4, "nightlife": 0.5, "leisure": 0.7, "other": 0.3}),
                ],
                "budget_max": [3, 3],
            }
        )

        ratings_df = pd.DataFrame(
            {
                "user_id": ["U1", "U1", "U2", "U2"],
                "activity_id": ["A1", "A2", "A3", "A4"],
                "rating": [4.0, 5.0, 4.5, 3.5],
            }
        )

        recommender = hybrid_mod.HybridRecommender(alpha=0.6, delta=0.2)
        recommender.fit(activities_df, users_df, ratings_df)

        result = recommender.recommend(
            user_id="U1",
            top_n=2,
            context={"budget_max": 3, "city": "Rome, Italy"},
            exclude_seen=False,
        )

        self.assertFalse(result.empty)
        self.assertIn("hybrid_score", result.columns)
        self.assertIn("vector_score", result.columns)
        self.assertLessEqual(len(result), 2)
