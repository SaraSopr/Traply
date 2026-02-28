"""
Hybrid Recommender System
==========================
Combines Content-Based Filtering and Collaborative Filtering
with an adaptive weighted ensemble.

Architettura:
    ┌─────────────────────┐    ┌──────────────────────────┐
    │  Content-Based (CB) │    │  Collaborative (CF)      │
    │  TF-IDF + cosine    │    │  SVD matrix factorization│
    │  similarity         │    │  (via Surprise library)  │
    └──────────┬──────────┘    └────────────┬─────────────┘
               │                            │
               └──────────┬─────────────────┘
                           ▼
                  ┌─────────────────┐
                  │ Weighted Merger │
                  │ α·CB + (1-α)·CF │
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │  Context Filter │  (budget, time, distance)
                  └────────┬────────┘
                           ▼
                    Top-N activities

Usage:
    rec = HybridRecommender()
    rec.fit(activities_df, users_df, ratings_df)
    results = rec.recommend(user_id="USR_00001", city="Barcellona", top_n=10)
"""

import json
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from typing import Optional
try:
    from src.recommender.time_aware import TimeAwareScorer, TemporalContext
except ModuleNotFoundError:
    from time_aware import TimeAwareScorer, TemporalContext

try:
    from src.recommender.vector_layer import VectorSemanticRecommender
except ModuleNotFoundError:
    from vector_layer import VectorSemanticRecommender

log = logging.getLogger(__name__)


class ContentBasedRecommender:
    """
    Recommends activities based on POI characteristics
    and user-declared preferences.

    Used features:
    - TF-IDF su: category + experience_type + tags + description
    - Normalized rating
    - Price tier
    """

    def __init__(self):
        self.tfidf     = TfidfVectorizer(max_features=500, stop_words="english")
        self.scaler    = MinMaxScaler()
        self.activity_matrix  = None   # shape: (n_activities, n_features)
        self.activities_df    = None

    def fit(self, activities_df: pd.DataFrame) -> "ContentBasedRecommender":
        """Builds the feature matrix for all activities."""
        df = activities_df.copy()
        self.activities_df = df.reset_index(drop=True)

        # ── Composite text for TF-IDF ───────────────────────────────────────
        df["text_features"] = (
            df["category"].fillna("") + " " +
            df["experience_type"].fillna("") + " " +
            df["tags"].fillna("").apply(self._parse_tags) + " " +
            df["description"].fillna("")
        )
        tfidf_matrix = self.tfidf.fit_transform(df["text_features"]).toarray()

        # ── Numeric features ─────────────────────────────────────────────────
        numeric = df[["rating", "price_tier", "review_count"]].fillna(0)
        numeric_scaled = self.scaler.fit_transform(numeric)

        # ── Final matrix ─────────────────────────────────────────────────────
        self.activity_matrix = np.hstack([tfidf_matrix, numeric_scaled])
        log.info(f"CB Fit: {len(df)} activities, {self.activity_matrix.shape[1]} features.")
        return self

    def score_for_user(self, user_preferences: dict) -> np.ndarray:
        """
        Calculates a CB score for each activity given a user profile.

        Args:
            user_preferences: dict {experience_type: float 0-1}

        Returns:
            Array of normalized scores [0,1] for each activity.
        """
        if self.activity_matrix is None:
            raise RuntimeError("Call fit() before score_for_user()")

        # Build user vector aligned with TF-IDF features
        experience_types = ["cultura", "natura", "cibo", "shopping", "vita_notturna", "svago", "altro"]
        user_text = " ".join([
            exp for exp in experience_types
            if user_preferences.get(exp, 0) > 0.5
        ])

        # Project user vector into TF-IDF space
        user_tfidf = self.tfidf.transform([user_text]).toarray()

        # Add zeros for numeric features
        n_numeric = 3
        user_vec = np.hstack([user_tfidf, np.zeros((1, n_numeric))])

        # Cosine similarity between user and each activity
        scores = cosine_similarity(user_vec, self.activity_matrix).flatten()

        # Normalize to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()

        return scores

    @staticmethod
    def _parse_tags(tags_str: str) -> str:
        """Converts JSON list of tags into text for TF-IDF."""
        try:
            tags = json.loads(tags_str)
            return " ".join(tags) if isinstance(tags, list) else ""
        except Exception:
            return ""


class CollaborativeFilteringRecommender:
    """
    Matrix Factorization with SVD (Singular Value Decomposition).
    Discovers latent patterns among similar users.

    Works well with at least 50+ users and 10+ ratings per user.
    For new users (cold start) → fallback to ContentBased.
    """

    def __init__(self, n_factors: int = 20):
        self.n_factors = n_factors
        self.user_factors     = None   # shape: (n_users, k)
        self.item_factors     = None   # shape: (n_items, k)
        self.global_mean      = 0.0
        self.user_index       = {}     # user_id → row index
        self.item_index       = {}     # activity_id → col index
        self.user_biases      = None
        self.item_biases      = None

    def fit(self, ratings_df: pd.DataFrame) -> "CollaborativeFilteringRecommender":
        """
        Factorizes the user-item matrix with SVD.

        Args:
            ratings_df: DataFrame with columns user_id, activity_id, rating
        """
        df = ratings_df.copy()

        # Build indices
        users     = df["user_id"].unique()
        items     = df["activity_id"].unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {a: i for i, a in enumerate(items)}

        n_users = len(users)
        n_items = len(items)

        # Sparse rating matrix
        rows = df["user_id"].map(self.user_index).values
        cols = df["activity_id"].map(self.item_index).values
        vals = df["rating"].values.astype(float)

        self.global_mean = vals.mean()

        # Center ratings (required for SVD)
        R = csr_matrix((vals - self.global_mean, (rows, cols)), shape=(n_users, n_items))

        # Truncated SVD
        k = min(self.n_factors, min(n_users, n_items) - 1)
        U, sigma, Vt = svds(R.toarray(), k=k)

        # Latent factors
        sigma_diag = np.diag(sigma)
        self.user_factors = U.dot(sigma_diag)   # (n_users, k)
        self.item_factors = Vt.T                # (n_items, k)

        log.info(f"CF Fit: {n_users} users, {n_items} activities, k={k}.")
        return self

    def score_for_user(self, user_id: str) -> Optional[dict]:
        """
        Predicts ratings for all known activities for a user.

        Returns:
            Dict {activity_id: predicted_score} or None if cold start.
        """
        if user_id not in self.user_index:
            return None  # Cold start — hybrid recommender will use only CB

        u_idx = self.user_index[user_id]
        u_vec = self.user_factors[u_idx]  # (k,)

        # Dot product with all items
        raw_scores = self.item_factors.dot(u_vec) + self.global_mean  # (n_items,)

        # Normalize to [0, 1]
        scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

        return {
            act_id: float(scores[i])
            for act_id, i in self.item_index.items()
        }


class HybridRecommender:
    """
    Weighted ensemble of Content-Based + Collaborative Filtering.

    α controls the balance:
    - α = 1.0  → Content-Based only  (new user, cold start)
    - α = 0.0  → Collaborative only  (user with many interactions)
    - α = 0.5  → balanced            (default)

    α is automatically adapted according to
    the user's amount of interactions.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        cold_start_threshold: int = 5,
        gamma: float = 0.25,
        delta: float = 0.15,
        vector_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.alpha = alpha
        self.cold_start_threshold = cold_start_threshold
        self.gamma = gamma
        self.delta = delta
        self.cb = ContentBasedRecommender()
        self.cf = CollaborativeFilteringRecommender()
        self.vector = VectorSemanticRecommender(model_name=vector_model_name)
        self.time_scorer    = TimeAwareScorer(gamma=gamma)
        self.activities_df  = None
        self.users_df       = None
        self.ratings_df     = None
        self._is_fitted     = False

    # ── Fit ──────────────────────────────────────────────────────────────────
    def fit(
        self,
        activities_df: pd.DataFrame,
        users_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        db=None,
    ) -> "HybridRecommender":
        """
        Trains both components.

        Args:
            activities_df:  POIs with features (Apify collector output)
            users_df:       Users with preferences_json
            ratings_df:     User-activity interactions
        """
        self.activities_df = activities_df.reset_index(drop=True)
        self.users_df      = users_df.set_index("user_id")
        self.ratings_df    = ratings_df

        log.info("Fitting Content-Based...")
        self.cb.fit(activities_df)

        log.info("Fitting Collaborative Filtering...")
        self.cf.fit(ratings_df)

        log.info("Fitting Vector Semantic Layer...")
        self.vector.fit(activities_df)
        if db is not None and hasattr(db, "save_activity_embeddings"):
            db.save_activity_embeddings(self.vector.get_embeddings_df())

        log.info("Fitting Time-Aware Scorer...")
        self.time_scorer.learn_from_interactions(ratings_df, activities_df)

        self._is_fitted = True
        log.info("HybridRecommender ready (CB + CF + Time-Aware).")
        return self

    # ── Recommend ─────────────────────────────────────────────────────────────
    def recommend(
        self,
        user_id: str,
        top_n: int = 10,
        context: Optional[dict] = None,
        exclude_seen: bool = True,
        db=None,
    ) -> pd.DataFrame:
        """
        Generates top-N recommendations for a user.

        Args:
            user_id:        User ID (must exist in users_df)
            top_n:          Number of activities to return
            context:        Optional contextual signals:
                            {"budget_max": 2, "experience_filter": ["cultura"]}
            exclude_seen:   Exclude activities already rated by user

        Returns:
            DataFrame with activities sorted by hybrid score.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before recommend()")

        context = context or {}
        # Build temporal context
        # In production pass dt=datetime.now() — here we use the passed context
        temporal = TemporalContext(
            slot     = context.get("time_slot"),
            day_type = context.get("day_type"),
            season   = context.get("season"),
        )

        # ── Retrieve user profile ────────────────────────────────────────────
        if user_id not in self.users_df.index:
            log.warning(f"User {user_id} not found. Using neutral profile.")
            preferences = {exp: 0.5 for exp in ["cultura", "natura", "cibo", "shopping", "vita_notturna", "svago", "altro"]}
            budget_max  = context.get("budget_max", 3)
        else:
            user = self.users_df.loc[user_id]
            preferences = json.loads(user["preferences_json"])
            budget_max  = context.get("budget_max", user.get("budget_max", 3))

        # ── Score Content-Based ──────────────────────────────────────────────
        cb_scores = self.cb.score_for_user(preferences)

        # ── Score Collaborative Filtering ────────────────────────────────────
        cf_score_dict = self.cf.score_for_user(user_id)

        # ── Alpha: read from DB if available, otherwise compute ─────────────
        # DB contains alpha updated in real-time by progressive profiling.
        # If unavailable (e.g., tests without DB), compute from rating count.
        if db is not None:
            alpha = db.get_user_alpha(user_id)
        else:
            n_user_ratings = len(self.ratings_df[self.ratings_df["user_id"] == user_id])
            alpha = self._adaptive_alpha(n_user_ratings)
            n_user_ratings = len(self.ratings_df[self.ratings_df["user_id"] == user_id])

        # ── Ensemble ─────────────────────────────────────────────────────────
        activities = self.activities_df.copy()
        activities["cb_score"] = cb_scores

        if cf_score_dict:
            activities["cf_score"] = activities["activity_id"].map(
                lambda aid: cf_score_dict.get(aid, 0.0)
            )
        else:
            activities["cf_score"] = 0.0

        vector_scores = self.vector.score_for_user(preferences)
        activities["vector_score"] = vector_scores
        if db is not None and hasattr(db, "search_by_embedding"):
            try:
                query_embedding = self.vector.build_query_embedding(preferences)
                ann_results = db.search_by_embedding(
                    query_embedding=query_embedding,
                    city=context.get("city"),
                    top_k=max(100, top_n * 10),
                )
                if not ann_results.empty:
                    ann_map = ann_results.set_index("activity_id")["vector_score"].to_dict()
                    ann_values = activities["activity_id"].map(ann_map)
                    activities["vector_score"] = ann_values.fillna(activities["vector_score"])
            except Exception as e:
                log.warning("Local Vector ANN fallback: %s", e)

        base_hybrid = (
            alpha * activities["cb_score"] +
            (1 - alpha) * activities["cf_score"] +
            self.delta * activities["vector_score"]
        )
        # Apply Time-Aware score
        time_scores = self.time_scorer.score(activities, temporal)
        activities["time_score"]   = time_scores
        activities["hybrid_score"] = self.time_scorer.apply_to_scores(
            pd.Series(base_hybrid.values, index=activities.index),
            activities,
            temporal,
        ).values

        # ── Context filters ──────────────────────────────────────────────────
        activities = self._apply_context_filters(activities, budget_max, context)

        # ── Exclude already seen ─────────────────────────────────────────────
        if exclude_seen:
            seen = set(self.ratings_df[self.ratings_df["user_id"] == user_id]["activity_id"])
            activities = activities[~activities["activity_id"].isin(seen)]

        # ── Final ranking ────────────────────────────────────────────────────
        result = (
            activities
            .sort_values("hybrid_score", ascending=False)
            .head(top_n)
            [["activity_id", "name", "category", "experience_type",
              "rating", "price_tier", "lat", "lng",
                            "cb_score", "cf_score", "vector_score", "time_score", "hybrid_score"]]
            .reset_index(drop=True)
        )

        log.info(
            f"Recommendations for {user_id}: {len(result)} results "
                        f"(α={alpha:.2f}, δ={self.delta:.2f}, {n_user_ratings} previous ratings)"
        )
        return result

    # ── Evaluation ───────────────────────────────────────────────────────────
    def evaluate(self, test_ratings: pd.DataFrame, k: int = 10) -> dict:
        """
        Evaluates recommender with Precision@K and NDCG@K.

        Args:
            test_ratings:   Test ratings (not seen during fit)
            k:              Top-K threshold

        Returns:
            Dict with aggregate metrics.
        """
        precisions, ndcgs = [], []

        test_users = test_ratings["user_id"].unique()

        for user_id in test_users:
            # Ground truth: activities with rating >= 3.5 (considered "positive")
            relevant = set(
                test_ratings[
                    (test_ratings["user_id"] == user_id) &
                    (test_ratings["rating"] >= 3.5)
                ]["activity_id"]
            )
            if not relevant:
                continue

            # Predictions
            try:
                recs = self.recommend(user_id, top_n=k, exclude_seen=False)
                recommended = list(recs["activity_id"])
            except Exception:
                continue

            # Precision@K
            hits = len(set(recommended[:k]) & relevant)
            precisions.append(hits / k)

            # NDCG@K
            ndcgs.append(self._ndcg(recommended[:k], relevant))

        return {
            "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
            "ndcg_at_k":      float(np.mean(ndcgs)) if ndcgs else 0.0,
            "k":              k,
            "n_users_eval":   len(precisions),
        }

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _adaptive_alpha(self, n_ratings: int) -> float:
        """
        Adaptive alpha: the more ratings a user has, the more CF is trusted.
        - 0 rating  → α = 1.0 (CB only, cold start)
        - 5+ rating → α = 0.5 (balanced)
        - 20+ rating→ α = 0.2 (CF-dominant)
        """
        if n_ratings == 0:
            return 1.0
        elif n_ratings < self.cold_start_threshold:
            return 1.0 - (n_ratings / self.cold_start_threshold) * 0.5
        else:
            return max(0.2, 0.5 - (n_ratings - self.cold_start_threshold) * 0.01)

    def _apply_context_filters(
        self, df: pd.DataFrame, budget_max: int, context: dict
    ) -> pd.DataFrame:
        """Applies contextual filters to candidate activities."""
        # Budget filter
        df = df[df["price_tier"].fillna(budget_max) <= budget_max]

        # Experience-type filter
        if "experience_filter" in context:
            allowed = context["experience_filter"]
            df = df[df["experience_type"].isin(allowed)]

        return df

    @staticmethod
    def _ndcg(recommended: list, relevant: set) -> float:
        """Computes NDCG for a single query."""
        dcg = sum(
            1.0 / np.log2(i + 2)
            for i, act in enumerate(recommended)
            if act in relevant
        )
        ideal_hits = min(len(relevant), len(recommended))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0


# ── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.collectors.synthetic_users import SyntheticUserGenerator, EXPERIENCE_TYPES
    import numpy as np

    print("=== Demo HybridRecommender ===\n")

    # Mock activities
    np.random.seed(42)
    n_act = 80
    activities = pd.DataFrame({
        "activity_id":    [f"ACT_{i:04d}" for i in range(n_act)],
        "name":           [f"Place {i}" for i in range(n_act)],
        "category":       np.random.choice(["museum","restaurant","park","bar","market"], n_act),
        "experience_type":np.random.choice(EXPERIENCE_TYPES, n_act),
        "rating":         np.random.uniform(3.0, 5.0, n_act),
        "review_count":   np.random.randint(10, 2000, n_act),
        "price_tier":     np.random.randint(1, 5, n_act),
        "lat":            np.random.uniform(41.0, 42.0, n_act),
        "lng":            np.random.uniform(12.0, 13.0, n_act),
        "description":    ["interesting place " + str(i) for i in range(n_act)],
        "tags":           ['["arte", "storia"]'] * n_act,
    })

    # Generate synthetic users and ratings
    gen     = SyntheticUserGenerator(seed=42)
    users   = gen.generate_users(n=100)
    ratings = gen.generate_ratings(users, activities, n_ratings_per_user=12)

    # Train/test split (80/20)
    split   = int(len(ratings) * 0.8)
    train_r = ratings.iloc[:split]
    test_r  = ratings.iloc[split:]

    # Fit and recommend
    rec = HybridRecommender(alpha=0.5)
    rec.fit(activities, users, train_r)

    test_user = users["user_id"].iloc[0]
    results   = rec.recommend(test_user, top_n=5)
    print(f"Top-5 for {test_user}:")
    print(results.to_string(index=False))

    # Evaluate
    metrics = rec.evaluate(test_r, k=10)
    print(f"\nMetrics (test set):")
    print(f"  Precision@10 : {metrics['precision_at_k']:.4f}")
    print(f"  NDCG@10      : {metrics['ndcg_at_k']:.4f}")
    print(f"  Users eval   : {metrics['n_users_eval']}")
