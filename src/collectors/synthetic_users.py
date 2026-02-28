"""
Synthetic User Generator
=========================
Generates synthetic users with realistic profiles to bootstrap
Collaborative Filtering before real users are available.

Theoretical basis:
- The 4 archetypes (families, couples, singles, groups) have different
    preference distributions → we simulate with different means/variances per archetype
- Interactions (ratings) are generated with a bias toward
    user-preferred categories + Gaussian noise

Usage:
    gen = SyntheticUserGenerator(seed=42)
    users_df = gen.generate_users(n=500)
    ratings_df = gen.generate_ratings(users_df, activities_df, n_ratings_per_user=15)
"""

import json
import numpy as np
import pandas as pd
from typing import Optional

# ── Archetype profiles ───────────────────────────────────────────────────────
# Fully English domain keys.
ARCHETYPE_PROFILES = {
    "family": {
        "culture":        0.6,
        "nature":         0.9,
        "food":           0.8,
        "shopping":       0.4,
        "nightlife":      0.1,
        "leisure":        0.9,
        "other":          0.5,
        # Behavioral parameters
        "budget_mean":    2.0,   # average tolerated price_tier
        "budget_std":     0.5,
        "trip_days_mean": 4.0,
        "group_size_mean": 4,
    },
    "couple": {
        "culture":        0.8,
        "nature":         0.7,
        "food":           0.9,
        "shopping":       0.6,
        "nightlife":      0.7,
        "leisure":        0.6,
        "other":          0.5,
        "budget_mean":    2.8,
        "budget_std":     0.7,
        "trip_days_mean": 3.5,
        "group_size_mean": 2,
    },
    "single": {
        "culture":        0.9,
        "nature":         0.6,
        "food":           0.7,
        "shopping":       0.4,
        "nightlife":      0.6,
        "leisure":        0.7,
        "other":          0.6,
        "budget_mean":    1.8,
        "budget_std":     0.6,
        "trip_days_mean": 2.5,
        "group_size_mean": 1,
    },
    "group": {
        "culture":        0.5,
        "nature":         0.6,
        "food":           0.8,
        "shopping":       0.7,
        "nightlife":      0.9,
        "leisure":        0.9,
        "other":          0.5,
        "budget_mean":    2.2,
        "budget_std":     0.6,
        "trip_days_mean": 3.0,
        "group_size_mean": 6,
    },
}

EXPERIENCE_TYPES = ["culture", "nature", "food", "shopping", "nightlife", "leisure", "other"]
ARCHETYPES = list(ARCHETYPE_PROFILES.keys())


class SyntheticUserGenerator:

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    # ── Generate users ───────────────────────────────────────────────────────
    def generate_users(self, n: int = 500) -> pd.DataFrame:
        """
        Generates n synthetic users with realistic preference profiles.

        Returns:
            DataFrame with columns: user_id, archetype, preferences_json,
            budget_max, trip_days, group_size, age_range
        """
        rows = []
        # Archetype distribution: families 30%, couples 30%, singles 25%, groups 15%
        archetype_probs = [0.30, 0.30, 0.25, 0.15]

        for i in range(n):
            archetype = self.rng.choice(ARCHETYPES, p=archetype_probs)
            profile   = ARCHETYPE_PROFILES[archetype]

            # Per-category preferences with individual noise
            preferences = {}
            for exp in EXPERIENCE_TYPES:
                base  = profile[exp]
                noise = self.rng.normal(0, 0.15)
                preferences[exp] = float(np.clip(base + noise, 0.0, 1.0))

            # Budget and trip parameters
            budget = int(np.clip(
                self.rng.normal(profile["budget_mean"], profile["budget_std"]), 1, 4
            ))
            trip_days = max(1, int(self.rng.normal(profile["trip_days_mean"], 1.0)))
            group_size = max(1, int(self.rng.normal(profile["group_size_mean"], 1.0)))

            rows.append({
                "user_id":          f"USR_{i:05d}",
                "archetype":        archetype,
                "preferences_json": json.dumps(preferences),
                "budget_max":       budget,
                "trip_days":        trip_days,
                "group_size":       group_size,
                "age_range":        self._random_age_range(archetype),
            })

        df = pd.DataFrame(rows)
        print(f"✅ Generated {len(df)} synthetic users.")
        print(df["archetype"].value_counts().to_string())
        return df

    # ── Generate ratings ─────────────────────────────────────────────────────
    def generate_ratings(
        self,
        users_df: pd.DataFrame,
        activities_df: pd.DataFrame,
        n_ratings_per_user: int = 15,
        implicit_only: bool = False,
    ) -> pd.DataFrame:
        """
                Generates a user-activity interaction matrix.

                Logic:
                - Each user rates ~n_ratings_per_user activities
                - Rating is influenced by alignment between
                    user preferences and activity category
                - Adds noise to simulate real behavior

        Args:
            users_df:            Output of generate_users()
            activities_df:       POI DataFrame with 'experience_type' column
            n_ratings_per_user:  Number of interactions per user
            implicit_only:       If True, generates only 0/1 (clicked/not clicked)

        Returns:
            DataFrame with: user_id, activity_id, rating, event_type
        """
        rows = []
        act_ids   = activities_df["activity_id"].values
        act_types = activities_df["experience_type"].values

        for _, user in users_df.iterrows():
            prefs = json.loads(user["preferences_json"])

            # Sample activities with preference-weighted probabilities
            weights = np.array([
                prefs.get(exp_type, 0.5) for exp_type in act_types
            ])
            weights = weights / weights.sum()

            n_sample = min(n_ratings_per_user, len(act_ids))
            sampled_idx = self.rng.choice(len(act_ids), size=n_sample, replace=False, p=weights)

            for idx in sampled_idx:
                act_id   = act_ids[idx]
                exp_type = act_types[idx]
                pref_val = prefs.get(exp_type, 0.5)

                if implicit_only:
                    # Binary interaction: 1 = visited
                    rating     = 1
                    event_type = "visit"
                else:
                    # Rating 1-5 based on preference + noise
                    base_rating = 1 + pref_val * 4          # 1-5 scale
                    noise       = self.rng.normal(0, 0.8)
                    rating      = float(np.clip(base_rating + noise, 1.0, 5.0))
                    rating      = round(rating, 1)
                    event_type  = "rating"

                rows.append({
                    "user_id":     user["user_id"],
                    "activity_id": act_id,
                    "rating":      rating,
                    "event_type":  event_type,
                    "archetype":   user["archetype"],   # useful for analysis
                })

        df = pd.DataFrame(rows)
        print(f"✅ Generated {len(df)} ratings ({n_ratings_per_user} per user).")
        return df

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _random_age_range(self, archetype: str) -> str:
        ranges = {
            "family": ["30-40", "35-45", "40-50"],
            "couple": ["25-35", "30-40", "35-45"],
            "single": ["20-30", "25-35", "30-40"],
            "group":  ["20-30", "22-32", "25-35"],
        }
        return str(self.rng.choice(ranges.get(archetype, ["25-35"])))


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Minimal activities_df simulation for quick test
    mock_activities = pd.DataFrame({
        "activity_id":    [f"ACT_{i:04d}" for i in range(50)],
        "experience_type": np.random.choice(EXPERIENCE_TYPES, 50),
    })

    gen = SyntheticUserGenerator(seed=42)
    users   = gen.generate_users(n=200)
    ratings = gen.generate_ratings(users, mock_activities, n_ratings_per_user=10)

    users.to_csv("data/processed/users_synthetic.csv", index=False)
    ratings.to_csv("data/processed/ratings_synthetic.csv", index=False)
    print("\nSample users:")
    print(users.head(3).to_string())
    print("\nSample rating:")
    print(ratings.head(5).to_string())
