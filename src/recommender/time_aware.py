"""
Time-Aware Recommendation Layer
=================================
Adds temporal context to the recommendation system.

Theory (for thesis citation):
  Baltrunas et al. (2011) — "Context-Aware Matrix Factorization"
  Adomavicius & Tuzhilin (2015) — "Context-Aware Recommender Systems"

Formula:
  score(u, i, t) = α·CB(u,i) + (1-α)·CF(u,i) + γ·Time(i,t)

Where Time(i,t) is the compatibility of activity i with temporal
context t = {time_slot, day_type, season}.

This layer works in 3 modes:
    1. Static prior      → matrix initialized with domain knowledge
    2. Learned from data → updated from real interactions
    3. Personalized      → each archetype has different temporal weights
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# ── Temporal slots ───────────────────────────────────────────────────────────
SLOTS      = ["morning", "afternoon", "evening", "night"]
DAY_TYPES  = ["weekday", "weekend"]
SEASONS    = ["spring", "summer", "autumn", "winter"]

# ── Prior domain knowledge: category × time-slot compatibility ──────────────
# Values in [0, 1]. Initialized with common-sense priors and updated from data.
# Source: tourism literature + common sense
TIME_PRIOR = {
#                        morn  aftn  even  nght
    "cultura":          [0.90, 0.75, 0.35, 0.05],
    "natura":           [0.85, 0.90, 0.60, 0.10],
    "cibo":             [0.55, 0.75, 0.95, 0.65],
    "shopping":         [0.50, 0.95, 0.55, 0.05],
    "vita_notturna":    [0.05, 0.15, 0.80, 1.00],
    "svago":            [0.70, 0.85, 0.65, 0.20],
    "alloggio":         [0.30, 0.30, 0.80, 0.90],
    "altro":            [0.60, 0.70, 0.60, 0.30],
}

# Weekend vs weekday prior: many activities are more popular on weekends
WEEKEND_BOOST = {
    "cultura":       1.10,
    "natura":        1.15,
    "cibo":          1.10,
    "shopping":      1.20,
    "vita_notturna": 1.25,
    "svago":         1.20,
    "alloggio":      1.00,
    "altro":         1.05,
}

# Seasonal prior: boost/penalty by season
SEASON_MULTIPLIER = {
#                    spring  summer  autumn  winter
    "natura":       [1.10,   1.20,   1.05,   0.75],
    "cibo":         [1.00,   1.00,   1.00,   1.00],
    "cultura":      [1.00,   0.90,   1.05,   1.10],  # museums more visited in winter
    "vita_notturna":[1.05,   1.20,   1.00,   0.85],
    "svago":        [1.05,   1.15,   1.00,   0.80],
    "shopping":     [1.00,   0.95,   1.10,   1.15],  # holiday shopping effect
    "alloggio":     [1.00,   1.00,   1.00,   1.00],
    "altro":        [1.00,   1.00,   1.00,   1.00],
}


class TemporalContext:
    """
    Extracts and represents current temporal context.
    Can be built from real datetime or manually specified values.
    """

    def __init__(
        self,
        dt: Optional[datetime] = None,
        slot: Optional[str]    = None,
        day_type: Optional[str] = None,
        season: Optional[str]  = None,
    ):
        dt = dt or datetime.now()

        self.slot     = slot     or self._hour_to_slot(dt.hour)
        self.day_type = day_type or ("weekend" if dt.weekday() >= 5 else "weekday")
        self.season   = season   or self._month_to_season(dt.month)
        self.hour     = dt.hour
        self.datetime = dt

    @staticmethod
    def _hour_to_slot(hour: int) -> str:
        """
        Maps hour of day to slot.
        6-11   → morning
        12-17  → afternoon
        18-22  → evening
        23-5   → night
        """
        if   6  <= hour < 12: return "morning"
        elif 12 <= hour < 18: return "afternoon"
        elif 18 <= hour < 23: return "evening"
        else:                  return "night"

    @staticmethod
    def _month_to_season(month: int) -> str:
        if   month in [3, 4, 5]:  return "spring"
        elif month in [6, 7, 8]:  return "summer"
        elif month in [9, 10, 11]: return "autumn"
        else:                      return "winter"

    def to_dict(self) -> dict:
        return {
            "slot":     self.slot,
            "day_type": self.day_type,
            "season":   self.season,
            "hour":     self.hour,
        }

    def __repr__(self):
        return f"TemporalContext({self.slot}, {self.day_type}, {self.season})"


class TimeAwareScorer:
    """
        Computes Time_score for each activity given a temporal context.

        The score combines 3 signals:
            1. Time slot  (morning / afternoon / evening / night)
            2. Day type   (weekday / weekend)
            3. Season     (spring / summer / autumn / winter)

        Prior can be updated with real data via learn_from_interactions().
    """

    def __init__(self, gamma: float = 0.25):
        """
        Args:
                 gamma: Time_score weight in final calculation
                   score = α·CB + (1-α)·CF + γ·Time
                     Typical value: 0.15-0.30
                     It should not dominate CB and CF — it is a context signal
        """
        self.gamma    = gamma
        self.prior    = {k: np.array(v, dtype=float) for k, v in TIME_PRIOR.items()}
        self.learned  = {}   # updates learned from real data
        self._fitted  = False

    # ── Main scoring ─────────────────────────────────────────────────────────

    def score(
        self,
        activities_df: pd.DataFrame,
        context: TemporalContext,
    ) -> np.ndarray:
        """
        Computes Time_score for each activity in the DataFrame.

        Returns:
            Normalized array [0,1] with one score per activity.
        """
        slot_idx    = SLOTS.index(context.slot)
        season_idx  = SEASONS.index(context.season)
        is_weekend  = context.day_type == "weekend"

        scores = np.zeros(len(activities_df))

        for i, (_, row) in enumerate(activities_df.iterrows()):
            exp_type = row.get("experience_type", "altro")

            # 1. Base score from slot
            prior = self._get_prior(exp_type)
            slot_score = prior[slot_idx]

            # 2. Weekend boost
            weekend_mult = WEEKEND_BOOST.get(exp_type, 1.0) if is_weekend else 1.0

            # 3. Seasonal multiplier
            season_mult = self._season_multiplier(exp_type, season_idx)

            # 4. Penalty if activity is likely closed (opening hours)
            open_factor = self._opening_hours_factor(row, context.hour)

            scores[i] = slot_score * weekend_mult * season_mult * open_factor

        # Normalize to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()

        return scores

    # ── Learning from data ───────────────────────────────────────────────────

    def learn_from_interactions(
        self,
        ratings_df: pd.DataFrame,
        activities_df: pd.DataFrame,
        learning_rate: float = 0.1,
    ):
        """
                Updates temporal prior based on real interactions.

                Logic:
                    If users rate museums highly in the evening (against the prior),
                    the system learns and increases the weight for "cultura × evening".

                This transforms static prior into a data-learned model —
                a key point for the thesis methodology section.
        """
        if ratings_df.empty or activities_df.empty:
            return

        # Join ratings with experience type
        merged = ratings_df.merge(
            activities_df[["activity_id", "experience_type"]],
            on="activity_id", how="left"
        )
        merged = merged.dropna(subset=["experience_type"])

        # Group by (experience_type, simulated hour)
        # Note: in production you would use real interaction timestamps
        # With synthetic data we use a simulated hourly distribution
        np.random.seed(42)
        merged["hour"] = np.random.choice(
            [8, 10, 14, 16, 19, 21, 23],
            size=len(merged),
            p=[0.10, 0.15, 0.20, 0.20, 0.15, 0.15, 0.05]
        )
        merged["slot"] = merged["hour"].apply(TemporalContext._hour_to_slot)

        learned = {}
        for exp_type in merged["experience_type"].unique():
            slot_scores = np.zeros(4)
            slot_counts = np.zeros(4)

            subset = merged[merged["experience_type"] == exp_type]
            for _, row in subset.iterrows():
                idx = SLOTS.index(row["slot"])
                normalized_rating = row["rating"] / 5.0
                slot_scores[idx] += normalized_rating
                slot_counts[idx] += 1

            # Mean per slot (where data exists)
            mask = slot_counts > 0
            avg_scores = np.where(mask, slot_scores / (slot_counts + 1e-9), 0)

            # Blend prior + learned
            prior = self._get_prior(exp_type)
            blended = (1 - learning_rate) * prior + learning_rate * avg_scores
            learned[exp_type] = blended

        self.learned  = learned
        self._fitted  = True
        log.info(f"TimeAwareScorer: prior updated for {len(learned)} categories.")

    # ── Integration with HybridRecommender ──────────────────────────────────

    def apply_to_scores(
        self,
        base_scores: pd.Series,
        activities_df: pd.DataFrame,
        context: TemporalContext,
    ) -> pd.Series:
        """
        Adds temporal contribution to existing hybrid scores.

        Usage:
            hybrid_scores = α·CB + (1-α)·CF           # existing score
            final_scores  = hybrid_scores + γ·Time     # with time-aware layer

        Args:
            base_scores:    Series with hybrid_score for each activity
            activities_df:  Activities DataFrame (same order as base_scores)
            context:        Current temporal context
        """
        time_scores = self.score(activities_df, context)

        final = base_scores.values + self.gamma * time_scores
        # Re-normalize
        if final.max() > 0:
            final = final / final.max()

        return pd.Series(final, index=base_scores.index)

    # ── Analysis and explainability ─────────────────────────────────────────

    def explain(self, experience_type: str, context: TemporalContext) -> dict:
        """
        Explains why a category is (or is not) recommended now.
        Useful for the LLM layer and thesis explainability section.

        Returns:
            Dict with score, textual reason, and comparison across slots.
        """
        slot_idx   = SLOTS.index(context.slot)
        season_idx = SEASONS.index(context.season)
        prior      = self._get_prior(experience_type)
        slot_score = prior[slot_idx]
        season_mult = self._season_multiplier(experience_type, season_idx)
        final_score = slot_score * season_mult

        # Best slot for this category
        best_slot_idx = int(np.argmax(prior))
        best_slot     = SLOTS[best_slot_idx]

        return {
            "experience_type": experience_type,
            "context":         context.to_dict(),
            "time_score":      round(float(final_score), 3),
            "best_slot":       best_slot,
            "is_optimal_time": context.slot == best_slot,
            "all_slots": {
                slot: round(float(prior[i]), 3)
                for i, slot in enumerate(SLOTS)
            },
            "explanation": self._generate_explanation(
                experience_type, context.slot, best_slot, final_score
            ),
        }

    def get_best_time_for_category(self, experience_type: str) -> str:
        """Returns the best slot for a category. Useful for itinerary planning."""
        prior = self._get_prior(experience_type)
        return SLOTS[int(np.argmax(prior))]

    def rank_activities_by_time(
        self, activities_df: pd.DataFrame, context: TemporalContext
    ) -> pd.DataFrame:
        """
        Adds time_score column to DataFrame and sorts it.
        Used by itinerary optimizer to sequence activities.
        """
        df = activities_df.copy()
        df["time_score"] = self.score(df, context)
        return df.sort_values("time_score", ascending=False)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_prior(self, experience_type: str) -> np.ndarray:
        """Returns prior (learned if available, static otherwise)."""
        if self._fitted and experience_type in self.learned:
            return self.learned[experience_type]
        return self.prior.get(experience_type, self.prior["altro"])

    @staticmethod
    def _season_multiplier(experience_type: str, season_idx: int) -> float:
        mult = SEASON_MULTIPLIER.get(experience_type, [1.0, 1.0, 1.0, 1.0])
        return mult[season_idx]

    @staticmethod
    def _opening_hours_factor(row: pd.Series, hour: int) -> float:
        """
        Penalizes activities likely closed at the given hour.
        If opening_hours is missing, assume open.
        """
        hours_json = row.get("opening_hours")
        if not hours_json:
            return 1.0
        try:
            hours = json.loads(hours_json)
            if not hours:
                return 1.0
            # Simplified rule: bars/restaurants likely closed very early morning
            exp_type = row.get("experience_type", "altro")
            if exp_type == "vita_notturna" and hour < 18:
                return 0.1
            if exp_type == "cibo" and hour < 7:
                return 0.2
            return 1.0
        except Exception:
            return 1.0

    @staticmethod
    def _generate_explanation(
        exp_type: str, current_slot: str, best_slot: str, score: float
    ) -> str:
        if score >= 0.80:
            return f"Optimal time for {exp_type} ({current_slot})."
        elif score >= 0.50:
            return f"Good time for {exp_type}. Better at {best_slot}."
        else:
            return f"Non-ideal time for {exp_type}. Recommended at {best_slot}."


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    print("=== Time-Aware Scorer Demo ===\n")

    # Build test contexts
    contexts = [
        TemporalContext(slot="morning",   day_type="weekday", season="summer"),
        TemporalContext(slot="afternoon", day_type="weekend", season="summer"),
        TemporalContext(slot="evening",   day_type="weekend", season="summer"),
        TemporalContext(slot="night",     day_type="weekend", season="summer"),
    ]

    # Mock activities
    activities = pd.DataFrame({
        "activity_id":    [f"ACT_{i:03d}" for i in range(6)],
        "name":           ["Museo Borghese", "Ristorante Da Mario", "Bar San Carlo",
                           "Villa Borghese", "Club Goa", "Mercato Testaccio"],
        "experience_type":["cultura", "cibo", "cibo", "natura", "vita_notturna", "cibo"],
        "opening_hours":  [None] * 6,
        "rating":         [4.8, 4.5, 4.2, 4.6, 4.3, 4.7],
    })

    scorer = TimeAwareScorer(gamma=0.25)

    print(f"{'Activity':<25} {'morning':>8} {'afternoon':>10} {'evening':>8} {'night':>6}")
    print("─" * 60)

    for _, act in activities.iterrows():
        scores = []
        for ctx in contexts:
            s = scorer.score(activities[activities["activity_id"] == act["activity_id"]], ctx)
            scores.append(f"{s[0]:.2f}")
        print(f"{act['name']:<25} {'  '.join(scores)}")

    # Explain
    print("\n── Contextual explanation ──")
    ctx_sera = TemporalContext(slot="evening", day_type="weekend", season="summer")
    for exp in ["cultura", "cibo", "vita_notturna"]:
        exp_info = scorer.explain(exp, ctx_sera)
        print(f"  {exp:<18} score={exp_info['time_score']:.2f}  | {exp_info['explanation']}")
