import unittest
import pandas as pd

from src.recommender.time_aware import TemporalContext, TimeAwareScorer


class TestTimeAwareLayer(unittest.TestCase):
    def setUp(self):
        self.scorer = TimeAwareScorer(gamma=0.25)
        self.activities = pd.DataFrame(
            {
                "activity_id": ["A1", "A2", "A3"],
                "experience_type": ["cultura", "cibo", "vita_notturna"],
                "opening_hours": [None, None, None],
            }
        )

    def test_temporal_context_manual(self):
        context = TemporalContext(slot="evening", day_type="weekend", season="summer")
        self.assertEqual(context.slot, "evening")
        self.assertEqual(context.day_type, "weekend")
        self.assertEqual(context.season, "summer")

    def test_time_scores_range(self):
        context = TemporalContext(slot="morning", day_type="weekday", season="spring")
        scores = self.scorer.score(self.activities, context)
        self.assertEqual(len(scores), len(self.activities))
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())
