import unittest
import pandas as pd

from src.collectors.synthetic_users import SyntheticUserGenerator


class TestSyntheticUserGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = SyntheticUserGenerator(seed=42)

    def test_generate_users_shape(self):
        users_df = self.generator.generate_users(n=30)
        self.assertEqual(len(users_df), 30)
        self.assertIn("user_id", users_df.columns)
        self.assertIn("preferences_json", users_df.columns)
        self.assertIn("archetype", users_df.columns)

    def test_generate_ratings_not_empty(self):
        users_df = self.generator.generate_users(n=10)
        activities_df = pd.DataFrame(
            {
                "activity_id": [f"ACT_{i:03d}" for i in range(20)],
                "experience_type": ["cultura", "natura", "cibo", "shopping", "svago"] * 4,
            }
        )
        ratings_df = self.generator.generate_ratings(users_df, activities_df, n_ratings_per_user=5)

        self.assertFalse(ratings_df.empty)
        self.assertIn("user_id", ratings_df.columns)
        self.assertIn("activity_id", ratings_df.columns)
        self.assertIn("rating", ratings_df.columns)
