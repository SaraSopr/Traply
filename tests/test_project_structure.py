import unittest


class TestProjectStructure(unittest.TestCase):
    def test_src_imports(self):
        from src.collectors.apify_collector import ApifyCollector
        from src.collectors.synthetic_users import SyntheticUserGenerator
        from src.recommender.hybrid_recommender import HybridRecommender
        from src.recommender.time_aware import TimeAwareScorer
        from src.recommender.vector_layer import VectorSemanticRecommender
        from src.utils.database import DatabaseManager

        self.assertTrue(ApifyCollector)
        self.assertTrue(SyntheticUserGenerator)
        self.assertTrue(HybridRecommender)
        self.assertTrue(TimeAwareScorer)
        self.assertTrue(VectorSemanticRecommender)
        self.assertTrue(DatabaseManager)
