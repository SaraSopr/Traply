import unittest

from src.utils.database import DatabaseManager


class TestDatabaseUtils(unittest.TestCase):
    def test_adaptive_alpha_floor(self):
        alpha0 = DatabaseManager._adaptive_alpha(0)
        alpha10 = DatabaseManager._adaptive_alpha(10)
        alpha100 = DatabaseManager._adaptive_alpha(100)

        self.assertGreaterEqual(alpha0, alpha10)
        self.assertGreaterEqual(alpha10, alpha100)
        self.assertGreaterEqual(alpha100, 0.20)

    def test_pgvector_literal_format(self):
        literal = DatabaseManager._to_pgvector_literal([0.1, 0.2, 0.3])
        self.assertTrue(literal.startswith("["))
        self.assertTrue(literal.endswith("]"))
        self.assertIn(",", literal)
