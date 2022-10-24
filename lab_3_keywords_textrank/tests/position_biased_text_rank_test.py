"""
Checks the third lab PositionBiasedTextRank
"""

import unittest

import pytest

from lab_3_keywords_textrank.main import PositionBiasedTextRank, AdjacencyMatrixGraph


class VanillaTextRankTest(unittest.TestCase):
    """
    Tests PositionBiasedTextRank
    """

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_score_vertices(self):
        graph = AdjacencyMatrixGraph()

        encoded_tokens = (1, 2, 3, 4, 3)

        graph.fill_from_tokens(encoded_tokens, 3)
        graph.fill_positions(encoded_tokens)
        graph.calculate_position_weights()

        text_rank = PositionBiasedTextRank(graph)

        text_rank.train()
        for token_id, score in text_rank.get_scores().items():
            self.assertIsInstance(token_id, int)
            self.assertIsInstance(score, float)
        self.assertEqual(text_rank.get_top_keywords(4), (3, 2, 1, 4))
