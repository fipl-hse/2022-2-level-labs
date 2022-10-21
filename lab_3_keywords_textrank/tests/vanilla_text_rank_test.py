"""
Checks the third lab VanillaTextRank
"""

import unittest

import pytest

from lab_3_keywords_textrank.main import VanillaTextRank, AdjacencyMatrixGraph


class VanillaTextRankTest(unittest.TestCase):
    """
    Tests VanillaTextRank
    """

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_score_vertices(self):
        graph = AdjacencyMatrixGraph()

        graph.fill_from_tokens((1, 2, 3, 4, 3), 3)

        text_rank = VanillaTextRank(graph)

        text_rank.score_vertices()
        for token_id, score in text_rank.get_scores().items():
            self.assertIsInstance(token_id, int)
            self.assertIsInstance(score, float)
        top = text_rank.get_top_keywords(4)
        self.assertEqual(top, (2, 3, 4, 1))

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_score_vertices_empty_graph(self):
        graph = AdjacencyMatrixGraph()

        text_rank = VanillaTextRank(graph)

        text_rank.score_vertices()
        self.assertEqual(text_rank.get_scores(), {})
