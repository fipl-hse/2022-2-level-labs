"""
Checks the third lab AdjacencyMatrixGraph
"""

import unittest

import pytest

from lab_3_keywords_textrank.main import AdjacencyMatrixGraph


class AdjacencyMatrixGraphTest(unittest.TestCase):
    """
    Tests AdjacencyMatrixGraph
    """

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_add_edge(self):
        graph = AdjacencyMatrixGraph()

        self.assertEqual(graph.add_edge(1, 2), 0)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        graph.add_edge(10, 3)
        self.assertEqual(len(graph.get_vertices()), 4)

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_add_edge_incorrect_input(self):
        graph = AdjacencyMatrixGraph()

        self.assertEqual(graph.add_edge(1, 1), -1)


    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_vertices(self):
        graph = AdjacencyMatrixGraph()

        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        graph.add_edge(10, 3)
        self.assertEqual(graph.get_vertices(), (1, 2, 3, 10))

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_edge_weight(self):
        graph = AdjacencyMatrixGraph()

        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        graph.add_edge(10, 3)

        self.assertEqual(graph.is_incidental(1, 2), 1)
        self.assertEqual(graph.is_incidental(2, 1), 1)
        self.assertEqual(graph.is_incidental(10, 2), 0)

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_edge_weight_incorrect_input(self):
        graph = AdjacencyMatrixGraph()
        graph.add_edge(1, 2)

        self.assertEqual(graph.is_incidental(1, 100), -1)
        self.assertEqual(graph.is_incidental(100, 1), -1)

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_inout_score(self):
        graph = AdjacencyMatrixGraph()

        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        graph.add_edge(10, 3)

        self.assertEqual(graph.calculate_inout_score(1), 2)
        self.assertEqual(graph.calculate_inout_score(10), 1)

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_inout_score_incorrect_input(self):
        graph = AdjacencyMatrixGraph()
        graph.add_edge(1, 2)

        self.assertEqual(graph.calculate_inout_score(100), -1)

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_fill_graph(self):
        graph = AdjacencyMatrixGraph()

        graph.fill_from_tokens((1, 2, 3, 4, 3), 3)

        self.assertEqual(graph.calculate_inout_score(1), 2)
        self.assertCountEqual(graph.get_vertices(), (1, 2, 3, 4))

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_positions_weights(self):
        graph = AdjacencyMatrixGraph()

        graph.fill_positions((1, 256, 4, 95, 1, 420, 5))

        graph.calculate_position_weights()
        for idx, weight in graph.get_position_weights().items():
            self.assertIsInstance(idx, int)
            self.assertIsInstance(weight, float)
        self.assertAlmostEqual(sum(graph.get_position_weights().values()), 1.0)
