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
        tokens = (1136, 1048, 1154, 1152, 1021, 1134, 1114, 1110, 1089, 1069, 1005, 1126, 1098, 1088,
                  1000, 1047, 1015, 1096, 1140, 1121, 1081, 1060, 1009, 1109, 1064, 1153, 1046, 1091,
                  1075, 1018, 1061, 1037, 1112, 1128, 1059, 1021, 1142, 1107, 1097, 1063, 1100, 1105,
                  1059, 1021, 1098, 1151, 1152, 1123, 1069, 1005, 1126, 1133, 1041, 1142, 1021, 1015,
                  1157, 1155, 1068, 1032, 1040, 1017, 1094, 1152, 1030, 1034, 1124, 1080, 1146, 1153,
                  1034, 1113, 1095, 1061, 1066, 1031, 1064, 1110, 1143, 1128, 1015, 1042, 1071, 1038,
                  1021, 1153, 1064, 1059, 1152, 1076, 1071, 1064, 1115, 1054, 1066, 1008, 1142, 1059)

        graph = AdjacencyMatrixGraph()

        graph.fill_from_tokens(tokens, 3)
        text_rank = VanillaTextRank(graph)

        text_rank.train()
        for token_id, score in text_rank.get_scores().items():
            self.assertIsInstance(token_id, int)
            self.assertIsInstance(score, float)
        top = text_rank.get_top_keywords(4)
        self.assertEqual(top, (1021, 1152, 1064, 1015))

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_sorting_order(self):
        graph = AdjacencyMatrixGraph()

        graph.fill_from_tokens((1, 2, 3, 4, 3), 3)

        text_rank = VanillaTextRank(graph)

        text_rank.train()
        for token_id, score in text_rank.get_scores().items():
            self.assertIsInstance(token_id, int)
            self.assertIsInstance(score, float)
        top = text_rank.get_top_keywords(4)
        self.assertEqual(top, (2, 3, 1, 4))

    @pytest.mark.lab_3_keywords_textrank
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_score_vertices_empty_graph(self):
        graph = AdjacencyMatrixGraph()

        text_rank = VanillaTextRank(graph)

        text_rank.train()
        self.assertEqual(text_rank.get_scores(), {})
