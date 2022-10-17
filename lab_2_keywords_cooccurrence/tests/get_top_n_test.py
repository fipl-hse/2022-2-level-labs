"""
Checks the second lab get_top_n function
"""

import unittest

import pytest

from lab_2_keywords_cooccurrence.main import get_top_n


class GetTopNTest(unittest.TestCase):
    """
    Tests get_top_n function
    """
    keyword_phrases_with_scores = {
        ('времена', 'советского', 'союза', 'исследование', 'космоса'): 23.0,
        ('одной',): 1.0,
        ('важнейших', 'задач'): 4.0,
        ('времена', 'союза', 'прошли'): 11.0
    }

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_n_basic(self):
        top_n = 2
        max_length = 3
        expected = ['времена союза прошли', 'важнейших задач']
        actual = get_top_n(self.keyword_phrases_with_scores, top_n, max_length)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_n_top_n_too_large(self):
        top_n = 1000000
        max_length = 5
        expected = ['времена советского союза исследование космоса',
                    'времена союза прошли',
                    'важнейших задач',
                    'одной']
        actual = get_top_n(self.keyword_phrases_with_scores, top_n, max_length)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_n_arg_top_n_too_large_max_len_is_limited(self):
        top_n = 1000000
        max_length = 3
        expected = ['времена союза прошли', 'важнейших задач', 'одной']
        actual = get_top_n(self.keyword_phrases_with_scores, top_n, max_length)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_n_max_len_too_small(self):
        keyword_phrases_with_scores = {
            ('времена', 'советского', 'союза', 'исследование', 'космоса'): 23.0,
            ('важнейших', 'задач'): 4.0,
            ('времена', 'союза', 'прошли'): 11.0
        }
        top_n = 1000000
        max_length = 1
        expected = []
        actual = get_top_n(keyword_phrases_with_scores, top_n, max_length)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_n_empty_keywords(self):
        keyword_phrases_with_scores = {}
        top_n = 10
        max_length = 10
        expected = None
        actual = get_top_n(keyword_phrases_with_scores, top_n, max_length)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_n_n_non_positive(self):
        max_length = 5
        expected = None

        top_n = 0
        actual = get_top_n(self.keyword_phrases_with_scores, top_n, max_length)
        self.assertEqual(expected, actual)

        top_n = -10
        actual = get_top_n(self.keyword_phrases_with_scores, top_n, max_length)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_n_max_len_non_positive(self):
        top_n = 10
        expected = None

        max_length = 0
        actual = get_top_n(self.keyword_phrases_with_scores, top_n, max_length)
        self.assertEqual(expected, actual)

        max_length = -10
        actual = get_top_n(self.keyword_phrases_with_scores, top_n, max_length)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_get_top_n_wrong_argument_types(self):
        top_n = 10
        max_length = 10
        expected = None

        wrong_keyword_inputs = [4, 4.0, False, ['list']]
        for wrong_keyword_input in wrong_keyword_inputs:
            actual = get_top_n(wrong_keyword_input, top_n, max_length)
            self.assertEqual(expected, actual)

        wrong_top_n_inputs = [4.0, False, ['list'], {0: 1}]
        for wrong_input in wrong_top_n_inputs:
            actual = get_top_n(self.keyword_phrases_with_scores, wrong_input, max_length)
            self.assertEqual(expected, actual)
            actual = get_top_n(self.keyword_phrases_with_scores, top_n, wrong_input)
            self.assertEqual(expected, actual)
