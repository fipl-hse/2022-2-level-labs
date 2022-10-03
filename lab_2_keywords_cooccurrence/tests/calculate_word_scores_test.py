"""
Checks the second lab calculate_word_scores function
"""

import unittest

import pytest

from lab_2_keywords_cooccurrence.main import calculate_word_scores


class CalculateWordScoresTest(unittest.TestCase):
    """
    Tests calculate_word_scores function
    """

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_word_scores_basic(self):
        word_degrees = {'времена': 8, 'советского': 5, 'союза': 8, 'исследование': 5, 'космоса': 5,
                        'одной': 1, 'важнейших': 2, 'задач': 2}
        word_frequencies = {'времена': 2, 'советского': 1, 'союза': 2, 'исследование': 1,
                            'космоса': 1, 'одной': 1, 'важнейших': 1, 'задач': 1}
        expected = {'времена': 4.0, 'советского': 5.0, 'союза': 4.0, 'исследование': 5.0,
                    'космоса': 5.0, 'одной': 1.0, 'важнейших': 2.0, 'задач': 2.0}
        actual = calculate_word_scores(word_degrees, word_frequencies)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_word_scores_mismatching_keys_in_dictionaries(self):
        word_degrees = {'времена': 8, 'советского': 5, 'союза': 8, 'исследование': 5, 'космоса': 5,
                        'одной': 1, 'важнейших': 2}
        word_frequencies = {'союза': 2, 'исследование': 1,
                            'космоса': 1, 'одной': 1, 'важнейших': 1, 'задач': 1}
        expected = None
        actual = calculate_word_scores(word_degrees, word_frequencies)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_word_scores_empty_arguments(self):
        word_degrees = {'времена': 8, 'советского': 5, 'союза': 8, 'исследование': 5, 'космоса': 5,
                        'одной': 1, 'важнейших': 2, 'задач': 2}
        word_frequencies = {'времена': 2, 'советского': 1, 'союза': 2, 'исследование': 1,
                            'космоса': 1, 'одной': 1, 'важнейших': 1, 'задач': 1}
        expected = None
        actual = calculate_word_scores(word_degrees, {})
        self.assertEqual(expected, actual)
        actual = calculate_word_scores({}, word_frequencies)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_word_scores_wrong_argument_types(self):
        word_degrees = {'времена': 8, 'советского': 5, 'союза': 8, 'исследование': 5, 'космоса': 5,
                        'одной': 1, 'важнейших': 2, 'задач': 2}
        word_frequencies = {'времена': 2, 'советского': 1, 'союза': 2, 'исследование': 1,
                            'космоса': 1, 'одной': 1, 'важнейших': 1, 'задач': 1}
        expected = None
        wrong_inputs = [4, 4.0, False, ['list']]
        for wrong_input in wrong_inputs:
            actual = calculate_word_scores(word_degrees, wrong_input)
            self.assertEqual(expected, actual)
            actual = calculate_word_scores(wrong_input, word_frequencies)
            self.assertEqual(expected, actual)
