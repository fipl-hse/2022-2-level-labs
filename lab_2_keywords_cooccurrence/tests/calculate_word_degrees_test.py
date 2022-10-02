"""
Checks the second lab calculate_word_degrees function
"""

import unittest

import pytest

from lab_2_keywords_cooccurrence.main import calculate_word_degrees


class CalculateWordDegreesTest(unittest.TestCase):
    """
    Tests calculate_word_degrees function
    """

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_word_degrees_basic(self):
        candidate_keyword_phrases = [('времена', 'советского', 'союза', 'исследование', 'космоса'),
                                     ('одной',), ('важнейших', 'задач'),
                                     ('времена', 'союза', 'прошли')]
        content_words = ['времена', 'советского', 'союза', 'исследование',
                         'космоса', 'одной', 'важнейших', 'задач', 'прошли']
        expected = {'времена': 8, 'советского': 5, 'союза': 8, 'исследование': 5, 'космоса': 5,
                    'одной': 1, 'важнейших': 2, 'задач': 2, 'прошли': 3}
        actual = calculate_word_degrees(candidate_keyword_phrases, content_words)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_word_degrees_content_words_are_not_in_phrases(self):
        candidate_keyword_phrases = [('времена', 'советского', 'союза', 'исследование', 'космоса'),
                                     ('одной',), ('важнейших', 'задач'),
                                     ('времена', 'союза', 'прошли')]
        content_words = ['времена', 'советского', 'союза', 'исследование',
                         'космоса', 'одной', 'важнейших', 'задач', 'прошли', 'именно', 'ссср']
        expected = {'времена': 8, 'советского': 5, 'союза': 8, 'исследование': 5, 'космоса': 5,
                    'одной': 1, 'важнейших': 2, 'задач': 2, 'прошли': 3, 'именно': 0, 'ссср': 0}
        actual = calculate_word_degrees(candidate_keyword_phrases, content_words)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_word_degrees_phrases_are_not_in_content_words(self):
        candidate_keyword_phrases = [('времена', 'советского', 'союза', 'исследование', 'космоса'),
                                     ('одной',), ('важнейших', 'задач'),
                                     ('времена', 'союза', 'прошли')]
        content_words = ['времена', 'советского', 'союза', 'исследование']
        expected = {'времена': 8, 'советского': 5, 'союза': 8, 'исследование': 5}
        actual = calculate_word_degrees(candidate_keyword_phrases, content_words)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_word_degrees_empty_arguments(self):
        candidate_keyword_phrases = [('времена', 'советского', 'союза', 'исследование', 'космоса'),
                                     ('одной',), ('важнейших', 'задач'),
                                     ('времена', 'союза', 'прошли')]
        content_words = ['времена', 'советского', 'союза', 'исследование',
                         'космоса', 'одной', 'важнейших', 'задач', 'прошли']
        expected = None
        actual = calculate_word_degrees(candidate_keyword_phrases, [])
        self.assertEqual(expected, actual)
        actual = calculate_word_degrees([], content_words)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_word_degrees_wrong_argument_types(self):
        candidate_keyword_phrases = [('времена', 'советского', 'союза', 'исследование', 'космоса'),
                                     ('одной',), ('важнейших', 'задач'),
                                     ('времена', 'союза', 'прошли')]
        content_words = ['времена', 'советского', 'союза', 'исследование',
                         'космоса', 'одной', 'важнейших', 'задач', 'прошли']
        expected = None

        wrong_inputs = [4, 4.0, False, {'dict': 0}]
        for wrong_input in wrong_inputs:
            actual = calculate_word_degrees(candidate_keyword_phrases, wrong_input)
            self.assertEqual(expected, actual)
            actual = calculate_word_degrees(wrong_input, content_words)
            self.assertEqual(expected, actual)
