"""
Checks the second lab calculate_frequencies_for_content_words function
"""

import unittest

import pytest

from lab_2_keywords_cooccurrence.main import calculate_frequencies_for_content_words


class ExtractContentWordsAndCalculateFrequenciesTest(unittest.TestCase):
    """
    Tests extract_content_words_and_calculate_frequencies function
    """

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_frequencies_for_content_words_basic(self):
        candidate_keyword_phrases = [('времена', 'советского', 'союза', 'исследование', 'космоса'),
                                     ('одной',), ('важнейших', 'задач', 'времена')]
        expected = {'времена': 2, 'советского': 1, 'союза': 1, 'исследование': 1,
                    'космоса': 1, 'одной': 1, 'важнейших': 1, 'задач': 1}

        actual = calculate_frequencies_for_content_words(candidate_keyword_phrases)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_frequencies_for_content_words_empty_argument(self):
        candidate_keyword_phrases = []
        expected = None
        actual = calculate_frequencies_for_content_words(candidate_keyword_phrases)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_frequencies_for_content_words_wrong_argument_types(self):
        wrong_inputs = [4, 4.0, False, {'dict': 0}, 'string']
        expected = None
        for candidate_keyword_phrases in wrong_inputs:
            actual = calculate_frequencies_for_content_words(candidate_keyword_phrases)
            self.assertEqual(expected, actual)
