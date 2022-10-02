"""
Checks the second lab calculate_cumulative_score_for_candidates_with_stop_words function
"""

import unittest

import pytest

from lab_2_keywords_cooccurrence.main import calculate_cumulative_score_for_candidates_with_stop_words


class CalculateCumulativeScoreForCandidatesWithStopWordsTest(unittest.TestCase):
    """
    Tests calculate_cumulative_score_for_candidates function
    """

    candidate_keyword_phrases = [('времена', 'советского', 'союза', 'исследование', 'космоса'),
                                 ('одной',), ('важнейших', 'задач'), ('времена', 'союза', 'прошли'),
                                 ('одной',), ('важнейших', 'задач'),
                                 ('одной', 'из', 'важнейших', 'задач')]

    stop_words = ['из']
    word_scores = {'времена': 4.0, 'советского': 5.0, 'союза': 4.0, 'исследование': 5.0,
                   'космоса': 5.0, 'одной': 1.0, 'важнейших': 2.0, 'задач': 2.0, 'прошли': 3.0}

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_cumulative_score_for_candidates_with_stop_words_basic(self):
        expected = {
            ('времена', 'советского', 'союза', 'исследование', 'космоса'): 23,
            ('одной',): 1,
            ('важнейших', 'задач'): 4,
            ('времена', 'союза', 'прошли'): 11,
            ('одной', 'из', 'важнейших', 'задач'): 5
        }
        actual = calculate_cumulative_score_for_candidates_with_stop_words(self.candidate_keyword_phrases,
                                                                           self.word_scores,
                                                                           self.stop_words)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_cumulative_score_for_candidates_with_stop_words_empty_candidate_kw(self):

        expected = None
        actual = calculate_cumulative_score_for_candidates_with_stop_words([],
                                                                           self.word_scores,
                                                                           self.stop_words)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_cumulative_score_for_candidates_with_stop_words_empty_word_scores(self):
        expected = None
        actual = calculate_cumulative_score_for_candidates_with_stop_words(self.candidate_keyword_phrases,
                                                                           {},
                                                                           self.stop_words)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_cumulative_score_for_candidates_with_stop_words_empty_stop_words(self):
        expected = None
        actual = calculate_cumulative_score_for_candidates_with_stop_words(self.candidate_keyword_phrases,
                                                                           self.word_scores,
                                                                           [])
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_cumulative_score_for_candidates_with_stop_words_wrong_argument_types(self):
        wrong_inputs = [4.0, 4, 'string', {'dict': 'dict'}, True]
        expected = None
        for wrong_input in wrong_inputs:
            actual = calculate_cumulative_score_for_candidates_with_stop_words(wrong_input,
                                                                               self.word_scores,
                                                                               self.stop_words)
            self.assertEqual(expected, actual)
            actual = calculate_cumulative_score_for_candidates_with_stop_words(self.candidate_keyword_phrases,
                                                                               self.word_scores,
                                                                               wrong_input)
            self.assertEqual(expected, actual)

        wrong_inputs = [4.0, 4, 'string', ['l', 'i', 's', 't'], True]
        for wrong_input in wrong_inputs:
            actual = calculate_cumulative_score_for_candidates_with_stop_words(self.candidate_keyword_phrases,
                                                                               wrong_input,
                                                                               self.stop_words)
            self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_calculate_cumulative_score_for_candidates_with_stop_words_stop_words_have_scores(self):
        self.word_scores.update({self.stop_words[0]: 1})
        expected = {
            ('времена', 'советского', 'союза', 'исследование', 'космоса'): 23,
            ('одной',): 1,
            ('важнейших', 'задач'): 4,
            ('времена', 'союза', 'прошли'): 11,
            ('одной', 'из', 'важнейших', 'задач'): 5
        }
        actual = calculate_cumulative_score_for_candidates_with_stop_words(self.candidate_keyword_phrases,
                                                                           self.word_scores,
                                                                           self.stop_words)
        self.assertEqual(expected, actual)
