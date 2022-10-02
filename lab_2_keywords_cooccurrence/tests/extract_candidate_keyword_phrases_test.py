"""
Checks the second lab extract_candidate_keyword_phrases function
"""

import unittest

import pytest

from lab_2_keywords_cooccurrence.main import extract_candidate_keyword_phrases


class ExtractCandidateKeywordPhrasesTest(unittest.TestCase):
    """
    Tests extract_candidate_keyword_phrases function
    """

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_candidate_keyword_phrases_basic(self):
        tokenized_sentences = ['Во времена Советского Союза исследование '
                               'космоса было одной из важнейших задач',
                               'И именно из СССР была отправлена ракета',
                               'совершившая первый полет в космос']
        stop_words = ['во', 'было', 'из', 'и', 'в']
        expected = [('времена', 'советского', 'союза', 'исследование', 'космоса'),
                    ('одной',), ('важнейших', 'задач'), ('именно',),
                    ('ссср', 'была', 'отправлена', 'ракета'),
                    ('совершившая', 'первый', 'полет'), ('космос',)]

        actual = extract_candidate_keyword_phrases(tokenized_sentences, stop_words)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_candidate_keyword_phrases_many_stop_words(self):
        tokenized_sentences = ['Во во во было из времена Советского Союза исследование '
                               'космоса было во было из одной из во было важнейших '
                               'задач было из во']
        stop_words = ['во', 'было', 'из']
        expected = [('времена', 'советского', 'союза', 'исследование', 'космоса'),
                    ('одной',), ('важнейших', 'задач')]

        actual = extract_candidate_keyword_phrases(tokenized_sentences, stop_words)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_candidate_keyword_phrases_stop_words_only(self):
        tokenized_sentences = ['Во во во было',
                               'было во было из из во было']
        stop_words = ['во', 'было', 'из']
        expected = []

        actual = extract_candidate_keyword_phrases(tokenized_sentences, stop_words)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_candidate_keyword_phrases_empty_inputs(self):
        tokenized_sentences = ['Во времена', 'Советского Союза']
        stop_words = ['во', 'было', 'из']
        expected = None
        actual = extract_candidate_keyword_phrases(tokenized_sentences, [])
        self.assertEqual(expected, actual)
        actual = extract_candidate_keyword_phrases([], stop_words)
        self.assertEqual(expected, actual)

    @pytest.mark.lab_2_keywords_cooccurrence
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_extract_candidate_keyword_phrases_wrong_input_type(self,):
        stop_words = ['во', 'было', 'из']
        tokenized_sentences = ['во', 'времена', 'советского', 'союза', 'исследование',
                               'космоса', 'было', 'одной', 'из', 'важнейших', 'задач']
        wrong_inputs = [4, 4.0, False, {'dict': 0}, 'string']
        expected = None
        for wrong_input in wrong_inputs:
            actual = extract_candidate_keyword_phrases(wrong_input, stop_words)
            self.assertEqual(expected, actual)
            actual = extract_candidate_keyword_phrases(tokenized_sentences, wrong_input)
            self.assertEqual(expected, actual)
